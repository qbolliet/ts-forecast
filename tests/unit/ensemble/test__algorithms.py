"""Unit tests of ``tsforecast.ensemble._algorithms``, the pure update kernel of online aggregation.

Scenarios: reference sequences of BOA and MLpol (squared and absolute losses,
linearised or not) computed by an independent implementation; two rounds computed
by hand from the default initial states; losses and gradients; weights on the
simplex; invariance to the scale of the losses; pinball at the median against the
absolute loss; convergence to a dominating expert; vanishing average regret on an
alternating sequence; rounds of several rows; degenerate cases (one expert, empty
sequence).
"""
from dataclasses import replace

import numpy as np
import pytest

from tsforecast.ensemble._algorithms import (
    ALGORITHMS,
    BOAState,
    Loss,
    MLpolState,
    boa_init,
    boa_update,
    boa_weights,
    excess_losses,
    make_loss,
    mlpol_init,
    mlpol_update,
    mlpol_weights,
    run,
)

pytestmark = pytest.mark.internal

# Jeu des valeurs de référence : 8 tours, 3 experts (proche, biaisé, bruité)
REFERENCE_Y = np.array([1.0, 2.5, 0.8, 3.1, 2.2, 1.7, 2.9, 0.4])
REFERENCE_EXPERTS = np.array([
    [1.2, 2.3, 1.0, 2.8, 2.5, 1.5, 3.2, 0.7],
    [2.0, 3.4, 1.9, 4.0, 3.1, 2.8, 3.8, 1.5],
    [0.1, 3.5, -0.5, 2.0, 3.6, 0.5, 2.0, 1.6],
]).T
# Valeur initiale des compteurs de l'implémentation de référence (2^-20), que l'état reçoit par injection
REFERENCE_START = 2.0 ** -20

# Valeurs de référence (algorithme, perte, perte linéarisée) -> poids de chaque tour, prévisions,
# poids finaux. Calculées une fois par le module Python opera-python (paquet `opera`, commit
# 9f8203363ba75206e9beb02002e3420037456200) exécuté hors du dépôt, avec poids initiaux uniformes :
# - MLpol : module tel quel ; il démarre les inverses des taux à 2^20 au lieu de 0 ;
# - BOA : module dont trois lignes de la mise à jour suivent la procédure publiée (étendue
#   2^(k+1) au lieu de 2^k ; cumul régularisé avec les taux du tour précédent ; terme correctif
#   +E·1{eta·l > 1/2} sur l'excès de perte l, au lieu de son opposé sur le regret) ; il démarre
#   taux et sommes de carrés à 2^-20 au lieu de 0.
# Les tests injectent ces valeurs de départ dans l'état initial du noyau.
OPERA_REFERENCES = {
    ('BOA', 'mse', True): {
        'weights': [
            [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            [0.7682314766053957, 0.09412743164894334, 0.13764109174566094],
            [0.6895553846663326, 0.12761015601989542, 0.18283445931377196],
            [0.6743807913365243, 0.11946729628505381, 0.2061519123784218],
            [0.9233898693456893, 0.04772072184956776, 0.02888940880474283],
            [0.9410312674739744, 0.038653622923378525, 0.020315109602647076],
            [0.9371438442609258, 0.043781673773031896, 0.019074481966042254],
            [0.9395702132941631, 0.039128495716501385, 0.021301290989335524],
        ],
        'predictions': [
            1.1, 2.5687094849086307,
            0.8405974514472478, 2.7784392256393264,
            2.560410782794958, 1.529934600197745,
            3.2033796259045686, 0.750473958463603,
        ],
        'final': [0.9535052342019589, 0.030394519549725815, 0.01610024624831525],
    },
    ('BOA', 'mse', False): {
        'weights': [
            [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            [0.9402545108795528, 0.0291633990329627, 0.030582090087484616],
            [0.9442247880747251, 0.028475522172897113, 0.027299689752377822],
            [0.9820239802770061, 0.00973247764074558, 0.0082435420822485],
            [0.9839327334187051, 0.00898956009944036, 0.007077706481854486],
            [0.987545761809352, 0.00767992982276153, 0.004774308367886372],
            [0.9896815880393296, 0.006477470885510037, 0.003840941075160403],
            [0.9906800881008421, 0.005850653977803728, 0.0034692579213542055],
        ],
        'predictions': [
            1.1, 2.3687782470412406,
            0.9846784353270407, 2.805084139503096,
            2.513179213189704, 1.5052096004017035,
            3.1992773532411136, 0.7078028553114617,
        ],
        'final': [0.9925835468951455, 0.004735727206771003, 0.0026807258980834276],
    },
    ('BOA', 'mae', True): {
        'weights': [
            [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            [0.7587525724856827, 0.09250234699066087, 0.1487450805236565],
            [0.5523337484775467, 0.1766235099532608, 0.27104274156919245],
            [0.7633971382247309, 0.1528502351112145, 0.08375262666405474],
            [0.7476378001555425, 0.1763269728494119, 0.07603522699504574],
            [0.7762155756574303, 0.16021053641931887, 0.06357388792325085],
            [0.7561485340065829, 0.1878827014995701, 0.05596876449384686],
            [0.7661055362219932, 0.1721290252201405, 0.061765438557866315],
        ],
        'predictions': [
            1.1, 2.580246678318115,
            0.752397046604146, 2.916418180802214,
            2.689434933404198, 1.6446998094218637,
            3.2455671035071254, 0.893292114878192,
        ],
        'final': [0.7951274897596076, 0.15142951943482252, 0.05344299080556985],
    },
    ('BOA', 'mae', False): {
        'weights': [
            [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            [0.8019801202932261, 0.09777237805460721, 0.1002475016521669],
            [0.5871090549543434, 0.20882504283334244, 0.20406590221231405],
            [0.7928771373895708, 0.1068991371556459, 0.10022372545478335],
            [0.8022223267206, 0.10377155629948405, 0.094006116979916],
            [0.8334393102904456, 0.09086763083860716, 0.07569305887094713],
            [0.8478943892342284, 0.0836862968848285, 0.068419313880943],
            [0.8551675198403033, 0.07968472610701303, 0.0651477540526839],
        ],
        'predictions': [
            1.1, 2.527846617842669,
            0.881843685231537, 2.8480999842229484,
            2.6656696624575984, 1.542434861219242,
            3.168108601473765, 0.822380759533026,
        ],
        'final': [0.8781289571982368, 0.06756539514407195, 0.054305647657691275],
    },
    ('MLpol', 'mse', True): {
        'weights': [
            [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            [0.0, 0.0, 1.0],
            [0.9153841930534358, 0.0076923458699741775, 0.07692346107659015],
            [0.8384457710245686, 0.0, 0.16155422897543134],
            [0.7167557254923628, 0.2832442745076373, 0.0],
            [0.8213281480681117, 0.17867185193188836, 0.0],
            [0.8401530244324265, 0.15984697556757352, 0.0],
            [0.9268930532529739, 0.03549964038760317, 0.03760730635942285],
        ],
        'predictions': [
            1.1, 3.5,
            0.8915379196680917, 2.6707566168196544,
            2.6699465647045826, 1.732273407511455,
            3.295908185340544, 0.762246288033563,
        ],
        'final': [1.0, 0.0, 0.0],
    },
    ('MLpol', 'mse', False): {
        'weights': [
            [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        'predictions': [
            1.1, 3.0666666666666664,
            1.0, 2.8,
            2.5, 1.5,
            3.2, 0.7,
        ],
        'final': [1.0, 0.0, 0.0],
    },
    ('MLpol', 'mae', True): {
        'weights': [
            [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            [0.0, 0.0, 1.0],
            [0.5238094167647185, 0.0, 0.4761905832352815],
            [0.6381911429058039, 0.2864318759955721, 0.075376981098624],
            [0.46934203083197035, 0.5306579691680298, 0.0],
            [0.5606336246254782, 0.4393663753745219, 0.0],
            [0.770625151128295, 0.22937484887170506, 0.0],
            [0.6278710600248157, 0.06334133362758186, 0.30878760634760244],
        ],
        'predictions': [
            1.1, 3.5,
            0.2857141251470777, 3.083416666315787,
            2.8183947815008183, 2.0711762879868783,
            3.3376249093230235, 1.0285819126149076,
        ],
        'final': [0.8078399903478063, 0.0, 0.19216000965219368],
    },
    ('MLpol', 'mae', False): {
        'weights': [
            [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        'predictions': [
            1.1, 3.0666666666666664,
            1.0, 2.8,
            2.5, 1.5,
            3.2, 0.7,
        ],
        'final': [1.0, 0.0, 0.0],
    },
}


# Fonction de construction de l'état de départ de l'implémentation de référence
def _reference_state(algorithm: str):
    """Initial state of ``algorithm`` with the starting values of the reference implementation."""
    state = ALGORITHMS[algorithm].init(np.ones(3))
    if algorithm == 'BOA':
        return replace(state, learning_rates=np.full(3, REFERENCE_START), sq_sums=np.full(3, REFERENCE_START))
    return replace(state, inverse_rates=np.full(3, 1.0 / REFERENCE_START))


class TestReferenceSequences:
    """BOA and MLpol reproduce the reference sequences round by round."""

    @pytest.mark.parametrize('key', sorted(OPERA_REFERENCES), ids=lambda key: '-'.join(map(str, key)))
    def test_weights_predictions_and_final_weights(self, key):
        algorithm, loss, gradient = key
        reference = OPERA_REFERENCES[key]
        state, weights, prediction = run(
            algorithm, _reference_state(algorithm), REFERENCE_EXPERTS, REFERENCE_Y, np.arange(8),
            make_loss(loss), gradient,
        )
        np.testing.assert_allclose(weights, reference['weights'], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(prediction, reference['predictions'], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(ALGORITHMS[algorithm].weights(state), reference['final'], rtol=1e-12, atol=1e-15)


class TestHandComputedRounds:
    """Rounds computed by hand from the default initial states."""

    def test_boa_three_rounds(self):
        state = boa_init(np.ones(2))
        np.testing.assert_allclose(boa_weights(state), [0.5, 0.5])
        # Tour 1 : étendue 2, taux précédents nuls, taux min(1/2, sqrt(log 2 / 1)) = 1/2
        state = boa_update(state, np.array([-1.0, 1.0]))
        np.testing.assert_allclose(state.losses, [-0.5, 0.5])
        np.testing.assert_allclose(state.learning_rates, [0.5, 0.5])
        expected = np.array([np.exp(0.25), np.exp(-0.25)])
        np.testing.assert_allclose(boa_weights(state), expected / expected.sum())
        # Tour 2 : étendues (2, 4) ; L1 += (0.5 * (1 + 0.25)) / 2 ; L2 += -2 * (1 - 1) / 2
        state = boa_update(state, np.array([0.5, -2.0]))
        np.testing.assert_allclose(state.losses, [-0.1875, 0.5])
        np.testing.assert_allclose(state.sq_sums, [1.25, 5.0])
        np.testing.assert_allclose(state.learning_rates, [0.5, 0.25])
        expected = np.array([0.5 * np.exp(0.5 * 0.1875), 0.25 * np.exp(-0.25 * 0.5)])
        np.testing.assert_allclose(boa_weights(state), expected / expected.sum())
        # Tour 3 : eta·l = 1 > 1/2 pour l'expert 1, terme correctif d'étendue 4
        state = boa_update(state, np.array([2.0, 0.0]))
        np.testing.assert_allclose(state.losses, [-0.1875 + 0.5 * (2.0 * 2.0 + 4.0), 0.5])

    def test_mlpol_two_rounds(self):
        state = mlpol_init(np.ones(2))
        np.testing.assert_allclose(mlpol_weights(state), [0.5, 0.5])
        # Tour 1 : regrets (1, -2), B² = 4, inverses des taux (1 + 4, 4 + 4)
        state = mlpol_update(state, np.array([-1.0, 2.0]))
        np.testing.assert_allclose(state.inverse_rates, [5.0, 8.0])
        np.testing.assert_allclose(mlpol_weights(state), [1.0, 0.0])
        # Tour 2 : regrets (0.5, 3), B² = 9 (+5), regrets cumulés (1.5, 1)
        state = mlpol_update(state, np.array([-0.5, -3.0]))
        np.testing.assert_allclose(state.regrets, [1.5, 1.0])
        np.testing.assert_allclose(state.inverse_rates, [10.25, 22.0])
        expected = np.array([1.5 / 10.25, 1.0 / 22.0])
        np.testing.assert_allclose(mlpol_weights(state), expected / expected.sum())

    def test_mlpol_fallback_weights_until_a_positive_regret(self):
        state = mlpol_init(np.array([1.0, 3.0]))
        np.testing.assert_allclose(mlpol_weights(state), [0.25, 0.75])
        # Regrets nuls : poids de repli conservés
        state = mlpol_update(state, np.zeros(2))
        np.testing.assert_allclose(mlpol_weights(state), [0.25, 0.75])

    def test_boa_prior_shapes_the_initial_weights(self):
        np.testing.assert_allclose(boa_weights(boa_init(np.array([1.0, 3.0]))), [0.25, 0.75])

    def test_updates_return_new_states(self):
        state = boa_init(np.ones(2))
        updated = boa_update(state, np.array([1.0, -1.0]))
        assert isinstance(updated, BOAState) and updated is not state
        np.testing.assert_array_equal(state.losses, [0.0, 0.0])
        initial = mlpol_init(np.ones(2))
        assert isinstance(mlpol_update(initial, np.array([1.0, -1.0])), MLpolState)
        np.testing.assert_array_equal(initial.regrets, [0.0, 0.0])


class TestLosses:
    """Predefined losses, their gradients and their validation."""

    def test_values_and_gradients(self):
        prediction, y = np.array([1.0, 3.0]), np.array([2.0, 2.0])
        mse, mae = make_loss('mse'), make_loss('mae')
        np.testing.assert_allclose(mse.value(prediction, y), [1.0, 1.0])
        np.testing.assert_allclose(mse.gradient(prediction, y), [-2.0, 2.0])
        np.testing.assert_allclose(mae.value(prediction, y), [1.0, 1.0])
        np.testing.assert_allclose(mae.gradient(prediction, y), [-1.0, 1.0])

    def test_pinball(self):
        pinball = make_loss('pinball', quantile=0.9)
        prediction, y = np.array([1.0, 3.0]), np.array([2.0, 2.0])
        # Sous-prévision pénalisée par 0.9, sur-prévision par 0.1
        np.testing.assert_allclose(pinball.value(prediction, y), [0.9, 0.1])
        np.testing.assert_allclose(pinball.gradient(prediction, y), [-0.9, 0.1])

    def test_custom_loss_is_returned_unchanged(self):
        loss = Loss(value=np.subtract, gradient=np.add)
        assert make_loss(loss) is loss

    @pytest.mark.parametrize('loss, quantile', [('huber', 0.5), ('pinball', 0.0), ('pinball', 1.0), ('pinball', True)])
    def test_invalid_loss(self, loss, quantile):
        with pytest.raises(ValueError):
            make_loss(loss, quantile)


class TestExcessLosses:
    """Excess losses of one round, linearised or not, averaged over the rows of the round."""

    def test_linearised_and_plain(self):
        experts, y, weights = np.array([[1.0, 3.0]]), np.array([1.0]), np.array([0.5, 0.5])
        prediction, linear = excess_losses(weights, experts, y, make_loss('mse'), True)
        # Gradient 2·(2 - 1) = 2 en la prévision 2, puis 2·(x - 2)
        np.testing.assert_allclose(prediction, [2.0])
        np.testing.assert_allclose(linear, [-2.0, 2.0])
        _, plain = excess_losses(weights, experts, y, make_loss('mse'), False)
        np.testing.assert_allclose(plain, [0.0 - 1.0, 4.0 - 1.0])

    @pytest.mark.parametrize('gradient', [True, False])
    def test_mean_over_the_rows_of_a_round(self, gradient):
        rng = np.random.default_rng(0)
        experts, y, weights = rng.normal(size=(4, 3)), rng.normal(size=4), np.array([0.2, 0.3, 0.5])
        _, excess = excess_losses(weights, experts, y, make_loss('mae'), gradient)
        rows = [excess_losses(weights, experts[i:i + 1], y[i:i + 1], make_loss('mae'), gradient)[1] for i in range(4)]
        np.testing.assert_allclose(excess, np.mean(rows, axis=0))


# Fonction de déroulement d'une séquence d'un tour par ligne
def _play(algorithm, experts, y, loss='mse', gradient=True, prior=None):
    """Play one round per row from the default initial state; return ``(state, weights, prediction)``."""
    prior = np.ones(experts.shape[1]) if prior is None else prior
    return run(algorithm, ALGORITHMS[algorithm].init(prior), experts, y, np.arange(len(y)), make_loss(loss), gradient)


# Combinaisons algorithme × perte × perte linéarisée
COMBINATIONS = [
    (algorithm, loss, gradient)
    for algorithm in ALGORITHMS
    for loss in ('mse', 'mae', 'pinball')
    for gradient in (True, False)
]


class TestProperties:
    """Properties of the weights whatever the sequence."""

    @pytest.mark.parametrize('algorithm, loss, gradient', COMBINATIONS)
    def test_weights_on_the_simplex(self, algorithm, loss, gradient):
        rng = np.random.default_rng(1)
        y = rng.normal(size=60)
        experts = y[:, None] + rng.normal(scale=[0.1, 1.0, 3.0, 0.5], size=(60, 4))
        state, weights, _ = _play(algorithm, experts, y, loss, gradient)
        final = ALGORITHMS[algorithm].weights(state)
        for vector in (*weights, final):
            assert np.all(vector >= 0)
            np.testing.assert_allclose(vector.sum(), 1.0)

    @pytest.mark.parametrize('algorithm, factor', [('MLpol', 10.0), ('BOA', 4.0)])
    @pytest.mark.parametrize('loss, gradient', [('mse', True), ('mse', False), ('mae', True)])
    def test_invariance_to_the_scale_of_the_target(self, algorithm, factor, loss, gradient):
        # MLpol : toute échelle ; BOA : une puissance de 2 (étendues estimées en puissances de 2)
        rng = np.random.default_rng(2)
        y = rng.normal(size=40)
        experts = y[:, None] + rng.normal(scale=[0.3, 1.0, 2.0], size=(40, 3))
        _, weights, _ = _play(algorithm, experts, y, loss, gradient)
        _, scaled, _ = _play(algorithm, factor * experts, factor * y, loss, gradient)
        np.testing.assert_allclose(scaled, weights, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize('algorithm', ALGORITHMS)
    @pytest.mark.parametrize('gradient', [True, False])
    def test_pinball_at_the_median_weighs_like_the_absolute_loss(self, algorithm, gradient):
        rng = np.random.default_rng(3)
        y = rng.normal(size=40)
        experts = y[:, None] + rng.normal(scale=[0.3, 1.0, 2.0], size=(40, 3))
        _, absolute, _ = _play(algorithm, experts, y, 'mae', gradient)
        _, pinball, _ = run(
            algorithm, ALGORITHMS[algorithm].init(np.ones(3)), experts, y, np.arange(40),
            make_loss('pinball', quantile=0.5), gradient,
        )
        np.testing.assert_allclose(pinball, absolute, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize('algorithm, loss, gradient', COMBINATIONS)
    def test_convergence_to_a_dominating_expert(self, algorithm, loss, gradient):
        rng = np.random.default_rng(4)
        y = rng.normal(size=300)
        # Biais de même signe : aucune combinaison convexe ne fait mieux que l'expert sans biais
        experts = y[:, None] + np.array([0.0, 1.0, 2.0]) + rng.normal(scale=0.1, size=(300, 3))
        state, _, _ = _play(algorithm, experts, y, loss, gradient)
        assert ALGORITHMS[algorithm].weights(state)[0] > 0.9

    @pytest.mark.parametrize('algorithm', ALGORITHMS)
    def test_average_regret_vanishes_on_an_alternating_sequence(self, algorithm):
        # Cible alternant 0 et 1 : les experts 0 et 1 se trompent un tour sur deux, le troisième (0.5) est le meilleur
        n = 400
        y = np.arange(n) % 2.0
        experts = np.column_stack([np.zeros(n), np.ones(n), np.full(n, 0.5)])
        _, _, prediction = _play(algorithm, experts, y)
        regret = np.cumsum((prediction - y) ** 2) - np.cumsum((experts[:, 2] - y) ** 2)
        assert regret[-1] / n < 0.02
        assert regret[-1] / n < regret[n // 4 - 1] / (n // 4)


class TestRuns:
    """Sequences of rounds: several rows per round, a single expert, an empty sequence."""

    @pytest.mark.parametrize('algorithm', ALGORITHMS)
    def test_rows_of_a_round_share_the_weights_and_one_update(self, algorithm):
        rng = np.random.default_rng(5)
        experts, y = rng.normal(size=(6, 3)), rng.normal(size=6)
        rounds = np.array([0, 0, 1, 2, 2, 2])
        functions = ALGORITHMS[algorithm]
        state, weights, _ = run(algorithm, functions.init(np.ones(3)), experts, y, rounds, make_loss('mse'), True)
        # Séquence rejouée à la main : une mise à jour par tour, sur l'excès de perte moyen
        manual = functions.init(np.ones(3))
        for rows in (slice(0, 2), slice(2, 3), slice(3, 6)):
            current = functions.weights(manual)
            np.testing.assert_allclose(weights[rows], np.broadcast_to(current, weights[rows].shape))
            manual = functions.update(manual, excess_losses(current, experts[rows], y[rows], make_loss('mse'), True)[1])
        np.testing.assert_allclose(functions.weights(state), functions.weights(manual))

    @pytest.mark.parametrize('algorithm', ALGORITHMS)
    def test_single_expert(self, algorithm):
        rng = np.random.default_rng(6)
        state, weights, _ = _play(algorithm, rng.normal(size=(5, 1)), rng.normal(size=5))
        np.testing.assert_allclose(weights, 1.0)
        np.testing.assert_allclose(ALGORITHMS[algorithm].weights(state), [1.0])
        assert all(np.all(np.isfinite(np.asarray(value, dtype=float))) for value in vars(state).values())

    def test_empty_sequence(self):
        state = boa_init(np.ones(2))
        final, weights, prediction = run('BOA', state, np.empty((0, 2)), np.empty(0), np.empty(0, dtype=int), make_loss('mse'), True)
        assert final is state and weights.shape == (0, 2) and prediction.shape == (0,)
