# Transformations par entité

## Le problème

Sur un panel (`MultiIndex (entité..., date)`), certaines transformations doivent
être **apprises séparément pour chaque entité** : une standardisation par pays,
un encodage propre à chaque produit, une imputation dont les paramètres dépendent
de l'entité. Appliquer un transformateur unique au panel entier mélangerait les
niveaux.

## La solution

`PanelwiseTransformer` enveloppe n'importe quel transformateur sklearn (ou
pipeline) et l'applique **indépendamment à chaque entité**, chacune recevant sa
propre instance ajustée.

Trois modes de paramétrage par entité :

| Mode | `transformer` | Effet |
|------|---------------|-------|
| Simple | un `BaseEstimator` | cloné à l'identique pour chaque entité |
| Fabrique | un `Callable[[entity_key], BaseEstimator]` | une instance construite par entité |
| `entity_kwargs` | un `BaseEstimator` + dict `{entité: kwargs}` | `set_params(**kwargs)` par entité avant `fit` |

```python
from sklearn.preprocessing import StandardScaler
from tsforecast.panel import PanelwiseTransformer

pt = PanelwiseTransformer(transformer=StandardScaler(), panel_cols=["country"])
df_scaled = pt.fit_transform(df)          # un StandardScaler par pays
```

## Points d'attention

- `n_jobs` : parallélisation (joblib) de l'ajustement et de la transformation.
- `error_handling` (`'raise'` / `'warn'` / `'ignore'`) : que faire si une entité
  échoue ; les entités en échec sont listées dans `failed_entities_`.
- État ajusté : `transformers_` (`{entité: transformateur ajusté}`), `entities_`.
- Réversible via `ReversibleTransformerMixin` si le transformateur de base l'est.

C'est cette classe que `PublicationDelayTransformer` utilise en interne pour
traiter un panel.

## Utilitaires panel

`tsforecast.panel.utils` expose les briques de manipulation de panels utilisées
partout dans le package : `is_panel_data()`, `get_unique_panel_entities()`,
`normalize_entity_key()`, `get_entity_mask()`,
`resolve_entity_column_frequencies()`, etc.

## Pour aller plus loin

- Tutoriel : [Transformations par entité](../tutorials/panelwise_transforms.md)
- API : [PanelwiseTransformer](../api/panel/PanelwiseTransformer.md),
  [Utilitaires panel](../api/panel/utils.md)
