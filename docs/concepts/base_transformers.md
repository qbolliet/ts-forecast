# Classes de base des transformateurs

`tsforecast.base.transformers` fournit les classes abstraites sur lesquelles
reposent les transformateurs temporels du package. On les utilise pour **écrire un
nouveau transformateur temporel** cohérent avec le reste de la boîte à outils.

## Les quatre briques

### `TimeSeriesTransformerMixin`

Mixin de gestion de l'index temporel : extraction d'un `DatetimeIndex` à partir
d'une colonne `time_col`, de l'index, ou du dernier niveau d'un `MultiIndex`. Les
étiquettes acceptées sont celles de `validate_temporal_data` : dates, chaînes de caractères
représentant des dates et périodes (lues à leur premier instant) ; les étiquettes numériques
(`RangeIndex`, années entières) sont refusées.

### `PanelTimeSeriesTransformer`

Classe de base abstraite (hérite de `BaseEstimator`, `TransformerMixin`) pour un
transformateur qui accepte **aussi bien une série qu'un panel**. Les sous-classes
implémentent `_fit` et `_transform` ; `fit` et `transform` prennent en charge :

- l'entrée : un `DataFrame` à deux dimensions, comme dans sklearn (dates dans
  l'index ou dans `time_col`, entités dans les premiers niveaux d'un `MultiIndex`
  ou dans `panel_cols`) ; une série seule se passe sous la forme
  `series.to_frame()` ;
- la validation temporelle (`validate_temporal_data`) : vérifie les dates, les doublons d'index et de colonnes, tri global
  et intra-entité, regroupement des entités ; erreur sous `strict_validation=True`,
  avertissement sinon ; tri sur demande (`auto_sort=True`) ;
- la non-modification des données de l'appelant (conversions sur des copies) ;
- les attributs ajustés de la convention sklearn (`is_panel_`, `n_features_in_`,
  `feature_names_in_`) et, au `transform`, le contrôle des colonnes vues au `fit`
  (mêmes noms, même ordre ; une sous-classe au contrat différent, comme
  `HighFrequencyImputer`, surcharge `_check_feature_names`) ;
- `convert_cols_to_index=True` : structure des colonnes `time_col` et `panel_cols`
  passées dans l'index, figée au `fit` (`conversion_metadata_`), qui permet de les
  rendre ; les lignes rendues sont numérotées à partir de 0.

### `ReversibleTransformerMixin`

Mixin des transformations réversibles : déclare `inverse_transform` comme méthode
abstraite (une sous-classe qui ne l'implémente pas ne s'instancie pas) et fournit
deux aides, `_store_transformation_info` (formes, colonnes et index avant / après)
et `_restore_structure_if_converted`, qui rend la structure du `fit` : périodes de
l'index et colonnes passées dans l'index.

### `InverseTransformer`

Enveloppe d'un transformateur **déjà ajusté** qui échange `transform` et
`inverse_transform`. Comme `sklearn.frozen.FrozenEstimator`, c'est une étape
pré-ajustée : `fit` ne fait rien et `clone` renvoie l'enveloppe elle-même, ce qui
la rend utilisable dans `Pipeline`, `cross_validate` ou `GridSearchCV`.

## Rapport avec les autres modules

```
PanelTimeSeriesTransformer
├── PanelwiseTransformer ............ un transformateur par entité (module panel),
│                                     avec ReversibleTransformerMixin
└── XYPanelTimeSeriesTransformer .... transforme X et y (module xy)
    └── HighFrequencyImputer ........ imputation multi-fréquences (module frequency)
```

`XYTransformerMixin` (module xy) et les transformateurs du module `delays`
(`PublicationDelayTransformer`, `ShiftTransformer`, `MaskTransformer`) ne dérivent
pas de `PanelTimeSeriesTransformer` ; ils s'appliquent à un panel par
`PanelwiseTransformer`.

## Pour aller plus loin

- API : [Transformateurs de base](../api/base/transformers.md)
- Concept lié : [Pipeline XY](xy_pipeline.md)
- Validation temporelle : [Utilitaires temporels](temporal_utils.md)
