# Classes de base des transformateurs

`tsforecast.base.transformers` fournit les classes abstraites sur lesquelles
reposent tous les transformateurs du package. On les utilise pour **écrire un
nouveau transformateur temporel** cohérent avec le reste de la boîte à outils.

## Les trois briques

### `TimeSeriesTransformerMixin`

Mixin de gestion de l'index temporel : extraction et validation d'un
`DatetimeIndex` à partir de formats variés (index, colonne `time_col`,
`MultiIndex`), quelle que soit la structure d'entrée.

### `PanelTimeSeriesTransformer`

Classe de base abstraite (hérite de `BaseEstimator`, `TransformerMixin`) pour un
transformateur qui accepte **aussi bien une série qu'un panel**. Elle prend en
charge :

- la validation temporelle (`validate_temporal_data`) et la restauration de la
  structure d'origine (`restore_original_structure`) en sortie ;
- la vérification du regroupement et du tri intra-groupe pour les panels ;
- les paramètres transverses `time_col`, `panel_cols`, `validate_input`,
  `auto_sort`.

Les sous-classes implémentent `_fit` et `_transform`.

### `ReversibleTransformerMixin`

Mixin pour les transformations réversibles : impose le contrat
`inverse_transform` et son articulation avec `_inverse_transform`.

## Rapport avec les autres modules

```
PanelTimeSeriesTransformer
├── XYTransformerMixin ........... transforme X et y (module xy)
├── PanelwiseTransformer ......... applique un transfo. par entité (module panel)
└── PublicationDelayTransformer .. délais de publication (module delays)
```

`HighFrequencyImputer` hérite de `XYPanelTimeSeriesTransformer`
(`PanelTimeSeriesTransformer` + `XYTransformerMixin`).

## Pour aller plus loin

- API : [Transformateurs de base](../api/base/transformers.md)
- Concept lié : [Pipeline XY](xy_pipeline.md)
- Validation temporelle : [Utilitaires temporels](temporal_utils.md)
