# Data quality report

## Canonical cohort

- Observed response pairs: 66,543
- Cell lines: 404
- Drugs: 184
- Tissues: 35
- Duplicate drug-cell pairs: 0
- Missing/non-finite responses: 0
- Response mean/std: 2.7740 / 2.8996
- Response min/median/max: -8.5639 / 3.2520 / 13.0915

## Graph and features

- Gene nodes: 8,412
- Drug nodes: 184
- PPI directed records: 672,784
- DTI edges: 1,124
- expression: shape=(404, 8412), finite=True
- mutation: shape=(404, 8412), finite=True
- copynumber: shape=(404, 8412), finite=True
- methylation: shape=(404, 8412), finite=True
- pathway_activity: shape=(404, 186), finite=True
- Strict unscaled cell artifact ready: False
- Strict unscaled drug artifact ready: False
- Raw cell preprocessing status: `{}`

## Source checksums

- `data/processed/drug_sensitivity/ic50_matrix.csv`: `bb0e6f282524eefd725572722c54a1c770c2324d63d638978e9f59f7d34f4e0b`
- `data/processed/protein_protein_interaction/ppi_dg_filtered.csv`: `0c010d118c453ba8048d612b8281d2998560c41a15cd2e35ae07e5ac6106c07f`
- `data/processed/drug_gene_interaction/interactions_filtered.csv`: `c63e9474ea014fef5bf66ac6de45a50c8398ee6abb4d9fb8de1b133788bcd9e8`
- `data/model/hetero_graph/hetero_graph_base.pt`: `1bf1b7ac1d5a96411893029cc772bcb6fbc301faf376e7078b3c64cd32dc5d52`
- `data/model/hetero_graph/cell_line_features.pt`: `2108d46ecb01c87fbd1abfed6385f506b26dd0a6f99fcbb9bc717f5922ccdd7b`
- `data/model/splits/eval-db/manifest.json`: `42f66903fbd08f1de094fa54e0f7e15e8d82ec8dc802b546ce97d8011dc8ff9b`
- `data/model/splits/eval-lco/manifest.json`: `2d3d93a29c843c5ac7d6e62f8182c177340886a4c2a6bbd2949f16f084ef9368`
- `data/model/splits/eval-ldo-kt/manifest.json`: `179b7ff93a2c2afdbfe7027873731552a3a28a7f11e251405ea19b5b978b72b2`
- `data/model/splits/eval-ldo-so/manifest.json`: `1611aba793b1ee710d9fda9f69b9d3ddce0df9038bab38265b9265b9828cd92b`
- `data/model/splits/eval-lpo/manifest.json`: `98ded79904740c608a3e93402b78c3091692535448ce2ca70c0c353717f4e02b`
- `data/model/splits/eval-lto/manifest.json`: `8616bb61605f762757b6cb2da6c809aa1a9005427f19014a747a3dce333063b7`

## Leakage policy

All response-dependent preprocessing and continuous-feature scaling in Ver2 are fitted on the training fold only. Mutation remains a discrete modality. Pathway scores are generated without response labels and standardized using the training fold.

Legacy globally standardized artifacts remain only for old-checkpoint compatibility and are rejected by formal Ver2 runs.
If either strict artifact readiness flag above is false, run the remote preprocessing sequence before any formal experiment. Smoke mode alone may use the legacy fallback and its metrics are not scientific results.
