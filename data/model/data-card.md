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
- Strict unscaled cell artifact ready: True
- Strict unscaled drug artifact ready: True
- Raw cell preprocessing status: `{"methylation_source": "data\\processed\\cell_line_omics\\methylation_raw.csv", "pathway_source": "data\\processed\\driver&pathway\\pathway_activity_raw.npy", "strict_unscaled": true}`

## Source checksums

- `data/processed/drug_sensitivity/ic50_matrix.csv`: `bb0e6f282524eefd725572722c54a1c770c2324d63d638978e9f59f7d34f4e0b`
- `data/processed/protein_protein_interaction/ppi_dg_filtered.csv`: `0c010d118c453ba8048d612b8281d2998560c41a15cd2e35ae07e5ac6106c07f`
- `data/processed/drug_gene_interaction/interactions_filtered.csv`: `c63e9474ea014fef5bf66ac6de45a50c8398ee6abb4d9fb8de1b133788bcd9e8`
- `data/model/hetero_graph/hetero_graph_base.pt`: `1bf1b7ac1d5a96411893029cc772bcb6fbc301faf376e7078b3c64cd32dc5d52`
- `data/model/hetero_graph/cell_line_features.pt`: `b59fdba3bc844fe673707a7caf12472542f23e72ce998d751694e924651822fe`
- `data/model/splits/eval-db/manifest.json`: `4498dfed9f133a0ba7ef2561ab8625380214626f2a1cd065503ebc34ef5af61f`
- `data/model/splits/eval-lco/manifest.json`: `cdeedd0340a356283774e059840c2a70d71aa86b03c9b66ca739d9c49e6e2054`
- `data/model/splits/eval-ldo-kt/manifest.json`: `0bcac945ccecfb4b928926843835d61fe217ecd1a129d5227e5b251f6839910a`
- `data/model/splits/eval-ldo-so/manifest.json`: `2682d9c72d53513b06dfbae6ddec4090934707538c495e67c3222c5876b4326a`
- `data/model/splits/eval-lpo/manifest.json`: `a6766a047e463ef812ea83d519b0c1d9f1f0350813e87f7651702e7b82a159af`
- `data/model/splits/eval-lto/manifest.json`: `159013ed97ddcdb853c59438c9b652897b9b8119d7c2be03e121581017dcc557`

## Leakage policy

All response-dependent preprocessing and continuous-feature scaling in Ver2 are fitted on the training fold only. Mutation remains a discrete modality. Pathway scores are generated without response labels and standardized using the training fold.

Legacy globally standardized artifacts remain only for old-checkpoint compatibility and are rejected by formal Ver2 runs.
If either strict artifact readiness flag above is false, run the remote preprocessing sequence before any formal experiment. Smoke mode alone may use the legacy fallback and its metrics are not scientific results.
