# Legacy evaluation status

The existing files named `generalization_report.txt` and
`generalization_report.json` were produced by applying checkpoints trained on
the random sample-pair split to the complete response pool and then grouping
predictions by drug or cancer type.

They remain as historical artifacts, but they are **not** strict LODO, LCO, or
cold-start results. Ver2 conclusions must use independently retrained folds
from `data/model/splits/eval-*` through `program/run_strict_experiment.py`.
