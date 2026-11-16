# Titanic survival classification

A logistic regression baseline for the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).

## Run

Use Python 3.11 or newer in a virtual environment:

```shell
python -m pip install -r requirements.txt
python titanic/titanic.py
python -m unittest discover -s tests
```

The script locates the bundled input files relative to its own location, prints
held-out validation accuracy and writes `titanic/submission.csv`. Override paths
with `--train`, `--test` and `--output`. Existing output files are overwritten.

## Evaluation

A stratified split with seed 45 reserves 20% of labelled data for validation.
Numeric imputation, scaling and categorical encoding are fitted only on the
training split. Validation and competition test rows are transformed using those
fitted values. Unseen categories are accepted without refitting the encoder.
After evaluation, a model is fitted on all labelled rows to generate the final
submission. The validation score is a local estimate, not a Kaggle leaderboard
score, and no accuracy improvement is promised.

Tests cover training-only imputation, unseen categories, deterministic output,
input preservation and submission columns and passenger ordering.

Licensed under MIT; see [LICENSE](LICENSE).
