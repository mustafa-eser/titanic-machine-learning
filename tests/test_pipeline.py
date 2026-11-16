import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("titanic_model", ROOT / "titanic/titanic.py")
titanic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(titanic)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.train = pd.read_csv(ROOT / "titanic/train.csv")
        self.test = pd.read_csv(ROOT / "titanic/test.csv")

    def test_evaluation_fits_on_split_before_final_refit(self):
        model = titanic.build_pipeline()
        with mock.patch.object(titanic, "build_pipeline", return_value=model):
            with mock.patch.object(model, "fit", wraps=model.fit) as fit:
                titanic.train_and_predict(self.train, self.test)
        self.assertEqual(fit.call_count, 2)
        split_features, split_target = fit.call_args_list[0].args
        full_features, full_target = fit.call_args_list[1].args
        self.assertEqual(len(split_features), int(0.8 * len(self.train)))
        self.assertEqual(split_features.index.tolist(), split_target.index.tolist())
        self.assertNotIn("Survived", split_features.columns)
        self.assertEqual(len(full_features), len(self.train))
        pd.testing.assert_series_equal(full_target, self.train["Survived"])

    def test_imputation_uses_training_rows_only(self):
        model = titanic.build_pipeline()
        features = self.train.drop(columns="Survived")
        model.fit(features, self.train["Survived"])
        imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
        before = imputer.statistics_.copy()
        np.testing.assert_allclose(before, features[titanic.NUMERIC_FEATURES].median().to_numpy())
        held_out = self.test.copy()
        held_out["Age"] = 10000
        held_out["Fare"] = np.nan
        model.predict(held_out)
        np.testing.assert_array_equal(imputer.statistics_, before)

    def test_prediction_handles_unseen_categories_without_refitting(self):
        model = titanic.build_pipeline()
        model.fit(self.train.drop(columns="Survived"), self.train["Survived"])
        encoder = model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encode"]
        before = [values.copy() for values in encoder.categories_]
        unseen = self.test.iloc[:3].copy()
        unseen["Embarked"] = "UNSEEN_PORT"
        self.assertEqual(len(model.predict(unseen)), 3)
        for old, current in zip(before, encoder.categories_):
            np.testing.assert_array_equal(old, current)

    def test_submission_schema_order_and_reproducibility(self):
        original = self.train.copy(deep=True)
        score, submission = titanic.train_and_predict(self.train, self.test)
        repeated_score, repeated_submission = titanic.train_and_predict(self.train, self.test)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertEqual(score, repeated_score)
        pd.testing.assert_frame_equal(submission, repeated_submission)
        pd.testing.assert_frame_equal(self.train, original)
        self.assertEqual(list(submission.columns), ["PassengerId", "Survived"])
        self.assertEqual(submission["PassengerId"].tolist(), self.test["PassengerId"].tolist())
        self.assertTrue(set(submission["Survived"]).issubset({0, 1}))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.csv"
            submission.to_csv(output, index=False)
            pd.testing.assert_frame_equal(pd.read_csv(output), submission)


if __name__ == "__main__":
    unittest.main()
