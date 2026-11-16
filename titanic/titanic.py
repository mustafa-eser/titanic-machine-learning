"""Train and evaluate a Titanic classifier without fitting on held-out data."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["Pclass", "SibSp", "Parch", "Fare", "Age"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


def build_pipeline():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classify", LogisticRegression(random_state=0, max_iter=1000)),
    ])


def train_and_predict(training_data, test_data):
    features = training_data.drop(columns="Survived")
    target = training_data["Survived"]
    x_train, x_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=45, stratify=target
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    score = accuracy_score(y_val, model.predict(x_val))

    # Refit on all labelled rows only after measuring held-out accuracy.
    model.fit(features, target)
    predictions = model.predict(test_data)
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"].to_numpy(),
        "Survived": predictions,
    })
    return score, submission


def main():
    data_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=data_dir / "train.csv")
    parser.add_argument("--test", type=Path, default=data_dir / "test.csv")
    parser.add_argument("--output", type=Path, default=data_dir / "submission.csv")
    args = parser.parse_args()
    score, submission = train_and_predict(
        pd.read_csv(args.train), pd.read_csv(args.test)
    )
    submission.to_csv(args.output, index=False)
    print(f"Validation accuracy: {score:.4f}")
    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
