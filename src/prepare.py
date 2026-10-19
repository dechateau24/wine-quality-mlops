import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

TARGET = "quality"


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["split"]

    df = pd.read_csv("data/raw/wine_quality.csv")

    # Les doublons gonfleraient le score si un meme vin est dans train ET test
    n_avant = len(df)
    df = df.drop_duplicates()
    print(f"[prepare] Doublons supprimes : {n_avant - len(df)}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print(f"[prepare] Train : {len(X_train)} | Test : {len(X_test)}")


if __name__ == "__main__":
    main()