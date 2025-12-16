import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    df = pd.read_csv("data/processed/model_ready_data.csv")

    FEATURES = ["studytime", "absences", "failures"]
    TARGET = "AtRisk"

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Model accuracy: {acc:.2f}")

    with open("models/random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(" Model saved to models/random_forest_model.pkl")

if __name__ == "__main__":
    main()
