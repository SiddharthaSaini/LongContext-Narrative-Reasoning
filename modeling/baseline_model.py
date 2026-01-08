from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

from reasoning.claim_parser import ClaimParser
from reasoning.signals import detect_contradictions


class BaselineConsistencyModel:
    def __init__(self):
        self.parser = ClaimParser()
        self.model = LogisticRegression()

    def _extract_features(self, text):
        """
        Convert reasoning signals into numerical features
        """
        claims = self.parser.split_into_claims(text)
        signals = detect_contradictions(claims)

        features = {
            "num_claims": len(claims),
            "num_contradictions": len(signals)
        }

        return np.array(list(features.values()))

    def train(self, df):
        X = np.vstack(df["content"].apply(self._extract_features))
        y = df["label"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_val)

        print("\nValidation Results:")
        print(classification_report(y_val, preds))

    def predict(self, text):
        features = self._extract_features(text).reshape(1, -1)
        return self.model.predict(features)[0]
