import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from reasoning.claim_parser import ClaimParser
from reasoning.signals import detect_contradictions
from reasoning.scoring import contradiction_score


class BaselineConsistencyModel:
    def __init__(self):
        self.parser = ClaimParser()
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )

    def _extract_features(self, text):
        claims = self.parser.split_into_claims(text)
        hard = detect_contradictions(claims)
        soft = contradiction_score(claims)

        return np.array([
            len(claims),        # narrative density
            len(hard),          # explicit contradictions
            soft                # semantic contradiction strength
        ], dtype=float)

    def train(self, df):
        X = np.vstack(df["content"].apply(self._extract_features))
        y = df["label"].values

        # scale features (CRITICAL FIX)
        X = self.scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_val)

        print("\nValidation Results:")
        print(classification_report(y_val, preds, zero_division=0))

    def predict(self, text):
        features = self._extract_features(text).reshape(1, -1)
        features = self.scaler.transform(features)
        return self.model.predict(features)[0]

    def predict_with_confidence(self, text):
        claims = self.parser.split_into_claims(text)
        hard = detect_contradictions(claims)

        features = self._extract_features(text).reshape(1, -1)
        features = self.scaler.transform(features)

        probas = self.model.predict_proba(features)[0]
        p_consistent = probas[1]
        confidence = float(max(probas))

        # ✅ HACKATHON-SAFE RULE
        if len(claims) == 1 and len(hard) == 0:
            prediction = 1  # consistent
        else:
            prediction = int(p_consistent >= 0.6)

        return prediction, confidence
    