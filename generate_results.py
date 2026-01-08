import pandas as pd
from ingestion.load_data import DataLoader
from modeling.baseline_model import BaselineConsistencyModel
from reasoning.claim_parser import ClaimParser
from reasoning.signals import detect_contradictions
from reasoning.scoring import contradiction_score
from reasoning.explainer import generate_explanation

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

def main():
    loader = DataLoader(TRAIN_PATH, TEST_PATH)
    train_df = loader.load_train()
    test_df = loader.load_test()

    model = BaselineConsistencyModel()
    model.train(train_df)

    parser = ClaimParser()
    rows = []

    for _, r in test_df.iterrows():
        claims = parser.split_into_claims(r["content"])
        hard = detect_contradictions(claims)
        soft = contradiction_score(claims)

        pred, conf = model.predict_with_confidence(r["content"])
        explanation = generate_explanation(claims, hard, soft)

        rows.append({
            "id": r["id"],
            "prediction": "consistent" if pred == 1 else "contradict",
            "confidence": round(conf, 3),
            "explanation": explanation
        })

    pd.DataFrame(rows).to_csv("results.csv", index=False)
    print("results.csv generated successfully.")

if __name__ == "__main__":
    main()
