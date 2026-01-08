from ingestion.load_data import DataLoader
from reasoning.claim_parser import ClaimParser
from reasoning.signals import detect_contradictions
from modeling.baseline_model import BaselineConsistencyModel

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

def main():
    # -------------------------
    # Load data (Day-1 hygiene)
    # -------------------------
    loader = DataLoader(TRAIN_PATH, TEST_PATH)
    train_df = loader.load_train()
    test_df = loader.load_test()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nLabel distribution (train):")
    print(train_df["label"].value_counts())

    # ------------------------------------------------
    # DAY-2: Reasoning Demo (Explainable Intelligence)
    # ------------------------------------------------
    print("\n" + "="*60)
    print("DAY-2: CLAIM REASONING DEMO")
    print("="*60)

    parser = ClaimParser()
    sample = train_df.iloc[0]

    print("\nBook:", sample["book_name"])
    print("Character:", sample["char"])
    print("\nBackstory Content:")
    print(sample["content"])

    # Atomic claims
    claims = parser.split_into_claims(sample["content"])

    print("\nAtomic Claims:")
    for c in claims:
        print("-", c)

    # Contradiction signals
    signals = detect_contradictions(claims)

    print("\nReasoning Signals:")
    if signals:
        for s in signals:
            print("-", s)
    else:
        print("No explicit contradiction signals detected.")

    # ------------------------------------------------
    # DAY-3: Baseline Model (Reasoning → Prediction)
    # ------------------------------------------------
    print("\n" + "="*60)
    print("DAY-3: BASELINE CONSISTENCY MODEL")
    print("="*60)

    model = BaselineConsistencyModel()
    model.train(train_df)

    # Predict on one test sample
    test_sample = test_df.iloc[0]

    print("\nTest Backstory:")
    print(test_sample["content"])

    prediction = model.predict(test_sample["content"])

    print("\nModel Prediction:")
    print("Consistent" if prediction == 1 else "Contradict")


if __name__ == "__main__":
    main()
