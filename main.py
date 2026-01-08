from ingestion.load_data import DataLoader
from reasoning.claim_parser import ClaimParser
from reasoning.signals import detect_contradictions
from modeling.baseline_model import BaselineConsistencyModel

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

def main():
    loader = DataLoader(TRAIN_PATH, TEST_PATH)
    train_df = loader.load_train()
    test_df = loader.load_test()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nLabel distribution:\n", train_df["label"].value_counts())

    # ---------------- DAY-2 DEMO ----------------
    print("\n" + "="*50)
    print("DAY-2: REASONING DEMO")
    print("="*50)

    parser = ClaimParser()
    sample = train_df.iloc[0]

    claims = parser.split_into_claims(sample["content"])
    signals = detect_contradictions(claims)

    print("\nAtomic Claims:")
    for c in claims:
        print("-", c)

    print("\nExplicit Signals:", signals if signals else "None")

    # ---------------- DAY-4 MODEL ----------------
    print("\n" + "="*50)
    print("DAY-4: BASELINE MODEL")
    print("="*50)

    model = BaselineConsistencyModel()
    model.train(train_df)

    test_sample = test_df.iloc[0]
    pred = model.predict(test_sample["content"])

    print("\nPrediction on test sample:")
    print("Consistent" if pred == 1 else "Contradict")

if __name__ == "__main__":
    main()
