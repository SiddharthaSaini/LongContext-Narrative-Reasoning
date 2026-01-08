from ingestion.load_data import DataLoader
from reasoning.claim_parser import ClaimParser
from reasoning.signals import detect_contradictions

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

def main():
    # Load data
    loader = DataLoader(TRAIN_PATH, TEST_PATH)
    train_df = loader.load_train()
    test_df = loader.load_test()

    # Basic dataset checks (professional hygiene)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nLabel distribution (train):")
    print(train_df["label"].value_counts())

    # Initialize reasoning components
    parser = ClaimParser()

    # Take one training sample for reasoning demo
    sample = train_df.iloc[0]

    print("\nBook:", sample["book_name"])
    print("Character:", sample["char"])
    print("\nBackstory Content:")
    print(sample["content"])

    # Step 1: Atomic claim decomposition
    claims = parser.split_into_claims(sample["content"])

    print("\nAtomic Claims:")
    for c in claims:
        print("-", c)

    # Step 2: Contradiction signal detection
    signals = detect_contradictions(claims)

    print("\nReasoning Signals:")
    if signals:
        for s in signals:
            print("-", s)
    else:
        print("No explicit contradiction signals detected.")

if __name__ == "__main__":
    main()
