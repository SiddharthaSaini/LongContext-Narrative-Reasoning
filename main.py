from ingestion.load_data import DataLoader

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

def main():
    loader = DataLoader(TRAIN_PATH, TEST_PATH)

    train_df = loader.load_train()
    test_df = loader.load_test()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nLabel distribution:")
    print(train_df["label"].value_counts())

    print("\nSample row:")
    print(train_df.iloc[0])

if __name__ == "__main__":
    main()
