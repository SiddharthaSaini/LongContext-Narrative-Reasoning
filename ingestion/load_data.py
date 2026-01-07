import pandas as pd

class DataLoader:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_train(self):
        df = pd.read_csv(self.train_path)
        return self._clean(df, is_train=True)

    def load_test(self):
        df = pd.read_csv(self.test_path)
        return self._clean(df, is_train=False)

    def _clean(self, df, is_train=True):
        df["content"] = df["content"].astype(str).str.strip()
        df["char"] = df["char"].astype(str).str.strip()
        df["book_name"] = df["book_name"].astype(str).str.strip()

        if is_train:
            df["label"] = df["label"].map({
                "consistent": 1,
                "contradict": 0
            })

        return df
