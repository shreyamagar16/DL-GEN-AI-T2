import pandas as pd
from utils import clean_text


def load_dataset(path):
    df = pd.read_csv(path)

    if "prompt" in df.columns:
        df["prompt"] = df["prompt"].apply(clean_text)

    return df


if __name__ == "__main__":
    print("Dataset utilities ready.")