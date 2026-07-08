import pandas as pd
from utils import clean_text


def load_dataset(path):
    df = pd.read_csv(path)

    if "prompt" in df.columns:
        df["prompt"] = df["prompt"].apply(clean_text)

    return df


if __name__ == "__main__":
    print("Dataset utilities ready.")

from utils import split_dataset

print("Dataset splitting ready.")

import torch
from torch.utils.data import Dataset


class MCQDataset(Dataset):

    def __init__(self, inputs, labels=None):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        if self.labels is None:
            return self.inputs[idx]

        return self.inputs[idx], self.labels[idx]