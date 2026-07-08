from utils import map_at_3

if __name__ == "__main__":

    actual = ["A", "C", "B"]

    predictions = [
        ["A", "C", "D"],
        ["B", "C", "A"],
        ["B", "D", "E"]
    ]

    score = map_at_3(actual, predictions)

    print(f"Validation MAP@3 : {score:.4f}")

from utils import label2id

print("Number of labels:", len(label2id))