import re


def clean_text(text):
    """Convert text to lowercase and remove extra characters."""
    text = str(text).lower()
    words = re.findall(r"[a-z0-9]+", text)
    return " ".join(words)


def tokenize(text):
    """Return a list of cleaned words."""
    return clean_text(text).split()


if __name__ == "__main__":
    sample = "Hello, NLP World!"
    print(tokenize(sample))


def average_precision_at_3(actual, predicted):
    """
    Compute Average Precision@3 for a single prediction.
    """
    if actual in predicted[:3]:
        return 1.0 / (predicted.index(actual) + 1)
    return 0.0


def map_at_3(actual_labels, predicted_labels):
    """
    Compute Mean Average Precision@3.
    """
    scores = []

    for actual, predicted in zip(actual_labels, predicted_labels):
        scores.append(average_precision_at_3(actual, predicted))

    return sum(scores) / len(scores)