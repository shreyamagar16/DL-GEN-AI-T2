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