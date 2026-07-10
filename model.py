from transformers import AutoTokenizer


def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-base"
    )

    return tokenizer

from transformers import AutoModelForMultipleChoice


def load_model():

    return AutoModelForMultipleChoice.from_pretrained(
        "microsoft/deberta-v3-base"
    )