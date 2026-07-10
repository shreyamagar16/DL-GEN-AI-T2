from transformers import AutoTokenizer


def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-base"
    )

    return tokenizer