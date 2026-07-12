from transformers import AutoTokenizer


# def load_tokenizer():

#     tokenizer = AutoTokenizer.from_pretrained(
#         "microsoft/deberta-v3-base"
#     )

#     return tokenizer

# from transformers import AutoModelForMultipleChoice


# def load_model():

#     return AutoModelForMultipleChoice.from_pretrained(
#         "microsoft/deberta-v3-base"
#     )




def load_deberta():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForMultipleChoice.from_pretrained(
        "microsoft/deberta-v3-base",
        torch_dtype=torch.float32
    )
    return tokenizer, model.float()


def load_roberta():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForMultipleChoice.from_pretrained(
        "roberta-base",
        torch_dtype=torch.float32
    )
    return tokenizer, model.float()