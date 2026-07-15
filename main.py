from model import load_deberta, load_roberta
from train import train_staged

deberta_tok, deberta_model = load_deberta()
deberta_model = train_staged(
    deberta_model,
    deberta_tok,
    "model2-deberta-v3-base-staged"
)

roberta_tok, roberta_model = load_roberta()
roberta_model = train_staged(
    roberta_model,
    roberta_tok,
    "model3-roberta-base-staged"
)

from inference import generate_submission

generate_submission(
    test,
    deberta_model,
    deberta_tok,
    roberta_model,
    roberta_tok,
    best_weight,
    get_probs,
    top3_from_probs,
)