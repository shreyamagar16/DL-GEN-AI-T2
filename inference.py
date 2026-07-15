import pandas as pd

def generate_submission(
    test,
    deberta_model,
    deberta_tok,
    roberta_model,
    roberta_tok,
    best_weight,
    get_probs,
    top3_from_probs,
):

    predictions = []

    for _, row in test.iterrows():
        p_deberta = get_probs(deberta_model, deberta_tok, row)
        p_roberta = get_probs(roberta_model, roberta_tok, row)

        p_final = best_weight * p_deberta + (1 - best_weight) * p_roberta

        predictions.append({
            "id": row["id"],
            "prediction": " ".join(top3_from_probs(p_final))
        })

    submission = pd.DataFrame(predictions)
    submission.to_csv("submission.csv", index=False)

    print("submission.csv saved")