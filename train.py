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

from dataset import MCQDataset

print("MCQ Dataset initialized.")

from model import load_tokenizer

tokenizer = load_tokenizer()

print("Tokenizer loaded.")

# def freeze_backbone(model):
#     for param in model.deberta.parameters():
#         param.requires_grad = False


# def unfreeze_backbone(model):
#     for param in model.deberta.parameters():
#         param.requires_grad = True



for epoch in range(epochs):

    model.train()

    for batch in train_loader:

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

import wandb

wandb.init(
    project="smart-mcq-solver"
)

wandb.log({
    "loss": loss.item()
})

def freeze_backbone(model):
    for name, p in model.named_parameters():
        if not name.startswith("classifier") and not name.startswith("pooler"):
            p.requires_grad = False


def unfreeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = True