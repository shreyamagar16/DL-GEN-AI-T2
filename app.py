import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice

MODEL = "24f100ds2698/smart-mcq-solver-roberta"

st.set_page_config(page_title="Smart MCQ Solver")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("🧠 Smart MCQ Solver")
st.write("Enter an MCQ and its five options.")

question = st.text_area("Question")

options = []
for letter in ["A", "B", "C", "D", "E"]:
    options.append(st.text_input(f"Option {letter}"))

if st.button("Solve"):
    if not question or any(not x for x in options):
        st.warning("Please enter the question and all 5 options.")
    else:
        choices = [f"{question} [SEP] {x}" for x in options]

        enc = tokenizer(
            choices,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].unsqueeze(0),
                attention_mask=enc["attention_mask"].unsqueeze(0)
            ).logits

        probs = torch.softmax(logits, dim=1)[0]
        answer = torch.argmax(probs).item()

        st.success(f"Predicted Answer: **{chr(65 + answer)}**")

        st.write("### Confidence")
        for i, p in enumerate(probs):
            st.write(f"{chr(65+i)}: {p.item()*100:.2f}%")
