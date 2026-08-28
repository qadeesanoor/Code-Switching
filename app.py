import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_ID = "qadeesanoor/code-switching-codesaviours-si26-qadeesa"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

classifier = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",  # merges sub-word pieces into whole words
)

LABEL_COLORS = {
    "URD": "#ffd6d6",
    "ENG": "#d6e4ff",
    "MIX": "#d6ffd9",
}


def predict(text):
    if not text or not text.strip():
        return {"text": "", "entities": []}

    results = classifier(text)

    entities = []
    for r in results:
        entities.append(
            {
                "entity": r["entity_group"],
                "start": r["start"],
                "end": r["end"],
            }
        )

    return {"text": text, "entities": entities}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Roman Urdu / English text",
        placeholder="e.g. mujhe ye model lgaya hai pr app.py b bnani hai iski",
        lines=4,
    ),
    outputs=gr.HighlightedText(
        label="Word-level language tags",
        color_map=LABEL_COLORS,
    ),
    title="Roman Urdu–English Code-Switching Tagger",
    description="Tags each word as URD (Urdu), ENG (English), or MIX (mixed/ambiguous). "
    "Model: xlm-roberta fine-tuned for token classification.",
    examples=[
        "mujhe ye model lgaya hai pr app.py b bnani hai iski",
        "yar this is so exciting, chalo start krtay hain",
    ],
)

if __name__ == "__main__":
    demo.launch()
