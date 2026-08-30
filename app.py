import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# ============================================================
# MODEL
# ============================================================

MODEL_ID = "qadeesanoor/code-switching-codesaviours-si26-qadeesa"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

classifier = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
)


# ============================================================
# LABEL COLORS  (dark-theme friendly)
# ============================================================

LABEL_COLORS = {
    "URD": "#64748b",   
    "ENG": "#94a3b8",   
    "MIX": "#a78bfa",   
}


# ============================================================
# CUSTOM CSS 
# ============================================================

CUSTOM_CSS = """
/* =========================================================
   GLOBAL
   ========================================================= */

.gradio-container {
    max-width: 1050px !important;
    margin: 0 auto !important;
    padding: 0 24px 20px 24px !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    background: #020617 !important;
    color: #e2e8f0 !important;
}

body {
    background: #020617 !important;
}


/* =========================================================
   HEADER
   ========================================================= */

#title-block {
    text-align: center;
    padding: 32px 20px 28px 20px;
    margin-bottom: 24px;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 20px;
    box-shadow: 0 0 50px -12px rgba(99, 102, 241, 0.3);
    position: relative;
    overflow: hidden;
}

#title-block::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center, rgba(99, 102, 241, 0.1) 0%, transparent 60%);
    pointer-events: none;
}

/* Top decorative line */
#title-block .decoration {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: #818cf8;
    font-size: 13px;
    letter-spacing: 3px;
    margin-bottom: 14px;
    position: relative;
}

#title-block .decoration::before,
#title-block .decoration::after {
    content: "";
    height: 1px;
    width: 90px;
    background: #4338ca;
}

/* Small category */
#title-block .category {
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 8px;
    position: relative;
}

/* Main title */
#title-block h1 {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    background: linear-gradient(90deg, #67e8f9, #a5b4fc, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 10px 0;
    position: relative;
}

/* Subtitle */
#title-block p {
    color: #94a3b8;
    font-size: 0.93rem;
    line-height: 1.6;
    margin: 0;
    position: relative;
}


/* =========================================================
   MAIN TOP ROW
   ========================================================= */

#main-row {
    align-items: stretch !important;
    gap: 20px !important;
}


/* =========================================================
   INPUT / OUTPUT COLUMNS
   ========================================================= */

#left-panel,
#right-panel {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.35) !important;
    padding: 20px !important;
}


/* =========================================================
   INPUT BOX
   ========================================================= */

#left-panel textarea {
    min-height: 300px !important;
    height: 300px !important;
    resize: vertical !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background: #0f172a !important;
    color: #f1f5f9 !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

#left-panel textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
}


/* =========================================================
   LABELS
   ========================================================= */

#left-panel label span,
#right-panel label span {
    color: #e2e8f0 !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
}


/* =========================================================
   BUTTONS
   ========================================================= */

#clear-button,
#tag-button {
    width: 100% !important;
    height: 42px !important;
    min-height: 42px !important;
    margin-top: 10px !important;
    border-radius: 12px !important;
    font-size: 0.9rem !important;
}

/* Clear */
#clear-button {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}

#clear-button:hover {
    background: #334155 !important;
    border-color: #475569 !important;
}

/* Tag */
#tag-button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.35);
    transition: all 0.22s ease !important;
}

#tag-button:hover {
    background: linear-gradient(135deg, #4338ca, #6d28d9) !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 28px rgba(124, 58, 237, 0.5);
}


/* =========================================================
   OUTPUT BOX
   ========================================================= */

#right-panel .highlightedtext,
#right-panel .prose,
#right-panel [data-testid="highlighted-text"],
#right-panel .token-classification {
    min-height: 382px !important;
    height: 382px !important;
    border-radius: 12px !important;
    border: 1px solid #1e293b !important;
    padding: 18px !important;
    line-height: 1.9 !important;
    font-size: 1.02rem !important;
    background: #0f172a !important;
    color: #f1f5f9 !important;
}

/* Make sure the inner container / spans also stay dark */
#right-panel .highlightedtext > div,
#right-panel .highlightedtext span {
    background-color: transparent !important;
    color: inherit !important;
}

/* Tag pills themselves */
#right-panel mark,
#right-panel .token,
#right-panel span[style*="background"] {
    border-radius: 4px !important;
    padding: 1px 4px !important;
    color: #0f172a !important;
}


/* =========================================================
   EXAMPLES
   ========================================================= */

#examples-section {
    margin-top: 20px !important;
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 16px !important;
    padding: 18px 20px !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.35) !important;
}

/* Example heading */
#examples-section label,
#examples-section label span {
    color: #e2e8f0 !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
}

/* Example buttons */
#examples-section button {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    text-align: left !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
}

#examples-section button:hover {
    background: #334155 !important;
    border-color: #6366f1 !important;
}


/* =========================================================
   FOOTER
   ========================================================= */

#footer {
    text-align: center !important;
    padding: 28px 10px 12px 10px !important;
    color: #64748b !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.02em !important;
}

#footer .footer-line {
    width: 100px;
    height: 1px;
    background: #1e293b;
    margin: 0 auto 12px auto;
}

#footer strong {
    color: #a5b4fc !important;
    font-weight: 600 !important;
}


/* =========================================================
   HIDE GRADIO FOOTER
   ========================================================= */

footer {
    display: none !important;
}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 768px) {
    .gradio-container {
        padding: 0 14px 15px 14px !important;
    }

    #title-block h1 {
        font-size: 1.55rem !important;
    }

    #title-block .decoration::before,
    #title-block .decoration::after {
        width: 45px;
    }

    #main-row {
        flex-direction: column !important;
    }

    #left-panel textarea {
        min-height: 230px !important;
        height: 230px !important;
    }

    #right-panel .highlightedtext {
        min-height: 300px !important;
        height: 300px !important;
    }
}
"""


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(text):
    if not text or not text.strip():
        return {
            "text": "",
            "entities": []
        }

    results = classifier(text)

    entities = []
    for result in results:
        entities.append(
            {
                "entity": result["entity_group"],
                "start": result["start"],
                "end": result["end"],
            }
        )

    return {
        "text": text,
        "entities": entities
    }


# ============================================================
# THEME
# ============================================================

theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[
        gr.themes.GoogleFont("Inter"),
        "system-ui",
        "sans-serif"
    ],
).set(
    body_background_fill="#020617",
    body_background_fill_dark="#020617",
    background_fill_primary="#0f172a",
    background_fill_secondary="#020617",
    block_background_fill="#0f172a",
    block_background_fill_dark="#0f172a",
    block_border_color="#1e293b",
    block_border_width="1px",
    block_radius="16px",
    block_shadow="0 2px 12px rgba(0, 0, 0, 0.35)",
    block_label_text_color="#e2e8f0",
    block_title_text_color="#f8fafc",
    button_primary_background_fill="linear-gradient(135deg, #4f46e5, #7c3aed)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4338ca, #6d28d9)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#1e293b",
    button_secondary_background_fill_hover="#334155",
    button_secondary_border_color="#334155",
    input_background_fill="#0f172a",
    input_border_color="#334155",
    input_radius="12px",
)


# ============================================================
# APP
# ============================================================

with gr.Blocks(
    theme=theme,
    css=CUSTOM_CSS,
    title="Roman Urdu–English Code-Switching Tagger",
) as demo:

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML(
        """
        <div id="title-block">
            <div class="decoration">✦</div>
            <div class="category">NLP • WORD-LEVEL LANGUAGE IDENTIFICATION</div>
            <h1>Roman Urdu–English Code-Switching Tagger</h1>
            <p>
                Detect and highlight Urdu, English & mixed tokens in code-switched text
            </p>
        </div>
        """
    )

    # ========================================================
    # INPUT + OUTPUT
    # ========================================================

    with gr.Row(
        elem_id="main-row",
        equal_height=True
    ):
        # LEFT: INPUT
        with gr.Column(
            scale=1,
            elem_id="left-panel"
        ):
            input_box = gr.Textbox(
                label="Enter Text",
                placeholder=(
                    "e.g. mujhe ye model lgaya hai "
                    "pr app.py b bnani hai iski"
                ),
                lines=10,
                max_lines=10,
                show_label=True,
            )

            clear_btn = gr.Button(
                "Clear",
                elem_id="clear-button",
                size="sm",
            )

            submit_btn = gr.Button(
                "Tag Text",
                elem_id="tag-button",
                variant="primary",
                size="sm",
            )

        # RIGHT: OUTPUT
        with gr.Column(
            scale=1,
            elem_id="right-panel"
        ):
            output_box = gr.HighlightedText(
                label="Word-level Language Tags",
                color_map=LABEL_COLORS,
                show_legend=False,
            )

    # ========================================================
    # EXAMPLES
    # ========================================================

    with gr.Column(
        elem_id="examples-section"
    ):
        gr.Examples(
            examples=[
                ["mujhe ye model lgaya hai pr app.py b bnani hai iski"],
                ["yar this is so exciting, chalo start krtay hain"],
                ["aaj meeting hai so please time pr aa jana"],
                ["this model bohat acha perform kar raha hai"],
            ],
            inputs=input_box,
            label="Try an example",
        )

    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div id="footer">
            <div class="footer-line"></div>
            <strong>Powered by XLM-RoBERTa</strong>
            &nbsp; · &nbsp;
            Built for Roman Urdu–English Code-Switching
        </div>
        """
    )

    # ========================================================
    # EVENTS
    # ========================================================

    clear_btn.click(
        fn=lambda: ("", {"text": "", "entities": []}),
        inputs=None,
        outputs=[input_box, output_box],
    )

    submit_btn.click(
        fn=predict,
        inputs=input_box,
        outputs=output_box,
    )

    input_box.submit(
        fn=predict,
        inputs=input_box,
        outputs=output_box,
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch()
