import gradio as gr
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rank import run_pipeline, validate_and_write, CONFIG

def run_ranking(jsonl_file):
    if jsonl_file is None:
        return None, None

    out_path = os.path.join(tempfile.gettempdir(), "team_ai_quartet.csv")

    try:
        rows = run_pipeline(jsonl_file.name, CONFIG)
        validate_and_write(rows, out_path, CONFIG)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]}), None

    df = pd.read_csv(out_path)
    return df.head(20), out_path


with gr.Blocks(title="Redrob Candidate Ranking") as demo:
    gr.Markdown("# 🎯 Redrob Intelligent Candidate Discovery & Ranking")
    gr.Markdown(
        "Upload `candidates.jsonl` (up to 100,000 candidate profiles) to rank them "
        "against the Senior AI Engineer job description. CPU-only, offline, ~60s."
    )

    file_input = gr.File(label="Upload candidates.jsonl", file_types=[".jsonl"])
    run_btn = gr.Button("Run Ranking", variant="primary")
    output_table = gr.Dataframe(label="Top 20 Preview")
    output_file = gr.File(label="Download Full team_ai_quartet.csv")

    run_btn.click(
        fn=run_ranking,
        inputs=[file_input],
        outputs=[output_table, output_file],
    )

if __name__ == "__main__":
    demo.launch()