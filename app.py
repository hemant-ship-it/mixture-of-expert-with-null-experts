import os
import time

import gradio as gr
import tiktoken
import torch

from train import MoENullModel, Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./checkpoints"

enc = tiktoken.get_encoding("gpt2")

def encode(text):
    return enc.encode(text, allowed_special={"<|endoftext|>"})

def decode(ids):
    return enc.decode(ids)

_cache = {}

def list_checkpoints():
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    return sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")], reverse=True)

def load_model(name):
    if name in _cache:
        return _cache[name]["model"]
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, name), map_location=DEVICE, weights_only=False)
    model = MoENullModel(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(DEVICE)
    _cache[name] = {"model": model, "step": ckpt.get("step", "?")}
    return model

@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature, top_k):
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    seq_len = getattr(getattr(model, "config", None), "seq_len", 128)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -seq_len:])
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        idx = torch.cat([idx, torch.multinomial(torch.softmax(logits, dim=-1), 1)], dim=1)
    return idx[0].tolist()

def run_generation(checkpoint_name, prompt, max_new_tokens, temperature, top_k):
    if not checkpoint_name:
        return "⚠️  No checkpoint selected. Add .pt files to ./checkpoints/"
    try:
        model = load_model(checkpoint_name)
    except Exception as e:
        return f"❌  Failed to load model:\n{e}"
    prompt_ids = encode(prompt)
    if not prompt_ids:
        return "⚠️  Empty prompt after encoding."
    start = time.time()
    try:
        out = generate(model, prompt_ids, int(max_new_tokens), temperature, int(top_k))
    except Exception as e:
        return f"❌  Generation error:\n{e}"
    elapsed = time.time() - start
    new_ids = out[len(prompt_ids):]
    step = _cache[checkpoint_name]["step"]
    stats = f"\n\n─────────────────────────────\n⚡ {len(new_ids)} tokens in {elapsed:.2f}s  ({len(new_ids)/elapsed:.1f} tok/s)  |  step {step}  |  device: {DEVICE}"
    return prompt + decode(new_ids) + stats

EXAMPLE_PROMPTS = [
    "ROMEO:",
    "To be, or not to be,",
    "HAMLET:\nWhat a piece of work is",
    "KING LEAR:\nBlow, winds, and crack your cheeks!",
    "First Citizen:\nBefore we proceed any further,",
    "All the world's a stage,",
]

checkpoints = list_checkpoints()
default_ckpt = checkpoints[0] if checkpoints else None

with gr.Blocks(
    title="Shakespeare MoE",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css="""
        #output-box textarea { font-family: 'Georgia', serif; font-size: 15px; line-height: 1.7; }
        #title { text-align: center; margin-bottom: 4px; }
        #subtitle { text-align: center; color: #888; margin-bottom: 20px; }
    """,
) as demo:

    gr.HTML("<h1 id='title'>🎭 Shakespeare MoE</h1>")
    gr.HTML("<p id='subtitle'>Mixture-of-Experts language model trained on Tiny Shakespeare</p>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Model")
            checkpoint_dd = gr.Dropdown(choices=checkpoints, value=default_ckpt, label="Checkpoint", info="Files in ./checkpoints/")
            refresh_btn = gr.Button("🔄 Refresh checkpoints", size="sm", variant="secondary")

            gr.Markdown("### 🎛️ Settings")
            max_tokens  = gr.Slider(10, 128, value=100, step=10,   label="Max new tokens")
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            top_k       = gr.Slider(0, 100,  value=40,  step=5,    label="Top-k  (0 = disabled)")

        with gr.Column(scale=2):
            gr.Markdown("### ✍️ Prompt")
            prompt_box = gr.Textbox(placeholder="Type a prompt or click an example below…", lines=4, show_label=False)
            generate_btn = gr.Button("✨ Generate", variant="primary", size="lg")

            gr.Markdown("### 📜 Output")
            output_box = gr.Textbox(lines=14, show_label=False, interactive=False, elem_id="output-box")

    gr.Markdown("### 💡 Example prompts — click to load, then hit Generate")
    with gr.Row():
        prompt_btns = [gr.Button(p.replace("\n", " "), size="sm") for p in EXAMPLE_PROMPTS]

    def refresh_checkpoints():
        ckpts = list_checkpoints()
        return gr.Dropdown(choices=ckpts, value=ckpts[0] if ckpts else None)

    refresh_btn.click(refresh_checkpoints, outputs=checkpoint_dd)

    for btn, prompt_text in zip(prompt_btns, EXAMPLE_PROMPTS):
        btn.click(fn=lambda p=prompt_text: p, outputs=prompt_box)

    generate_btn.click(run_generation, inputs=[checkpoint_dd, prompt_box, max_tokens, temperature, top_k], outputs=output_box)
    prompt_box.submit(run_generation,  inputs=[checkpoint_dd, prompt_box, max_tokens, temperature, top_k], outputs=output_box)

if __name__ == "__main__":
    demo.launch(share=False)