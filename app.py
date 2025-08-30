# app.py — SpectralMix Super UI: One Head, Two Modalities + Simulator
# HF Spaces-friendly, CPU-ready (Gradio 4.x)

import io
import json
import platform
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# -------- local imports (должны быть в проекте) --------
# infer.py обязан экспонировать: load_classes, load_head, predict_image, routing_for_text
from infer import load_classes, load_head, predict_image, routing_for_text

# ---------------- Config ----------------
ROOT = Path(".")
CLASS_CANDIDATES = [ROOT / "classes.json", ROOT / "классы.json"]
WEIGHT_CANDIDATES = [ROOT / "head.pt", ROOT / "weights.pt", ROOT / "голова.pt", ROOT / "веса.pt"]
TAU_DEFAULT = 0.5  # стартовая температура (меняется слайдером)

# ---------------- Utilities ----------------
def pick_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def load_classes_fallback():
    p = pick_first_existing(CLASS_CANDIDATES)
    if p:
        try:
            return load_classes(str(p))
        except Exception:
            pass
    # fallback: try meta.json (root or 'artifacts'), else CIFAR-10 defaults
    for mp in [ROOT / "meta.json", ROOT / "artifacts" / "meta.json"]:
        if mp.exists():
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    if isinstance(meta.get("classes"), list):
                        return meta["classes"]
            except Exception:
                pass
    return ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Classes
CLASSES = load_classes_fallback()

# Weights (root only)
_weight_path = pick_first_existing(WEIGHT_CANDIDATES)
if _weight_path is None:
    print("[error] Model weights not found in repo root. Expected one of:",
          ", ".join([p.name for p in WEIGHT_CANDIDATES]))

# Загружаем голову (и переводим в eval)
HEAD = None
if _weight_path is not None:
    HEAD = load_head(str(_weight_path), num_classes=len(CLASSES))
    if hasattr(HEAD, "eval"):
        HEAD.eval()
    print(f"[info] Using local weights: {_weight_path}")

# ---------------- Plotting helpers (matplotlib; no custom colors) ----------------
def _fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_topk(labels, probs, title="Top-5 classes"):
    fig = plt.figure()
    if labels and probs:
        x = list(range(len(labels)))
        plt.bar(x, probs)
        plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Probability")
    plt.title(title)
    return _fig_to_pil(fig)

def plot_expert_bars(vec, title="Expert activations (predicted class)"):
    vec = np.asarray(vec, dtype=np.float64)
    fig = plt.figure()
    x = list(range(len(vec)))
    plt.bar(x, vec)
    plt.xticks(x, [f"E{i}" for i in x], rotation=0)
    plt.ylabel("Routing weight")
    plt.title(title)
    return _fig_to_pil(fig)

def plot_routing_heatmap(routing, class_labels, title="Routing heatmap (classes × experts)"):
    arr = np.array(routing, dtype=np.float32)
    fig = plt.figure()
    im = plt.imshow(arr, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im)
    plt.yticks(range(len(class_labels)), class_labels)
    plt.xlabel("Experts")
    plt.ylabel("Classes")
    plt.title(title)
    return _fig_to_pil(fig)

def _softmax_np(x, tau=1.0):
    x = np.asarray(x, dtype=np.float64) / max(tau, 1e-12)
    x = x - x.max()
    ex = np.exp(x)
    s = ex.sum()
    return (ex / (s if s > 0 else 1.0)).astype(np.float64)

def cosine_sim(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ---------------- Inference wrappers ----------------
def _ensure_ready():
    if HEAD is None:
        raise gr.Error(
            "Model weights not found in repo root. "
            "Place `head.pt` (или `weights.pt` / `голова.pt` / `веса.pt`) рядом с app.py."
        )

def _preprocess_image(img: Image.Image) -> Image.Image:
    if img is None:
        raise gr.Error("Upload an image.")
    img = img.convert("RGB")
    # лёгкий ресайз, чтобы не ронять CPU
    max_side = max(img.size)
    if max_side > 1024:
        scale = 1024.0 / max_side
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    return img

@torch.no_grad()
def run_image(img: Image.Image, tau: float):
    _ensure_ready()
    img = _preprocess_image(img)
    # Предсказание (детерминированно, без Gumbel-шума — в infer.py выключен шум на eval)
    idxs, vals, routing = predict_image(HEAD, img, CLASSES, tau=float(tau))  # routing [K,R]
    if len(idxs) == 0:
        empty = {"class": [], "prob": []}
        blank = plot_topk([], [], title="Top-5 (image)")
        return empty, blank, blank, blank, "No prediction."
    labels = [CLASSES[i] for i in idxs]
    probs = [float(v) for v in vals]
    pred_idx = int(idxs[0])
    expert_vec = routing[pred_idx].tolist()

    top5_plot = plot_topk(labels, probs, title="Top-5 (image)")
    experts_plot = plot_expert_bars(expert_vec, title="Experts (image, predicted class)")
    heatmap_plot = plot_routing_heatmap(routing, CLASSES, title="Routing heatmap (image)")

    top_ex = torch.topk(torch.tensor(expert_vec), k=min(5, len(expert_vec)))
    details = (
        f"Predicted: {CLASSES[pred_idx]}  |  τ={tau:.3f}\n"
        f"Top experts (idx:weight): " +
        ", ".join([f"{int(i)}:{float(w):.3f}" for i, w in zip(top_ex.indices.tolist(), top_ex.values.tolist())]) +
        f"\nSum(weights)={float(np.sum(expert_vec)):.3f}"
    )
    return {"class": labels, "prob": probs}, top5_plot, experts_plot, heatmap_plot, details

@torch.no_grad()
def run_text(text: str, tau: float):
    _ensure_ready()
    if not text or not text.strip():
        empty = {"class": [], "prob": []}
        blank = plot_topk([], [], title="Top-5 (text)")
        return empty, blank, blank, blank, "Enter a non-empty text."
    probs, routing = routing_for_text(HEAD, text, tau=float(tau))  # probs [K], routing [K,R]
    if probs is None or routing is None:
        blank = plot_topk([], [], title="Top-5 (text)")
        return {"class": [], "prob": []}, blank, blank, blank, "No output from text pipeline."
    top = torch.topk(probs, k=min(5, len(CLASSES)))
    labels = [CLASSES[i] for i in top.indices.tolist()]
    pvals = [float(v) for v in top.values.tolist()]
    pred_idx = int(torch.argmax(probs).item())
    expert_vec = routing[pred_idx].tolist()

    top5_plot = plot_topk(labels, pvals, title="Top-5 (text)")
    experts_plot = plot_expert_bars(expert_vec, title="Experts (text, predicted class)")
    heatmap_plot = plot_routing_heatmap(routing, CLASSES, title="Routing heatmap (text)")

    top_ex = torch.topk(torch.tensor(expert_vec), k=min(5, len(expert_vec)))
    details = (
        f"Predicted: {CLASSES[pred_idx]}  |  τ={tau:.3f}\n"
        f"Top experts (idx:weight): " +
        ", ".join([f"{int(i)}:{float(w):.3f}" for i, w in zip(top_ex.indices.tolist(), top_ex.values.tolist())]) +
        f"\nSum(weights)={float(np.sum(expert_vec)):.3f}"
    )
    return {"class": labels, "prob": pvals}, top5_plot, experts_plot, heatmap_plot, details

@torch.no_grad()
def run_compare(img: Image.Image, text: str, tau: float):
    _ensure_ready()
    if img is None or not text or not text.strip():
        blank = plot_topk([], [], title="Top-5")
        return blank, blank, "Provide both image and text."
    img = _preprocess_image(img)
    # image
    idxs_i, _, routing_i = predict_image(HEAD, img, CLASSES, tau=float(tau))
    if len(idxs_i) == 0:
        blank = plot_topk([], [], title="Top-5")
        return blank, blank, "No image prediction."
    pred_i = int(idxs_i[0])
    vec_i = routing_i[pred_i].tolist()
    # text
    probs_t, routing_t = routing_for_text(HEAD, text, tau=float(tau))
    if probs_t is None:
        blank = plot_topk([], [], title="Top-5")
        return blank, blank, "No text prediction."
    pred_t = int(torch.argmax(probs_t).item())
    vec_t = routing_t[pred_t].tolist()

    sim = cosine_sim(vec_i, vec_t)
    bar_i = plot_expert_bars(vec_i, title=f"Experts (image → {CLASSES[pred_i]})")
    bar_t = plot_expert_bars(vec_t, title=f"Experts (text  → {CLASSES[pred_t]})")
    info = (
        f"Predicted (image): {CLASSES[pred_i]} | Predicted (text): {CLASSES[pred_t]}\n"
        f"Cosine similarity of expert vectors: {sim:.3f}  (1.0 = identical, 0 = orthogonal)  |  τ={tau:.3f}"
    )
    return bar_i, bar_t, info

# ---------------- Simulator (how the head works) ----------------
_rng_global = np.random.default_rng(12345)

def _gumbel_softmax_np(logits, tau=1.0, hard=False, rng=None):
    rng = rng or _rng_global
    U = rng.uniform(low=1e-8, high=1-1e-8, size=len(logits))
    g = -np.log(-np.log(U))
    y = _softmax_np(np.asarray(logits) + g, tau=tau)
    if hard:
        hard_vec = np.zeros_like(y)
        hard_vec[int(np.argmax(y))] = 1.0
        return hard_vec
    return y

def simulate_mix(R=4, C=10, tau=0.5, hard=False, couple=0.5, seed=42, use_gumbel=False):
    """
    Возвращает:
      W: [R, C] — компонентные логиты для классов
      alpha_img, alpha_txt: [R] — смеси экспертов
      z_img, z_txt: [C] — итоговые логиты по классам
    """
    rng = np.random.default_rng(int(seed))
    # компонентные "головы": W_r \in R^{C}
    W = rng.normal(loc=0.0, scale=1.0, size=(R, C)).astype(np.float64)
    # гейтовые логиты
    gate_img = rng.normal(size=R)
    noise = rng.normal(size=R)
    gate_txt = couple * gate_img + (1.0 - couple) * noise

    if use_gumbel:
        alpha_img = _gumbel_softmax_np(gate_img, tau=tau, hard=hard, rng=rng)
        alpha_txt = _gumbel_softmax_np(gate_txt, tau=tau, hard=hard, rng=rng)
    else:
        alpha_img = _softmax_np(gate_img, tau=tau)
        alpha_txt = _softmax_np(gate_txt, tau=tau)
        if hard:
            h = np.zeros_like(alpha_img); h[int(np.argmax(alpha_img))] = 1.0; alpha_img = h
            h = np.zeros_like(alpha_txt); h[int(np.argmax(alpha_txt))] = 1.0; alpha_txt = h

    z_img = alpha_img @ W
    z_txt = alpha_txt @ W
    return W, alpha_img, alpha_txt, z_img, z_txt

def plot_class_logits(logits, class_labels, title):
    probs = _softmax_np(np.asarray(logits), tau=1.0)
    fig = plt.figure()
    x = list(range(len(class_labels)))
    plt.bar(x, probs)
    plt.xticks(x, class_labels, rotation=30, ha="right")
    plt.ylabel("Probability")
    plt.title(title)
    return _fig_to_pil(fig)

def plot_W_heatmap(W, title="Component heads W (R × C)"):
    arr = np.asarray(W, dtype=np.float32)
    if arr.shape[1] > 0:
        col_std = arr.std(axis=0, keepdims=True) + 1e-9
        arr = (arr - arr.mean(axis=0, keepdims=True)) / col_std
    fig = plt.figure()
    im = plt.imshow(arr, aspect="auto")
    plt.colorbar(im)
    plt.xlabel("Classes")
    plt.ylabel("Experts")
    plt.title(title)
    return _fig_to_pil(fig)

def run_simulator(R, C, tau, couple, seed, hard, use_gumbel):
    # валидные диапазоны
    R = int(max(2, min(16, R)))
    C = int(max(2, min(20, C)))
    # классы для подписи
    if len(CLASSES) >= C:
        cls = CLASSES[:C]
    else:
        cls = (CLASSES + [f"class_{i}" for i in range(C - len(CLASSES))])[:C]

    W, a_img, a_txt, z_img, z_txt = simulate_mix(
        R=R, C=C, tau=float(tau), hard=bool(hard),
        couple=float(couple), seed=int(seed), use_gumbel=bool(use_gumbel)
    )
    bars_img = plot_expert_bars(a_img, title="α (image)")
    bars_txt = plot_expert_bars(a_txt, title="α (text)")
    heat_W  = plot_W_heatmap(W, title="Components W (experts × classes)")
    cls_img = plot_class_logits(z_img, cls, title="Mixed logits → probs (image)")
    cls_txt = plot_class_logits(z_txt, cls, title="Mixed logits → probs (text)")

    sim = cosine_sim(a_img, a_txt)
    info = (
        f"Cosine(α_image, α_text) = {sim:.3f}  |  τ={tau:.3f} "
        f"| {'Gumbel' if use_gumbel else 'Softmax'} | {'hard' if hard else 'soft'} | couple={couple:.2f}"
    )
    top_ex_img = ", ".join([f"E{i}:{w:.3f}" for i, w in enumerate(a_img)])
    top_ex_txt = ", ".join([f"E{i}:{w:.3f}" for i, w in enumerate(a_txt)])
    details = f"α_image: [{top_ex_img}]\nα_text : [{top_ex_txt}]"
    return bars_img, bars_txt, heat_W, cls_img, cls_txt, info, details

# ---------------- UI helpers ----------------
def _env_banner(weights_path: str, classes_len: int):
    import importlib
    pkgs = {}
    try:
        import torchvision as _tv
        pkgs["torchvision"] = _tv.__version__
    except Exception:
        pkgs["torchvision"] = "n/a"
    try:
        import transformers as _tf
        pkgs["transformers"] = _tf.__version__
    except Exception:
        pkgs["transformers"] = "n/a"
    try:
        import open_clip_torch as _oc
        pkgs["open_clip_torch"] = getattr(_oc, "__version__", "present")
    except Exception:
        pkgs["open_clip_torch"] = "n/a"

    rows = [
        f"- **torch**: {torch.__version__}",
        f"- **torchvision**: {pkgs['torchvision']}",
        f"- **gradio**: {gr.__version__}",
        f"- **numpy**: {np.__version__}",
        f"- **matplotlib**: {plt.matplotlib.__version__}",
        f"- **transformers**: {pkgs['transformers']}",
        f"- **open_clip_torch**: {pkgs['open_clip_torch']}",
    ]
    sysrow = f"- **Python**: {platform.python_version()}  |  **Device**: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    wrow = f"**Using weights**: `{weights_path}`  |  **#classes**: {classes_len}"
    return wrow + "<br>" + sysrow + "<br>" + "<br>".join(rows)

# ---------------- UI ----------------
with gr.Blocks(title="SpectralMix — One Head, Two Modalities (Image/Text)") as demo:
    gr.HTML("""
    <style>.notranslate { translate: no; }</style>
    <div class="notranslate" style="font-size:28px; font-weight:700;">🧠 SpectralMix — One Head, Two Modalities</div>
    <div>One classifier head for <i>image</i> and <i>text</i>. We mix a few <b>orthogonal experts</b> with (Gumbel-)Softmax.<br>
    Same class ⇒ same expert (similarity≈1). Different classes ⇒ different experts (≈0). This avoids catastrophic forgetting across stages.</div>
    """)
    gr.Markdown(_env_banner(str(_weight_path) if _weight_path else "N/A", len(CLASSES)))

    # Общая температура для инференса
    tau_slider = gr.Slider(0.1, 2.0, value=TAU_DEFAULT, step=0.05, label="Temperature τ (inference)")

    with gr.Tab("Image → classes & experts"):
        with gr.Row():
            img_in = gr.Image(type="pil", label="Image", height=260)
            with gr.Column():
                top5_out = gr.JSON(label="Top-5 classes")
                details = gr.Textbox(label="Details", lines=3)
        with gr.Row():
            probs_plot   = gr.Image(label="Top-5 probabilities", height=260)
            experts_plot = gr.Image(label="Expert activations (predicted class)", height=260)
            routing_plot = gr.Image(label="Routing heatmap (classes × experts)", height=260)

        gr.Button("Predict").click(
            fn=run_image,
            inputs=[img_in, tau_slider],
            outputs=[top5_out, probs_plot, experts_plot, routing_plot, details],
            concurrency_limit=2
        )

    with gr.Tab("Text → classes & experts"):
        with gr.Row():
            txt_in = gr.Textbox(label="Text prompt (e.g., 'a photo of a dog')", lines=2)
            with gr.Column():
                txt_top = gr.JSON(label="Top-5 classes")
                txt_details = gr.Textbox(label="Details", lines=3)
        with gr.Row():
            txt_probs_plot   = gr.Image(label="Top-5 probabilities", height=260)
            txt_experts_plot = gr.Image(label="Expert activations (predicted class)", height=260)
            txt_routing_plot = gr.Image(label="Routing heatmap (classes × experts)", height=260)

        gr.Button("Run").click(
            fn=run_text, inputs=[txt_in, tau_slider],
            outputs=[txt_top, txt_probs_plot, txt_experts_plot, txt_routing_plot, txt_details],
            concurrency_limit=2
        )

    with gr.Tab("Compare (Image vs Text)"):
        with gr.Row():
            c_img = gr.Image(type="pil", label="Image", height=240)
            c_txt = gr.Textbox(label="Text prompt", lines=2)
        with gr.Row():
            bar_i = gr.Image(label="Experts (image)", height=260)
            bar_t = gr.Image(label="Experts (text)", height=260)
        info = gr.Textbox(label="Similarity", lines=2)
        gr.Button("Compare").click(
            run_compare, inputs=[c_img, c_txt, tau_slider], outputs=[bar_i, bar_t, info],
            concurrency_limit=2
        )
        gr.Markdown(
            "Tip: same-class pairs should give similarity ≈ **1.0**, different-class pairs → ≈ **0.0**."
        )

    with gr.Tab("Simulator (how the head works)"):
        gr.Markdown("Interactive simulator of routing and mixing. "
                    "Adjust #experts, temperature, and modality coupling; see α-vectors and class probabilities.")
        with gr.Row():
            R_in   = gr.Slider(2, 16, value=4, step=1, label="#Experts (R)")
            C_in   = gr.Slider(2, 20, value=min(10, len(CLASSES)), step=1, label="#Classes (C)")
            tau_in = gr.Slider(0.1, 2.0, value=0.5, step=0.05, label="Temperature τ (sim)")
        with gr.Row():
            couple_in  = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Coupling of modalities (0=different, 1=identical)")
            seed_in    = gr.Number(value=42, precision=0, label="Seed")
            hard_in    = gr.Checkbox(False, label="Hard routing (one-hot)")
            gumbel_in  = gr.Checkbox(False, label="Use Gumbel-Softmax (stochastic)")

        with gr.Row():
            sim_alpha_img = gr.Image(label="α (image)", height=220)
            sim_alpha_txt = gr.Image(label="α (text)",  height=220)
            sim_W_heat    = gr.Image(label="Components (W)", height=220)
        with gr.Row():
            sim_cls_img = gr.Image(label="Image → probs", height=240)
            sim_cls_txt = gr.Image(label="Text  → probs", height=240)

        sim_info    = gr.Textbox(label="Summary", lines=2)
        sim_details = gr.Textbox(label="Details", lines=3)

        gr.Button("Simulate").click(
            run_simulator,
            inputs=[R_in, C_in, tau_in, couple_in, seed_in, hard_in, gumbel_in],
            outputs=[sim_alpha_img, sim_alpha_txt, sim_W_heat, sim_cls_img, sim_cls_txt, sim_info, sim_details],
            concurrency_limit=2
        )

    # Healthcheck (скрытый блок, удобно при отладке)
    with gr.Row(visible=False):
        btn = gr.Button("Healthcheck")
        txt = gr.Textbox()
        def _healthcheck():
            try:
                _ensure_ready()
                return "ok"
            except Exception as e:
                return f"error: {e}"
        btn.click(lambda: _healthcheck(), outputs=txt, concurrency_limit=1)

# очередь и запуск (Gradio 4.x — без concurrency_count)
demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
