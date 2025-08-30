# infer.py (hotfix, совместимая версия)
import os, json, torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Literal, Optional
from torchvision import models, transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
SHARED_DIM = 512

# ---------- Энкодеры ----------
USE_OPEN_CLIP = False
try:
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device
    )
    clip_model.eval().requires_grad_(False)
    preprocess = clip_preprocess
    USE_OPEN_CLIP = True
except Exception:
    preprocess = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    _res = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
    for p in _res.parameters(): p.requires_grad = False
    class Identity(nn.Module):
        def forward(self, x): return x
    _res.fc = Identity()

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
bert = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()
for p in bert.parameters(): p.requires_grad = False
TEXT_DIM = getattr(bert.config, "hidden_size", 768)

# текстовая проекция (по умолчанию identity→512)
_text_proj = nn.Linear(TEXT_DIM, SHARED_DIM, bias=False).to(device).eval()
for p in _text_proj.parameters(): p.requires_grad = False
with torch.no_grad():
    W = torch.zeros((SHARED_DIM, TEXT_DIM), device=device)
    eye_dim = min(SHARED_DIM, TEXT_DIM)
    W[:eye_dim, :eye_dim] = torch.eye(eye_dim, device=W.device)
    _text_proj.weight.copy_(W)

# ---------- Вспомогательные ----------
def _ensure_2d(x: torch.Tensor, name: str, last_dim: int=None) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if last_dim is not None and x.size(-1) != last_dim:
        raise RuntimeError(f"{name} last dim mismatch: expected {last_dim}, got {x.size(-1)}")
    return x.contiguous()

@torch.no_grad()
def encode_images_pil(img: Image.Image) -> torch.Tensor:
    xb = preprocess(img).unsqueeze(0).to(device, non_blocking=True)
    if USE_OPEN_CLIP:
        feats = clip_model.encode_image(xb).float()
    else:
        feats = _res(xb).float()
    return F.normalize(feats, dim=-1)

@torch.no_grad()
def encode_texts(sentences: List[str]) -> torch.Tensor:
    batch = tok(sentences, return_tensors="pt", padding=True, truncation=True, max_length=48)
    for k in batch: batch[k] = batch[k].to(device, non_blocking=True)
    out = bert(**batch).last_hidden_state
    attn = batch["attention_mask"].unsqueeze(-1).float()
    summed = (out * attn).sum(dim=1)
    denom = attn.sum(dim=1).clamp_min(1.0)
    pooled = summed / denom
    proj = _text_proj(pooled)
    return F.normalize(proj.float(), dim=-1)

# ---------- Голова ----------
class SpectralMixHead(nn.Module):
    def __init__(self, in_dim, out_dim, R=8, num_modalities=2, num_tags=10, d_embed=16, k_active=2):
        super().__init__()
        self.R = int(R)
        self.k_active = int(k_active)
        self.out_dim = int(out_dim)
        self.num_tags = int(num_tags)

        self.W = nn.Parameter(torch.randn(self.R, self.out_dim, in_dim) * (0.02 / (in_dim ** 0.5)))
        self.bias = nn.Parameter(torch.zeros(self.out_dim))

        self.mod_emb = nn.Embedding(num_modalities, d_embed)
        self.tag_emb = nn.Embedding(self.num_tags, d_embed)
        self.gate = nn.Sequential(
            nn.Linear(2*d_embed, 4*self.R),
            nn.ReLU(True),
            nn.Linear(4*self.R, self.R)
        )

    def coeffs_all_tags(self, mod_ids: torch.Tensor, tau: float = 0.5,
                        mode: Literal["soft","hard","topk"] = "soft") -> torch.Tensor:
        B = mod_ids.size(0); device = mod_ids.device
        K = self.tag_emb.num_embeddings
        all_tags = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
        z_mod = self.mod_emb(mod_ids).unsqueeze(1).expand(B, K, -1)
        z_tag = self.tag_emb(all_tags)
        z = torch.cat([z_mod, z_tag], dim=-1)
        logits = self.gate(z)
        a = torch.softmax(logits / max(tau, 1e-6), dim=-1)
        if mode == "soft":
            return a
        if mode == "hard" or self.k_active <= 0:
            idx = torch.argmax(a, dim=-1, keepdim=True)
            out = torch.zeros_like(a).scatter_(-1, idx, 1.0)
            return out
        k = min(self.k_active, self.R)
        vals, idx = torch.topk(a, k=k, dim=-1)
        denom = vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = vals / denom
        out = torch.zeros_like(a).scatter_(-1, idx, weights)
        return out

    def forward(self, x: torch.Tensor, mod_ids: torch.Tensor,
                tau: float = 0.5, return_all: bool = False,
                routing_mode: Literal["soft","hard","topk"] = "soft"):
        x = _ensure_2d(x, "x", None)
        mod_ids = mod_ids.view(-1)
        a_all = self.coeffs_all_tags(mod_ids, tau=tau, mode=routing_mode)
        Wmix_rows = torch.einsum('bkr,rki->bki', a_all, self.W)
        logits = torch.einsum('bi,bki->bk', x, Wmix_rows) + self.bias
        return (logits, a_all) if return_all else logits

# ---------- I/O ----------
def load_classes(path="classes.json"):
    with open(path,"r", encoding="utf-8") as f:
        classes = json.load(f)
    return classes

def load_head(ckpt_path="./artifacts/head.pt", num_classes=10,
              R=6, d_embed=16, k_active=2) -> SpectralMixHead:
    sd = torch.load(ckpt_path, map_location=device)
    cfg = sd.get("config", {})
    head = SpectralMixHead(
        in_dim=cfg.get("in_dim", SHARED_DIM),
        out_dim=cfg.get("out_dim", num_classes),
        R=cfg.get("R", R),
        num_modalities=cfg.get("num_modalities", 2),
        num_tags=cfg.get("num_tags", num_classes),
        d_embed=cfg.get("D_EMBED", d_embed),
        k_active=cfg.get("K_ACTIVE", k_active),
    ).to(device)
    head.load_state_dict(sd["state_dict"], strict=False)
    head.eval()
    return head

# ---------- Инференс ----------
@torch.no_grad()
def predict_image(head: SpectralMixHead, img: Image.Image, classes: List[str],
                  tau=0.5, routing_mode: Literal["soft","hard","topk"]="soft"):
    x = encode_images_pil(img)
    mod = torch.zeros(1, dtype=torch.long, device=device)
    logits, a_all = head(x, mod, tau=tau, return_all=True, routing_mode=routing_mode)
    probs = logits.softmax(-1).squeeze(0).cpu()
    K = min(5, len(classes))
    topk = torch.topk(probs, k=K)
    routing = a_all.squeeze(0).cpu()[topk.indices]
    return topk.indices.tolist(), topk.values.tolist(), routing

@torch.no_grad()
def routing_for_text(head: SpectralMixHead, text: str,
                     tau=0.5, routing_mode: Literal["soft","hard","topk"]="soft"):
    x = encode_texts([text])
    mod = torch.ones(1, dtype=torch.long, device=device)
    logits, a_all = head(x, mod, tau=tau, return_all=True, routing_mode=routing_mode)
    probs = logits.softmax(-1).squeeze(0).cpu()
    K = min(5, probs.numel())
    topk = torch.topk(probs, k=K)
    routing = a_all.squeeze(0).cpu()[topk.indices]
    return probs, routing