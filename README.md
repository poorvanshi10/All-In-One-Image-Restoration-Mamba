<div align="center">

# 🌊 Diff-Mamba

### Adaptive Multi-Degradation Diff-Mamba for Efficient All-in-One Blind Image Restoration

*One backbone. Every degradation. No retraining.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)]()
[![Paper](https://img.shields.io/badge/Paper-PDF-B31B1B)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Made at](https://img.shields.io/badge/Built%20at-MANIT%20Bhopal%20%7C%20Medi--Caps-6f42c1)]()

**[Overview](#-what-is-diff-mamba) · [How it works](#%EF%B8%8F-how-it-works) · [Results](#-results) · [Quick Start](#-quick-start) · [Citation](#-citation)**

</div>

---

## 💡 What Is Diff-Mamba?

Real photos are almost never *cleanly* degraded. A single image can be noisy **and** blurry **and** streaked with rain — all at once. Most State Space Model (SSM) restoration networks like MambaIR are fast and linear-time, but they're built for **one** degradation at a time and fall apart the moment the corruption pattern shifts.

Diff-Mamba fixes this at the source. Instead of bolting a "hint" onto the input (prompts, contrastive embeddings), it reaches **inside the SSM recurrence itself** and rewrites it — per image, per forward pass, with zero retraining.

> 🧠 A 128-dim diagnostic token reads the degradation.
> ⚙️ It directly edits the SSM's Δ, B, C parameters.
> 🎨 A lightweight diffusion pass restores the texture that regression loss erases.

All of this stays **linear in sequence length** — so you get Transformer-grade adaptivity at SSM-grade efficiency.

---

## ⚙️ How It Works

```
        ┌────────────────────┐
 I_LQ ─▶│  Diagnostic Encoder │──▶  z_d  (128-d degradation token)
        └────────────────────┘        │
                                       ▼
                         ┌─────────────────────────┐
                         │   Conditional Mamba Block │
                         │   Conditional Mamba Block │  ◀── Δ, B, C rewritten per-sample
                         └─────────────────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │  Reverse Diffusion (T=50) │
                         └─────────────────────────┘
                                       │
                                       ▼
                                    I_HQ
```

**The core trick**, in three lines:

```
Δt = Softplus( Linear(LN(xt)) + Ω_Δ(z_d) )   # how far to look
Bt = Linear(LN(xt)) + Ω_B(z_d)               # what to let in
Ct = Linear(LN(xt)) + Ω_C(z_d)               # what to read out
```

Small Δ → the model zooms in on fine-grained noise. Large Δ → it widens its lens for spatially spread-out blur. The image itself decides which regime it needs.

### 🧩 Three Moving Parts

| Component | Job |
|---|---|
| 🔍 **Diagnostic Encoder** | Conv stem + 4 residual blocks → GAP → 128-d degradation token `z_d` |
| 🎛️ **Adaptive SSM Routing** | Three MLPs offset Δ, B, C — physically reshaping the Mamba recurrence kernel |
| 🎨 **Diffusion Refiner** | 50-step conditional reverse diffusion recovers texture ℓ2 loss smooths away |

The **MambaIRv2 backbone stays frozen** the entire time — only the encoder and routing MLPs are trained. Everything downstream of that is pure inference-time adaptation.

---

## 📊 Results

### 🏆 Headline number: **+0.1216 dB PSNR on Urban100**

*(texture-rich, geometry-heavy — exactly where adaptive routing should matter most)*

<details>
<summary><b>📐 Super-resolution benchmarks (×4) — click to expand</b></summary>
<br>

| Dataset | Base PSNR | Ours PSNR | ΔPSNR | Base SSIM | Ours SSIM | ΔSSIM |
|---|---|---|---|---|---|---|
| Set5 | 36.4791 | 36.5069 | +0.0278 | 0.9546 | 0.9549 | +0.00024 |
| Set14 | 32.6077 | 32.6819 | +0.0742 | 0.9143 | 0.9148 | +0.00044 |
| BSD100 | 26.6903 | 26.6811 | −0.0092 | 0.7902 | 0.7903 | +0.00012 |
| **Urban100** | 31.9148 | **32.0364** | **+0.1216** | 0.9576 | 0.9583 | +0.00076 |
| Manga109 | 26.8007 | 26.8047 | +0.0041 | 0.9140 | 0.9146 | +0.00063 |

BSD100 is the honest exception — PSNR dips slightly while SSIM still creeps up, hinting the model trades a sliver of pixel accuracy for structural fidelity there.

</details>

<details>
<summary><b>🌪️ All-in-one, mixed-degradation comparison (PSNR, BSD100 / Urban100) — click to expand</b></summary>
<br>

*Denoise: σ=25 · Deblur: 17×17 kernel · Derain: Rain100H*

| Method | Type | Denoise | Deblur | Derain |
|---|---|---|---|---|
| DnCNN | Single | 31.73 / – | – / – | – / – |
| NAFNet | Single | 31.02 / – | 32.51 / – | – / – |
| AirNet | All-in-one | 30.91 / 29.33 | 30.24 / 28.45 | 32.98 / 31.67 |
| PromptIR | All-in-one | 31.31 / 30.58 | 31.40 / 29.98 | 34.81 / 33.52 |
| MambaIRv2 | Single | 31.62 / 31.09 | 32.18 / 30.71 | 35.04 / 33.89 |
| **Diff-Mamba (Ours)** | **All-in-one** | **31.85 / 31.40** | **32.47 / 31.15** | **35.29 / 34.21** |

The only *all-in-one* model here that beats the frozen single-task MambaIRv2 backbone on every track.

</details>

<details>
<summary><b>🔬 Ablation study (Set5, ×4) — click to expand</b></summary>
<br>

| Configuration | PSNR (dB) | SSIM |
|---|---|---|
| Baseline MambaIRv2 | 36.4842 | 0.9546 |
| + Diagnostic branching | 36.5084 | 0.9548 |
| + Adaptive refinement | 36.4748 | 0.9547 |
| **Full Diff-Mamba** | **36.5063** | **0.9549** |

Diagnostic branching drives the PSNR gain; adaptive refinement drives the SSIM gain. Together they're complementary, not redundant.

</details>

### ⚡ Efficiency

| | Diff-Mamba | Comparable Transformer |
|---|---|---|
| Memory @ 2048×2048 | **3.8 GB** | 22.0 GB |
| Memory scaling | Linear | Quadratic |
| Inference | Full-resolution, no patch stitching | — |

---

## 🧪 Datasets Used

**Super-resolution:** Set5 · Set14 · BSD100 · Urban100 · Manga109
**Mixed-degradation tracks:** Gaussian denoising (σ ∈ {15, 25, 50}) · Motion deblur (13×13, 17×17) · Rain streaks (Rain100L/H)

---

## 🚀 Quick Start

```bash
git clone https://github.com/<your-username>/diff-mamba.git
cd diff-mamba
pip install -r requirements.txt
```

```bash
# Inference on a single image
python scripts/eval.py --input path/to/image.png --checkpoint checkpoints/diff-mamba.pth

# Train the diagnostic encoder + routing MLPs (backbone frozen)
python scripts/train.py --config configs/diff_mamba.yaml
```

> 🔧 Backbone: pretrained **MambaIRv2** (frozen) · Optimizer: AdamW, LR 2e-4 (cosine decay) · 256×256 patches, batch size 8, single RTX 3090 · Diffusion: T = 50, linear schedule

---

## 👥 Authors

| | |
|---|---|
| **Poorvanshi Lowanshi** | Dept. of CSE, Medi-Caps University, Indore |
| **Ramesh Kumar Thakur** | Centre of Excellence in Product Design and Smart Manufacturing, MANIT Bhopal |
| **Shashank Gupta** | Dept. of CSE, MANIT Bhopal |
| **Tanveer Fatema Khan** | Dept. of CSE, MANIT Bhopal |

## 📄 License

Released under the [MIT License](LICENSE).

<div align="center">

⭐ **If this project is useful to you, consider starring the repo!** ⭐

</div>
