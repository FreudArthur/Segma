"""
Visualiser comment un noyau de convolution produit une carte de caractéristiques.

Trois figures sont générées dans outputs/feature_maps/ :
  1. 1_noyaux_classiques.png : noyaux faits à la main (Sobel, Laplacien, flou...)
     -> on voit que CHAQUE noyau produit SA carte (bords verticaux, horizontaux...).
  2. 2_noyaux_appris_yolo.png : les 16 noyaux 3x3 APPRIS par la 1re couche de YOLO,
     chacun à côté de la carte qu'il produit sur l'image.
  3. 3_profondeur_yolo.png : cartes de caractéristiques à différentes profondeurs du réseau
     -> plus on descend, plus la résolution baisse et plus les cartes sont abstraites.

Usage : python scripts/feature_maps.py [chemin_image]
"""

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "img/dog_cat.png"
OUT_DIR = Path("outputs/feature_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(IMG_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# 1. Noyaux classiques : un noyau = un type de motif détecté
# ---------------------------------------------------------------------------
KERNELS = {
    "Sobel X\n(bords verticaux)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "Sobel Y\n(bords horizontaux)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "Diagonale\n(bords à 45°)": np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),
    "Laplacien\n(tous les contours)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    "Flou moyen\n(lissage)": np.ones((3, 3)) / 9,
    "Netteté\n(sharpen)": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Relief\n(emboss)": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
}

fig, axes = plt.subplots(2, len(KERNELS) + 1, figsize=(3 * (len(KERNELS) + 1), 6.5))
axes[0, 0].imshow(gray, cmap="gray")
axes[0, 0].set_title("Image (gris)")
axes[1, 0].axis("off")
axes[1, 0].text(0.5, 0.5, "noyau  ⊛  image\n=\ncarte de\ncaractéristiques",
                ha="center", va="center", fontsize=12)

for j, (name, k) in enumerate(KERNELS.items(), start=1):
    k = k.astype(np.float32)
    # Noyau affiché avec ses valeurs
    ax = axes[1, j]
    lim = np.abs(k).max()
    ax.imshow(k, cmap="bwr", vmin=-lim, vmax=lim)
    for (r, c), v in np.ndenumerate(k):
        ax.text(c, r, f"{v:.2g}", ha="center", va="center", fontsize=10)
    ax.set_title(name, fontsize=9)
    # Carte obtenue : cv2.filter2D fait une corrélation, comme Conv2d dans PyTorch
    fmap = cv2.filter2D(gray, cv2.CV_32F, k)
    axes[0, j].imshow(np.abs(fmap) if k.sum() == 0 else fmap, cmap="magma")
    axes[0, j].set_title("Carte : " + name.split("\n")[0], fontsize=9)

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Chaque noyau fait ressortir un motif différent", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "1_noyaux_classiques.png", dpi=110)
plt.close(fig)


# ---------------------------------------------------------------------------
# Modèle : YOLO (déjà présent en local, pas de téléchargement)
# ---------------------------------------------------------------------------
model = YOLO("models/yolo26n-seg.pt").model.float().eval()

# Pré-traitement : RGB, [0,1], taille multiple de 32 (comme YOLO)
H, W = img_rgb.shape[:2]
scale = 640 / max(H, W)
h32, w32 = int(round(H * scale / 32)) * 32, int(round(W * scale / 32)) * 32
x = cv2.resize(img_rgb, (w32, h32)).astype(np.float32) / 255.0
x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


# ---------------------------------------------------------------------------
# 2. Les noyaux APPRIS de la 1re couche et la carte que chacun produit
# ---------------------------------------------------------------------------
conv0 = model.model[0].conv  # Conv2d(3, 16, 3x3, stride 2)
w = conv0.weight.detach()     # (16, 3, 3, 3)
with torch.no_grad():
    maps0 = F.conv2d(x, w, stride=conv0.stride, padding=conv0.padding)[0]  # (16, H/2, W/2)

n = w.shape[0]
cols = 8
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows * 2, cols, figsize=(2.3 * cols, 2.3 * rows * 2))
for i in range(n):
    r, c = (i // cols) * 2, i % cols
    # Noyau 3x3x3 affiché en couleur (normalisé dans [0,1])
    k = w[i].permute(1, 2, 0).numpy()
    k = (k - k.min()) / (k.max() - k.min() + 1e-8)
    axes[r, c].imshow(k, interpolation="nearest")
    axes[r, c].set_title(f"noyau {i}", fontsize=9)
    axes[r + 1, c].imshow(maps0[i].numpy(), cmap="viridis")
    axes[r + 1, c].set_title(f"→ carte {i}", fontsize=9)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Couche 1 de YOLO : 16 noyaux 3×3×3 appris → 16 cartes de caractéristiques", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "2_noyaux_appris_yolo.png", dpi=110)
plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Cartes à différentes profondeurs (hooks sur le backbone)
# ---------------------------------------------------------------------------
LAYERS = [0, 1, 2, 4, 6, 8]  # Conv, Conv, C3k2, C3k2, C3k2, C3k2
N_MAPS = 6
activations = {}
hooks = [
    model.model[i].register_forward_hook(
        lambda m, inp, out, i=i: activations.__setitem__(i, out.detach()[0])
    )
    for i in LAYERS
]
with torch.no_grad():
    model(x)
for h in hooks:
    h.remove()

fig, axes = plt.subplots(len(LAYERS), N_MAPS + 1, figsize=(2.4 * (N_MAPS + 1), 2.2 * len(LAYERS)))
for r, i in enumerate(LAYERS):
    a = activations[i]  # (C, h, w)
    axes[r, 0].imshow(img_rgb)
    axes[r, 0].set_ylabel(f"couche {i}\n{type(model.model[i]).__name__}\n{a.shape[0]} canaux\n{a.shape[1]}×{a.shape[2]}",
                          fontsize=9, rotation=0, ha="right", va="center")
    # On montre les canaux les plus actifs (énergie moyenne la plus haute)
    top = a.abs().mean(dim=(1, 2)).argsort(descending=True)[:N_MAPS]
    for c, ch in enumerate(top.tolist(), start=1):
        axes[r, c].imshow(a[ch].numpy(), cmap="viridis")
        axes[r, c].set_title(f"canal {ch}", fontsize=8)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Plus on descend : résolution ↓, nombre de canaux ↑, motifs de plus en plus abstraits", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "3_profondeur_yolo.png", dpi=110)
plt.close(fig)

print(f"Figures enregistrées dans {OUT_DIR.resolve()}")
