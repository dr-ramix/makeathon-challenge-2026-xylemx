"""Generate model architecture diagrams for README usage."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("content/images")


def _setup_ax(fig_w: float = 14.0, fig_h: float = 7.0):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, text, fc="#f3f4f6", ec="#1f2937", lw=1.8, fontsize=10.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#111827")


def _arrow(ax, x1, y1, x2, y2, color="#374151", lw=1.8):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arr)


def generate_snapshot_architecture(path: Path) -> None:
    fig, ax = _setup_ax(15, 7)
    ax.text(0.5, 0.95, "Snapshot Segmentation Architecture", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(
        0.5,
        0.90,
        "Used by scripts/train.py with configurable encoder-decoder combinations",
        ha="center",
        va="center",
        fontsize=11,
        color="#4b5563",
    )

    _box(ax, 0.05, 0.38, 0.14, 0.22, "Input tensor\nC x H x W\n(multimodal features)", fc="#dbeafe", ec="#1d4ed8")
    _box(ax, 0.25, 0.34, 0.18, 0.30, "Backbone Encoder\n(resnet / convnext /\nconvnextv2 / coatnet / vgg)", fc="#ede9fe", ec="#6d28d9")
    _box(ax, 0.50, 0.34, 0.16, 0.30, "Decoder Head\n(unet / fpn /\nunet++ / upernet /\ndeeplabv3+)", fc="#dcfce7", ec="#15803d")
    _box(ax, 0.72, 0.38, 0.11, 0.22, "Segmentation\nHead\n1x1 Conv", fc="#fee2e2", ec="#b91c1c")
    _box(ax, 0.87, 0.38, 0.10, 0.22, "Mask logits\n1 x H x W\n(sigmoid)", fc="#fef3c7", ec="#b45309")

    _arrow(ax, 0.19, 0.49, 0.25, 0.49)
    _arrow(ax, 0.43, 0.49, 0.50, 0.49)
    _arrow(ax, 0.66, 0.49, 0.72, 0.49)
    _arrow(ax, 0.83, 0.49, 0.87, 0.49)

    _box(
        ax,
        0.25,
        0.12,
        0.41,
        0.13,
        "Losses: BCE / Dice / BCE+Dice / Focal\nwith pixel-level weight_map + ignore_mask",
        fc="#f9fafb",
        ec="#6b7280",
        fontsize=10,
    )
    _arrow(ax, 0.78, 0.38, 0.63, 0.22, color="#6b7280", lw=1.4)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_temporal_architecture(path: Path) -> None:
    fig, ax = _setup_ax(16, 8)
    ax.text(0.5, 0.95, "Temporal Dual-Head FiLM U-Net Architecture", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(
        0.5,
        0.90,
        "Used by film_temporal_unet / film_temporal_unet_plus",
        ha="center",
        va="center",
        fontsize=11,
        color="#4b5563",
    )

    _box(ax, 0.04, 0.58, 0.17, 0.20, "Temporal input\nT x C x H x W\n(or flattened)", fc="#dbeafe", ec="#1d4ed8")
    _box(ax, 0.04, 0.25, 0.17, 0.18, "Condition vector\n(geo / quality /\nseason metadata)", fc="#ffedd5", ec="#c2410c")

    _box(ax, 0.27, 0.55, 0.14, 0.24, "Stem\n2D conv\nfeature prep", fc="#ede9fe", ec="#6d28d9")
    _box(ax, 0.45, 0.55, 0.16, 0.24, "Encoder\nwith FiLM\nmodulation", fc="#e0e7ff", ec="#4338ca")
    _box(ax, 0.65, 0.55, 0.14, 0.24, "Bottleneck\n(context block)", fc="#dcfce7", ec="#15803d")
    _box(ax, 0.83, 0.55, 0.13, 0.24, "Decoder\n(skip fusion)", fc="#ccfbf1", ec="#0f766e")

    _box(ax, 0.80, 0.20, 0.15, 0.16, "Mask head\n1 x H x W", fc="#fee2e2", ec="#b91c1c")
    _box(ax, 0.60, 0.20, 0.15, 0.16, "Time head\nK x H x W\n(time bins)", fc="#fef3c7", ec="#b45309")

    _arrow(ax, 0.21, 0.68, 0.27, 0.68)
    _arrow(ax, 0.41, 0.68, 0.45, 0.68)
    _arrow(ax, 0.61, 0.68, 0.65, 0.68)
    _arrow(ax, 0.79, 0.68, 0.83, 0.68)

    _box(ax, 0.35, 0.24, 0.18, 0.16, "FiLM MLP\n(gamma, beta)", fc="#fff7ed", ec="#9a3412")
    _arrow(ax, 0.21, 0.34, 0.35, 0.32, color="#9a3412", lw=1.5)
    _arrow(ax, 0.53, 0.32, 0.53, 0.55, color="#9a3412", lw=1.5)

    _arrow(ax, 0.89, 0.55, 0.88, 0.36)
    _arrow(ax, 0.85, 0.55, 0.68, 0.36)

    _box(
        ax,
        0.30,
        0.04,
        0.44,
        0.11,
        "Training objective: mask_loss + lambda_time * time_loss\nTargets: fused binary mask + event-time bins",
        fc="#f9fafb",
        ec="#6b7280",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_family_overview(path: Path) -> None:
    fig, ax = _setup_ax(14, 7)
    ax.text(0.5, 0.94, "Model Family Overview", ha="center", va="center", fontsize=18, weight="bold")

    _box(ax, 0.40, 0.78, 0.20, 0.11, "Multimodal Inputs", fc="#dbeafe", ec="#1d4ed8", fontsize=12)
    _arrow(ax, 0.50, 0.78, 0.28, 0.63)
    _arrow(ax, 0.50, 0.78, 0.72, 0.63)

    _box(ax, 0.12, 0.50, 0.32, 0.16, "Snapshot Segmentation Branch\n(resnet/convnext/... + decoder head)", fc="#ede9fe", ec="#6d28d9")
    _box(ax, 0.56, 0.50, 0.32, 0.16, "Temporal FiLM Branch\n(dual-head U-Net with condition vector)", fc="#dcfce7", ec="#15803d")

    _arrow(ax, 0.28, 0.50, 0.28, 0.34)
    _arrow(ax, 0.72, 0.50, 0.72, 0.34)

    _box(ax, 0.14, 0.20, 0.28, 0.12, "Output: Deforestation mask\n(1 x H x W)", fc="#fee2e2", ec="#b91c1c")
    _box(ax, 0.58, 0.20, 0.28, 0.12, "Outputs: mask + event-time\n(1 x H x W, K x H x W)", fc="#fef3c7", ec="#b45309")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_snapshot_architecture(OUT_DIR / "model_architecture_snapshot.png")
    generate_temporal_architecture(OUT_DIR / "model_architecture_temporal.png")
    generate_family_overview(OUT_DIR / "model_architecture_overview.png")


if __name__ == "__main__":
    main()
