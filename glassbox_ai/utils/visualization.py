import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image


# ── colour palette (one distinct colour per class bucket) ──────────────────
_PALETTE = [
    (0, 212, 255), (255, 87, 34), (76, 175, 80), (156, 39, 176),
    (255, 193, 7),  (233, 30, 99), (0, 150, 136), (103, 58, 183),
    (244, 67, 54),  (33, 150, 243),
]

def _class_color(class_id: int) -> tuple:
    """Return a BGR colour for a given class id."""
    r, g, b = _PALETTE[class_id % len(_PALETTE)]
    return (b, g, r)


# ── image annotation ────────────────────────────────────────────────────────
def annotate_image(image_bgr: np.ndarray, detections: dict,
                   show_labels: bool = True,
                   show_confidence: bool = True,
                   line_thickness: int = 2,
                   font_scale: float = 0.55) -> np.ndarray:
    """Draw bounding boxes + labels onto a BGR image copy."""
    out = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box, conf, cls_id, cls_name in zip(
        detections['boxes'],
        detections['confidences'],
        detections['class_ids'],
        detections['class_names'],
    ):
        x1, y1, x2, y2 = box
        color = _class_color(cls_id)

        # bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)

        # label
        parts = []
        if show_labels:
            parts.append(cls_name)
        if show_confidence:
            parts.append(f"{conf:.0%}")
        label = "  ".join(parts)

        if label:
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
            label_y = max(y1 - 6, th + 6)
            cv2.rectangle(out,
                          (x1, label_y - th - 6),
                          (x1 + tw + 8, label_y + baseline - 2),
                          color, -1)
            cv2.putText(out, label,
                        (x1 + 4, label_y - 2),
                        font, font_scale, (10, 10, 10), 1, cv2.LINE_AA)

    return out


# ── chart helpers ────────────────────────────────────────────────────────────
def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def confidence_bar_chart(detections: dict) -> Image.Image:
    """
    Horizontal bar chart — one bar per detection, coloured by class.
    Returns a PIL Image.
    """
    if not detections['confidences']:
        return None

    labels = [
        f"{n}  #{i+1}" for i, n in enumerate(detections['class_names'])
    ]
    confs  = detections['confidences']
    colors = [
        tuple(c / 255 for c in _PALETTE[cid % len(_PALETTE)])
        for cid in detections['class_ids']
    ]

    fig_h = max(3.0, len(labels) * 0.38 + 1.2)
    fig, ax = plt.subplots(figsize=(7, fig_h), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')

    y_pos = range(len(labels))
    bars = ax.barh(list(y_pos), confs, color=colors, height=0.62,
                   edgecolor='none')

    # value labels on bars
    for bar, conf in zip(bars, confs):
        ax.text(min(conf + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                f"{conf:.1%}", va='center', ha='left',
                color='white', fontsize=8.5, fontweight='bold')

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, color='#cccccc', fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel('Confidence', color='#888888', fontsize=9)
    ax.set_title('Detection Confidence Scores', color='white',
                 fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors='#666666')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_tick_params(labelcolor='#666666')
    ax.axvline(0.5, color='#333333', linestyle='--', linewidth=0.8)

    fig.tight_layout()
    return _fig_to_pil(fig)


def class_frequency_chart(all_detections: list) -> Image.Image:
    """
    Bar chart of detected class frequencies across one or more images.
    `all_detections` is a list of detection dicts.
    """
    counts: dict = {}
    for det in all_detections:
        for cls_name in det['class_names']:
            counts[cls_name] = counts.get(cls_name, 0) + 1

    if not counts:
        return None

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    classes, freqs = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.7 + 1.5), 4.5),
                           facecolor='#0e1117')
    ax.set_facecolor('#0e1117')

    bar_colors = [
        tuple(c / 255 for c in _PALETTE[i % len(_PALETTE)])
        for i in range(len(classes))
    ]
    bars = ax.bar(classes, freqs, color=bar_colors, edgecolor='none', width=0.65)

    for bar, freq in zip(bars, freqs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(freq), ha='center', va='bottom',
                color='white', fontsize=9, fontweight='bold')

    ax.set_ylabel('Count', color='#888888', fontsize=9)
    ax.set_title('Detected Class Frequency', color='white',
                 fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(axis='x', colors='#cccccc', rotation=30, labelsize=9)
    ax.tick_params(axis='y', colors='#666666')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylim(0, max(freqs) * 1.18)

    fig.tight_layout()
    return _fig_to_pil(fig)


def confidence_matrix(all_detections: list) -> Image.Image:
    """
    Heatmap: rows = classes detected, columns = confidence buckets.
    Cell value = number of detections in that (class, bucket) cell.
    """
    buckets     = ['25–40%', '40–55%', '55–70%', '70–85%', '85–100%']
    bucket_edges = [0.25, 0.40, 0.55, 0.70, 0.85, 1.01]

    # collect unique classes
    all_classes: list = []
    for det in all_detections:
        for cn in det['class_names']:
            if cn not in all_classes:
                all_classes.append(cn)

    if not all_classes:
        return None

    all_classes = sorted(all_classes)
    matrix = np.zeros((len(all_classes), len(buckets)), dtype=int)

    for det in all_detections:
        for cn, conf in zip(det['class_names'], det['confidences']):
            row = all_classes.index(cn)
            for b_idx in range(len(buckets)):
                if bucket_edges[b_idx] <= conf < bucket_edges[b_idx + 1]:
                    matrix[row, b_idx] += 1
                    break

    fig_h = max(3.5, len(all_classes) * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')

    cmap = LinearSegmentedColormap.from_list(
        'det', ['#0e1117', '#003f5c', '#00b4d8', '#00ffc8'])

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0)

    # cell annotations
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            txt_color = 'white' if val > matrix.max() * 0.4 else '#555555'
            ax.text(c, r, str(val), ha='center', va='center',
                    color=txt_color, fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets, color='#aaaaaa', fontsize=9)
    ax.set_yticks(range(len(all_classes)))
    ax.set_yticklabels(all_classes, color='#cccccc', fontsize=9)
    ax.set_xlabel('Confidence Bucket', color='#888888', fontsize=9, labelpad=8)
    ax.set_title('Confidence Matrix  (class × confidence bucket)',
                 color='white', fontsize=11, fontweight='bold', pad=12)
    for spine in ax.spines.values():
        spine.set_color('#333333')

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.ax.yaxis.set_tick_params(color='#666666', labelcolor='#666666')
    cb.outline.set_edgecolor('#333333')

    fig.tight_layout()
    return _fig_to_pil(fig)