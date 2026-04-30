"""
Object Detection Lab
====================
A clean YOLOv8 Streamlit app with:
  Tab 1 – Single image detection
  Tab 2 – Batch test (upload multiple images)
  Tab 3 – Confidence matrix & analytics
"""

import matplotlib.pyplot as plt
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import yaml
import requests
from pathlib import Path
from io import BytesIO

from utils.detector import ObjectDetector
from utils.visualization import (
    annotate_image,
    confidence_bar_chart,
    class_frequency_chart,
    confidence_matrix,
)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Object Detection Lab",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0a0a0a 0%, #0d1f2d 50%, #001a2c 100%);
    border: 1px solid #1a3a4a;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(0,212,255,0.07) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700;
    color: #00d4ff; margin: 0 0 6px;
    letter-spacing: -1px;
}
.hero p { color: #7a9aaa; font-size: 0.95rem; margin: 0; }

.metric-card {
    background: #0d1f2d;
    border: 1px solid #1a3a4a;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700; color: #00d4ff;
}
.metric-card .lbl { font-size: 0.8rem; color: #7a9aaa; margin-top: 4px; }

.det-pill {
    display: inline-block;
    background: #0d2233;
    border: 1px solid #1a4060;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 3px;
    font-size: 0.82rem; color: #00d4ff;
    font-family: 'Space Mono', monospace;
}

.section-label {
    font-size: 0.75rem; font-weight: 600;
    color: #7a9aaa; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── load config ──────────────────────────────────────────────────────────────
_cfg_path = Path(__file__).parent / 'config.yaml'
with open(_cfg_path) as f:
    config = yaml.safe_load(f)

# ── session state ─────────────────────────────────────────────────────────────
if 'detector' not in st.session_state:
    with st.spinner("Loading YOLOv8 model…"):
        st.session_state.detector = ObjectDetector(config)

if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []   # list of {name, detections, time}

detector: ObjectDetector = st.session_state.detector

# ── hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎯 Object Detection Lab</h1>
  <p>YOLOv8 · COCO 80-class · Real confidence analysis</p>
</div>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    conf_thresh = st.slider(
        "Confidence threshold", 0.10, 0.95,
        value=config['detection']['confidence_threshold'], step=0.05,
        help="Only show detections above this score"
    )
    iou_thresh = st.slider(
        "IOU threshold (NMS)", 0.10, 0.95,
        value=config['detection']['iou_threshold'], step=0.05,
        help="Non-max suppression overlap cutoff"
    )
    st.divider()
    st.markdown("### 🎨 Display")
    show_labels = st.checkbox("Show class labels", value=True)
    show_conf   = st.checkbox("Show confidence on boxes", value=True)
    st.divider()
    st.markdown("### 🤖 Model")
    model_choice = st.selectbox(
        "YOLOv8 variant",
        ["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
        index=0,
        help="n=fastest, l=most accurate"
    )
    if st.button("Reload model"):
        config['model']['name'] = model_choice
        with st.spinner(f"Loading {model_choice}…"):
            st.session_state.detector = ObjectDetector(config)
        st.success("Model reloaded!")
        st.rerun()

    if st.session_state.batch_results:
        st.divider()
        if st.button("🗑 Clear batch results"):
            st.session_state.batch_results = []
            st.rerun()


# ── helper ────────────────────────────────────────────────────────────────────
def load_image_to_pil(source) -> Image.Image | None:
    """Accept UploadedFile, BytesIO, or URL string → PIL RGB image."""
    try:
        if isinstance(source, str):
            r = requests.get(source, timeout=10)
            r.raise_for_status()
            source = BytesIO(r.content)
        return Image.open(source).convert("RGB")
    except Exception as e:
        st.error(f"Could not load image: {e}")
        return None


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(arr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📷  Single Image", "📦  Batch Test", "📊  Analytics"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single image
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    src_mode = st.radio("Source", ["Upload file", "Paste URL"], horizontal=True,
                        label_visibility="collapsed")

    pil_image = None
    if src_mode == "Upload file":
        upl = st.file_uploader("Drop an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if upl:
            pil_image = load_image_to_pil(upl)
    else:
        url = st.text_input("Image URL", placeholder="https://…")
        if url:
            pil_image = load_image_to_pil(url)

    if pil_image:
        col_orig, col_det = st.columns(2, gap="medium")

        with col_orig:
            st.markdown('<div class="section-label">Original</div>', unsafe_allow_html=True)
            st.image(pil_image, use_container_width=True)

        with col_det:
            st.markdown('<div class="section-label">Detections</div>', unsafe_allow_html=True)
            run = st.button("🔍 Run Detection", type="primary", use_container_width=True)

            if run:
                t0 = time.perf_counter()
                image_bgr = pil_to_bgr(pil_image)
                results = detector.detect(image_bgr, conf_thresh, iou_thresh)
                elapsed = time.perf_counter() - t0

                annotated_bgr = annotate_image(
                    image_bgr, results,
                    show_labels=show_labels,
                    show_confidence=show_conf,
                    line_thickness=config['visualization']['line_thickness'],
                    font_scale=config['visualization']['font_scale'],
                )
                st.image(bgr_to_rgb(annotated_bgr), use_container_width=True)

                # ── metrics row
                n = len(results['boxes'])
                avg_c = np.mean(results['confidences']) if results['confidences'] else 0.0
                top_c = max(results['confidences'], default=0.0)

                m1, m2, m3, m4 = st.columns(4)
                for col, val, lbl in zip(
                    [m1, m2, m3, m4],
                    [n, f"{avg_c:.0%}", f"{top_c:.0%}", f"{elapsed*1000:.0f}ms"],
                    ["Objects", "Avg Conf", "Top Conf", "Inference"],
                ):
                    col.markdown(
                        f'<div class="metric-card"><div class="val">{val}</div>'
                        f'<div class="lbl">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

                # ── detected pills
                if results['class_names']:
                    st.markdown("")
                    pills = "".join(
                        f'<span class="det-pill">{cn} {cf:.0%}</span>'
                        for cn, cf in zip(results['class_names'], results['confidences'])
                    )
                    st.markdown(pills, unsafe_allow_html=True)

                # ── confidence bar chart
                chart = confidence_bar_chart(results)
                if chart:
                    st.markdown("")
                    st.image(chart, use_container_width=True)

                # ── download
                buf = BytesIO()
                Image.fromarray(bgr_to_rgb(annotated_bgr)).save(buf, format="PNG")
                st.download_button(
                    "📥 Download annotated image",
                    data=buf.getvalue(),
                    file_name="detection_result.png",
                    mime="image/png",
                    use_container_width=True,
                )

                # ── stash in batch results for analytics
                st.session_state.batch_results.append({
                    'name': getattr(upl, 'name', 'url_image') if src_mode == "Upload file" else 'url_image',
                    'detections': results,
                    'time': elapsed,
                })
    else:
        st.info("Upload an image or paste a URL to get started.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Batch test
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        "Upload several images at once. "
        "Results are saved automatically and appear in the **Analytics** tab.",
    )

    uploaded_batch = st.file_uploader(
        "Select images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_batch:
        if st.button("▶️ Run Batch Detection", type="primary", use_container_width=True):
            progress = st.progress(0, text="Processing…")
            new_results = []

            for i, upl_file in enumerate(uploaded_batch):
                pil_img = load_image_to_pil(upl_file)
                if pil_img is None:
                    continue

                t0 = time.perf_counter()
                bgr = pil_to_bgr(pil_img)
                dets = detector.detect(bgr, conf_thresh, iou_thresh)
                elapsed = time.perf_counter() - t0

                new_results.append({
                    'name': upl_file.name,
                    'detections': dets,
                    'time': elapsed,
                })
                progress.progress(
                    (i + 1) / len(uploaded_batch),
                    text=f"{upl_file.name} — {len(dets['boxes'])} objects found",
                )

            st.session_state.batch_results.extend(new_results)
            st.success(f"✅ Processed {len(new_results)} images. "
                       "Switch to the **Analytics** tab to explore results.")

    # show thumbnail grid with detection counts
    if st.session_state.batch_results:
        st.divider()
        st.markdown(f"**{len(st.session_state.batch_results)} images in batch**")

        cols = st.columns(min(4, len(st.session_state.batch_results)))
        for idx, res in enumerate(st.session_state.batch_results):
            with cols[idx % 4]:
                n_det = len(res['detections']['boxes'])
                avg_c = (np.mean(res['detections']['confidences'])
                         if res['detections']['confidences'] else 0.0)
                st.markdown(
                    f"**{res['name']}**  \n"
                    f"`{n_det}` objects · avg {avg_c:.0%} · {res['time']*1000:.0f}ms"
                )
    else:
        st.info("No batch results yet. Run single detections (Tab 1) or upload a batch here.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Analytics
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if not st.session_state.batch_results:
        st.info("Run some detections first (Tab 1 or Tab 2) — "
                "results accumulate here automatically.")
    else:
        all_dets = [r['detections'] for r in st.session_state.batch_results]
        all_times = [r['time'] for r in st.session_state.batch_results]

        # ── summary metrics
        total_objects = sum(len(d['boxes']) for d in all_dets)
        all_confs = [c for d in all_dets for c in d['confidences']]
        avg_conf = np.mean(all_confs) if all_confs else 0.0
        avg_ms   = np.mean(all_times) * 1000

        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl in zip(
            [m1, m2, m3, m4],
            [len(all_dets), total_objects, f"{avg_conf:.1%}", f"{avg_ms:.0f}ms"],
            ["Images", "Total Objects", "Avg Confidence", "Avg Inference"],
        ):
            col.markdown(
                f'<div class="metric-card"><div class="val">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── confidence matrix (primary chart)
        st.markdown("#### Confidence Matrix")
        st.caption(
            "Rows = detected classes · Columns = confidence buckets · "
            "Cell = number of detections. Brighter = more detections."
        )
        cm_img = confidence_matrix(all_dets)
        if cm_img:
            st.image(cm_img, use_container_width=True)

        st.markdown("---")

        # ── class frequency
        col_freq, col_conf = st.columns(2, gap="large")

        with col_freq:
            st.markdown("#### Class Frequency")
            freq_img = class_frequency_chart(all_dets)
            if freq_img:
                st.image(freq_img, use_container_width=True)

        with col_conf:
            st.markdown("#### Confidence Distribution")
            if all_confs:
                fig, ax = plt.subplots(figsize=(5, 3.8), facecolor='#0e1117')
                ax.set_facecolor('#0e1117')
                ax.hist(all_confs, bins=20, color='#00d4ff',
                        edgecolor='none', alpha=0.85)
                ax.axvline(avg_conf, color='#ff6b35', linestyle='--',
                           linewidth=1.5, label=f'Mean {avg_conf:.1%}')
                ax.set_xlabel('Confidence', color='#888888', fontsize=9)
                ax.set_ylabel('Count',      color='#888888', fontsize=9)
                ax.set_title('Confidence Histogram', color='white',
                             fontsize=11, fontweight='bold')
                ax.legend(frameon=False, labelcolor='#ff6b35', fontsize=9)
                ax.tick_params(colors='#666666')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight',
                            facecolor='#0e1117')
                buf.seek(0)
                st.image(Image.open(buf), use_container_width=True)
                plt.close(fig)

        # ── per-image table
        st.markdown("---")
        st.markdown("#### Per-image Breakdown")
        table_data = []
        for r in st.session_state.batch_results:
            d = r['detections']
            confs = d['confidences']
            unique_cls = list(set(d['class_names']))
            table_data.append({
                "Image": r['name'],
                "Objects": len(d['boxes']),
                "Classes": ", ".join(sorted(unique_cls)) or "—",
                "Avg Conf": f"{np.mean(confs):.1%}" if confs else "—",
                "Max Conf": f"{max(confs):.1%}" if confs else "—",
                "Time (ms)": f"{r['time']*1000:.0f}",
            })
        st.dataframe(table_data, use_container_width=True, hide_index=True)

# needed for histogram inside tab3
import matplotlib.pyplot as plt