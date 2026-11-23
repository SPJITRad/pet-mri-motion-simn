"""Live simulation companion to the NCISRT 2026 ePoster:
'An AI-Driven Framework for Cross-Modality Motion Correction and Scan
Acceleration in Hybrid PET-MRI Systems' - Simi Paxleal J.
Run locally:  streamlit run app.py      Deploy free: see DEPLOY.md
"""
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

NAVY, PANEL, GRIDC = "#0B1830", "#13233F", "#23415F"
CYAN, AMBER, TEXTC, MUTED = "#35E1FF", "#FFB454", "#EAF2FB", "#A9BDD6"

GITHUB_PROFILE = "https://github.com/SPJITRad"
EPOSTER_URL = "https://github.com/SPJITRad/pet-mri-motion-simn/blob/main/ePoster_NCISRT2026_SIMI_PAXLEAL.pdf"

st.set_page_config(page_title="AI PET-MRI Motion Correction", page_icon="🧠", layout="wide")
st.markdown(f"""
<style>
  .stApp, [data-testid="stAppViewContainer"] {{ background-color: {NAVY} !important; }}
  [data-testid="stSidebar"] {{ background-color: {PANEL} !important; border-right: 1px solid {GRIDC}; }}
  h1, h2, h3 {{ color: {TEXTC} !important; }}
  p, span, label, .stCaption {{ color: {MUTED} !important; }}
  [data-testid="stMetricValue"] {{ color: {CYAN} !important; }}
  [data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}
  .stMarkdown a {{ color: {CYAN} !important; }}
</style>""", unsafe_allow_html=True)

st.markdown("## 🧠 Dual-Stream CNN · Live Motion-Correction Simulation")
st.caption("Companion demo · NCISRT 2026 ePoster — An AI-Driven Framework for Cross-Modality Motion Correction and Scan Acceleration in Hybrid PET-MRI Systems · Simi Paxleal J.")

with st.sidebar:
    st.header("Patient & protocol")
    resp  = st.slider("Respiratory rate (breaths/min)", 8, 24, 14)
    drift = st.slider("Baseline drift (mm/min)", 0.05, 0.30, 0.15, 0.01)
    micro = st.slider("Micro-movement noise σ (mm)", 0.00, 0.20, 0.05, 0.01)
    R     = st.slider("Stream-1 · k-space acceleration (×)", 2, 8, 4)
    lock  = st.toggle("Stream-2 · AI motion lock", value=True)
    st.divider()
    t_now = st.slider("Scan clock (min)", 0.0, 30.0, 20.0, 0.5)

t = np.linspace(0, 30, 1200)
rng = np.random.default_rng(11)
breath = (0.10 + 0.012 * t) * np.sin(2 * np.pi * (resp / 60.0) * t)
noise = np.convolve(rng.normal(0, micro, t.size), np.ones(15) / 15, mode="same")
unc = np.clip(drift * t + breath + noise, 0, None)
resid = np.clip(np.abs(0.16 + 0.10 * np.sin(2 * np.pi * (resp / 60.0) * t) + rng.normal(0, 0.04, t.size)), 0.03, 0.5)

mri_frac = 0.45
saved = min(40.0, 100 * mri_frac * (1 - 1 / R) + (4.0 if lock else 0.0))
cnr = (10 + 3 * np.log(R)) if lock else (3 + 1.5 * np.log(R))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Scan time saved", f"{saved:.0f}%")
c2.metric("Peak uncorrected drift", f"{float(unc.max()):.1f} mm")
c3.metric("Residual (AI lock)", f"≤ {float(resid.max()):.2f} mm" if lock else "OFF")
c4.metric("CNR gain", f"+{cnr:.0f}%")

colA, colB = st.columns([3, 2])
with colA:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
    fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
    ax.axhspan(0, 0.5, color=CYAN, alpha=0.12)
    ax.plot(t, unc, color=AMBER, lw=2, ls=(0, (5, 3)), label="Uncorrected drift")
    if lock:
        ax.plot(t, resid, color=CYAN, lw=2, label="AI motion-locked")
    ax.axvline(t_now, color=TEXTC, ls=":", lw=1.2)
    ax.set_xlim(0, 30); ax.set_ylim(-0.3, max(5.4, float(unc.max()) + 0.6))
    ax.set_xlabel("scan time (min)", color=MUTED); ax.set_ylabel("residual drift (mm)", color=MUTED)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values(): s.set_color(GRIDC)
    ax.legend(frameon=False, labelcolor=TEXTC, loc="upper left")
    st.pyplot(fig); plt.close(fig)
with colB:
    off = float(np.interp(t_now, t, unc)); d = np.clip(off * 0.5, -2.2, 2.2)
    fig2, ax2 = plt.subplots(figsize=(3.9, 3.6), dpi=140)
    fig2.patch.set_facecolor(PANEL); ax2.set_facecolor(PANEL)
    ax2.set_xlim(-8, 8); ax2.set_ylim(-3.1, 3.1); ax2.set_aspect("equal"); ax2.axis("off")
    ax2.add_patch(Ellipse((-4.4, 0), 3.4, 4.6, fill=False, ec=CYAN, lw=2))
    ax2.add_patch(Circle((-4.4 + d, 0.9), 0.75, color=AMBER, alpha=0.9))
    ax2.annotate("", xy=(-4.4 + d, 0.25), xytext=(-4.4, 0.25), arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.5))
    ax2.text(-4.4 + d / 2, -0.95, f"\u0394 = {off:.1f} mm", color=AMBER, fontsize=8, ha="center")
    ax2.text(-4.4, -2.7, "uncorrected · PET–MRI offset", color=MUTED, fontsize=8, ha="center")
    ax2.add_patch(Ellipse((4.4, 0), 3.4, 4.6, fill=False, ec=CYAN, lw=2))
    ax2.add_patch(Circle((4.4, 0.55), 0.75, color=AMBER, alpha=0.9))
    ax2.text(4.4, -2.7, "AI-locked · ≤ 0.5 mm", color=CYAN, fontsize=8, ha="center")
    st.pyplot(fig2); plt.close(fig2)

with st.expander("🔬 How the dual-stream CNN works"):
    st.markdown(
        f"**Stream 1 — DLR k-space reconstruction.** Full-resolution MRI is reconstructed from heavily "
        f"under-sampled k-space (the ×{R} slider), shortening the MR portion of the hybrid acquisition.\n\n"
        "**Stream 2 — 6-DoF co-registration.** Ultra-fast MR navigator sequences track respiration and "
        "micro-movement; a rigid 6-DoF transform plus a non-rigid deformation field is inverted and applied "
        "to PET sinogram data in real time, holding voxel-to-voxel alignment for the whole scan.\n\n"
        "Both run **inside the acquisition pipeline** — no gating hardware, no breath-hold protocols, no manual post-processing.")
with st.expander("📐 Reading the numbers (honesty box)"):
    st.markdown(
        "- **Scan-time saved** — illustrative model: ~45% of a hybrid slot is MR time; ×R under-sampling removes "
        "(1 − 1/R) of it, plus a small respiratory-efficiency term when the lock is on. Capped at the poster's 'up to 40%'.\n"
        "- **Residual drift** — simulation of the corrected error band (≤ 0.5 mm). The poster's fixed example peaks at ≈ 4.5 mm uncorrected.\n"
        "- **CNR gain** — from DLR denoising plus motion-lock (signal stops smearing across misaligned voxels).\n\n"
        "**This is a conceptual framework demonstrated on visual simulation models — not clinical data.** "
        "Validation roadmap: motion-simulated phantoms → retrospective paired PET-MRI → prospective studies.")

st.divider()

def _chip(url, label):
    st.markdown(
        f'<a href="{url}" target="_blank" rel="noopener" style="display:inline-block;'
        f"background:#0F2138;border:1px solid #2C4E74;border-radius:8px;padding:7px 18px;"
        f"text-decoration:none;color:{CYAN};font-weight:600;font-size:13px;"
        f'">{label}</a>',
        unsafe_allow_html=True,
    )

_fc1, _fc2 = st.columns([1, 1])
with _fc1:
    _chip(GITHUB_PROFILE, "\u2b50 Source code \u00b7 GitHub")
with _fc2:
    _chip(EPOSTER_URL, "\U0001f4c4 Companion ePoster \u00b7 PDF")

st.caption("Conceptual framework · visual simulation models, not clinical data · Companion to the NCISRT 2026 ePoster · Simi Paxleal J. · Dr. Jeyasekharan Medical Trust College of Allied Health Sciences, Nagercoil")
