import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PET-MRI AI Agent | NCISRT 2026",
    page_icon="🧠",
    layout="wide",
)

# --- CUSTOM STYLING ---
st.markdown(
    """
<style>
    .main-banner {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        padding: 1.4rem;
        border-radius: 10px;
        border-left: 6px solid #14B8A6;
        color: white;
        margin-bottom: 1.2rem;
    }
    .badge {
        background-color: #14B8A6;
        color: #0F172A;
        padding: 4px 10px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.82rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- HEADER BANNER ---
st.markdown(
    """
<div class="main-banner">
    <span class="badge">NCISRT 2026 • LIVE AI AGENT SIMULATION</span>
    <h2 style="margin-top: 8px; margin-bottom: 4px;">AI-Driven Cross-Modality Motion Correction & Scan Acceleration in Hybrid PET-MRI</h2>
    <p style="color: #94A3B8; margin-bottom: 0;">
        <b>Author:</b> Radiography Grad &nbsp;|&nbsp; B.Tech IT &nbsp;|&nbsp; MBA HR &nbsp;•&nbsp; <i>Healthcare Data Specialist / Clinical Data Analyst</i>
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# --- INTERACTIVE CONTROLS (SIDEBAR) ---
st.sidebar.header("🎛️ Scanner & AI Agent Controls")
resp_amp = st.sidebar.slider(
    "Respiratory Drift Severity (mm)", 1.0, 6.0, 4.2, 0.2
)
resp_rate = st.sidebar.slider(
    "Breathing Rate (breaths/min)", 8, 24, 15, 1
)
include_twitch = st.sidebar.checkbox(
    "Inject Sudden Cough / Micro-Twitch", value=True
)
agent_active = st.sidebar.toggle("Dual-Stream CNN Agent Active", value=True)
accel_factor = st.sidebar.select_slider(
    "Stream-1 k-Space Acceleration",
    options=["1x (Standard)", "2x (Fast)", "4x (Ultra-Fast AI)"],
    value="4x (Ultra-Fast AI)",
)
run_live = st.sidebar.button(
    "▶️ Simulate Real-Time Acquisition", use_container_width=True
)

# --- SIMULATION PHYSICS ENGINE ---
np.random.seed(42)
frames = 120
t = np.linspace(0, 20, frames)
freq_hz = resp_rate / 60.0

# 1. Raw Patient Motion (Sinusoidal Breathing + Sudden Micro-Twitch + Sensor Noise)
breathing = resp_amp * np.sin(2 * np.pi * freq_hz * t)
twitch = (
    np.where((t > 10.5) & (t < 14.0), 2.4 * np.sin(2 * np.pi * 0.85 * t), 0.0)
    if include_twitch
    else np.zeros_like(t)
)
noise = np.random.normal(0, 0.14, frames)
raw_drift = breathing + twitch + noise

# 2. Stream-2 AI Agent Inverse Transformation Matrix (1-Frame Feedback Loop)
ai_inverse = np.zeros(frames)
residual_error = np.zeros(frames)

for i in range(frames):
    if agent_active and i >= 1:
        # Agent perceives MRI navigator echo at t-1 and applies inverse matrix to PET sinogram
        ai_inverse[i] = -raw_drift[i - 1] * 0.97
    residual_error[i] = raw_drift[i] + ai_inverse[i]

# 3. Stream-1 Clinical & Operational KPIs
base_mins = 30.0
if accel_factor == "1x (Standard)":
    scan_mins, cnr_val = 30.0, 14.2
elif accel_factor == "2x (Fast)":
    scan_mins, cnr_val = 22.5, (16.8 if agent_active else 10.4)
else:
    scan_mins, cnr_val = 18.0, (19.5 if agent_active else 7.2)

time_saved_pct = int(((base_mins - scan_mins) / base_mins) * 100)
peak_error = float(np.max(np.abs(residual_error[5:])))
lock_badge = "VOXEL LOCKED (<0.6 mm)" if peak_error < 0.65 else "MOTION BLUR"

# --- TOP KPI ROW ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Hybrid Scan Duration", f"{scan_mins:.1f} min", f"-{time_saved_pct}% Time")
k2.metric(
    "Peak Voxel Drift",
    f"{peak_error:.2f} mm",
    lock_badge,
    delta_color="normal" if peak_error < 0.65 else "inverse",
)
k3.metric(
    "Contrast-to-Noise (CNR)",
    f"{cnr_val:.1f} dB",
    "+37% Clarity"
    if (agent_active and accel_factor == "4x (Ultra-Fast AI)")
    else "Standard/Degraded",
)
k4.metric(
    "Post-Scan Alignment",
    "0.0 min" if agent_active else "12.0 min",
    "Automated" if agent_active else "Manual Needed",
)

st.divider()

# --- HELPER FUNCTION TO BUILD MOTION GRAPH ---
def build_motion_chart(end_idx):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t[:end_idx],
            y=raw_drift[:end_idx],
            mode="lines",
            name="Uncorrected Patient Drift (mm)",
            line=dict(color="#EF4444", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t[:end_idx],
            y=residual_error[:end_idx],
            mode="lines",
            name="AI Agent Residual Error (mm)",
            line=dict(color="#10B981" if agent_active else "#F59E0B", width=3),
        )
    )
    fig.add_hrect(
        y0=-0.5,
        y1=0.5,
        fillcolor="#14B8A6",
        opacity=0.15,
        line_width=0,
        annotation_text="Sub-Voxel Diagnostic Target (±0.5 mm)",
        annotation_position="top left",
    )
    fig.update_layout(
        xaxis_title="Acquisition Timeline (Seconds)",
        yaxis_title="Spatial Displacement (mm)",
        xaxis_range=[0, 20],
        yaxis_range=[-7.5, 7.5],
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(l=10, r=10, t=35, b=10),
        height=380,
    )
    return fig


# --- MAIN DASHBOARD VISUALS ---
col_graph, col_phantom = st.columns([1.45, 1])

with col_graph:
    st.subheader("📈 Stream-2: Real-Time Sinogram Motion Tracking")
    chart_slot = st.empty()

    if run_live:
        for step in range(10, frames + 1, 6):
            chart_slot.plotly_chart(build_motion_chart(step), use_container_width=True)
            time.sleep(0.06)
    else:
        chart_slot.plotly_chart(build_motion_chart(frames), use_container_width=True)

with col_phantom:
    st.subheader("🔬 Stream-1 & 2: Synthetic PET-MRI Fusion")
    size = 120
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]

    # Structural MRI Anatomy + Focal Metabolic PET Lesion
    mri_struct = np.zeros((size, size))
    mri_struct[(x**2 + y**2) <= 48**2] = 0.4
    mri_struct[(x**2 + y**2) <= 42**2] = 0.85
    pet_lesion = np.exp(-((x - 14) ** 2 + (y + 10) ** 2) / (2 * 4.8**2))

    if not agent_active:
        shift_px = max(1, int(resp_amp * 2.2))
        smeared_pet = np.zeros_like(pet_lesion)
        for s in range(-shift_px, shift_px + 1):
            smeared_pet += np.roll(pet_lesion, s, axis=1) / (2 * shift_px + 1)
        noise_lvl = 0.24 if accel_factor == "4x (Ultra-Fast AI)" else 0.08
        fused = (
            gaussian_filter(mri_struct, sigma=1.7)
            + smeared_pet * 1.4
            + np.random.normal(0, noise_lvl, (size, size))
        )
        img_note = "⚠️ Agent OFF: Motion smearing & undersampling noise obscure lesion margins."
    else:
        fused = mri_struct + pet_lesion * 1.65 + np.random.normal(0, 0.03, (size, size))
        img_note = "✅ Agent ON: Full-resolution DLR + inverse matrix preserves focal lesion sharpness."

    fig_img = go.Figure(
        data=go.Heatmap(z=fused, colorscale="Viridis", showscale=False)
    )
    fig_img.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(
            showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x"
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=325,
    )
    st.plotly_chart(fig_img, use_container_width=True)
    st.caption(img_note)

# --- TELEMETRY EXPORT & RECRUITER CONNECT ---
st.divider()
c_ops, c_cta = st.columns([1.45, 1])

with c_ops:
    st.markdown("### 🏥 Clinical & Hospital Operations Impact")
    st.markdown("""
    - **40% Scan Acceleration:** Cuts standard 30-min protocols to 18 mins, adding **4–5 patient slots per shift**.
    - **Zero Call-Back Overhead:** Real-time PET sinogram correction prevents repeat scans from breathing artifacts.
    - **Hardware-Agnostic Upgrade:** Software-only CNN pipeline modernizes legacy PET-MRI scanners.
    """)

with c_cta:
    st.markdown("### 📊 Telemetry Data & Connect")
    telemetry_df = pd.DataFrame({
        "Time_Sec": np.round(t, 2),
        "Raw_Patient_Drift_mm": np.round(raw_drift, 3),
        "AI_Inverse_Correction_mm": np.round(ai_inverse, 3),
        "Residual_Voxel_Error_mm": np.round(residual_error, 3),
    })
    st.download_button(
        label="⬇️ Download Simulation Telemetry (CSV)",
        data=telemetry_df.to_csv(index=False).encode("utf-8"),
        file_name="NCISRT_2026_PET_MRI_Telemetry.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.link_button(
        "🔗 Connect on LinkedIn",
        "https://linkedin.com",
        use_container_width=True,
    )
