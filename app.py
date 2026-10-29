"""
================================================================================
INTERACTIVE WEB SIMULATION DASHBOARD: YEAST-PCM-TEG HARVESTER
Run locally via: py -m streamlit run app.py
================================================================================
"""

import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(page_title="Yeast-PCM-TEG Multi-Physics Simulator", layout="wide")

st.title("⚡ Yeast-PCM-TEG Bio-Thermal Harvester: Digital Twin")
st.markdown("""
This interactive simulation solves coupled **Microbial Monod Growth Kinetics**, **Apparent Latent Heat Phase Transitions**, 
and **Dynamic Thermoelectric Transduction** to evaluate thermal pulse shaping and exergy recovery.
""")

# --- Sidebar Controls ---
st.sidebar.header("🔬 1. Bioreactor & Microbial Kinetics")
S_initial = st.sidebar.slider("Initial Glucose Concentration [g/L]", 40.0, 160.0, 100.0, 5.0)
X_initial = st.sidebar.slider("Initial Yeast Biomass Pitch [g/L]", 1.0, 10.0, 4.0, 0.5)
V_broth = st.sidebar.slider("Broth Volume [L]", 0.25, 1.0, 0.50, 0.05)
T_pitch = st.sidebar.slider("Initial Pitching Temperature [°C]", 22.0, 32.0, 26.0, 0.5)

st.sidebar.header("🧊 2. Phase Change Material (PCM) Buffer")
enable_pcm = st.sidebar.checkbox("Enable PCM Buffer Layer", value=True)
m_pcm_g = st.sidebar.slider("PCM Mass [grams]", 20.0, 150.0, 60.0, 5.0)
T_pcm_melt = st.sidebar.slider("PCM Melting Temperature [°C]", 25.0, 35.0, 28.0, 0.5)
L_f = st.sidebar.slider("Latent Heat of Fusion [kJ/kg]", 120.0, 240.0, 180.0, 10.0) * 1000.0

st.sidebar.header("🧱 3. Thermal Network & TEG Transducer")
R_insul = st.sidebar.slider("Insulation Resistance R_insul [K/W]", 10.0, 60.0, 35.0, 2.5)
R_th_teg = st.sidebar.slider("TEG Thermal Resistance R_teg [K/W]", 1.0, 4.0, 2.2, 0.2)
S_teg_mV = st.sidebar.slider("Seebeck Coefficient [mV/K]", 25.0, 60.0, 45.0, 1.0)
S_teg = S_teg_mV / 1000.0
T_amb = st.sidebar.slider("Ambient Temperature [°C]", 18.0, 28.0, 22.0, 0.5)

# --- Simulation Physics ---
m_broth = V_broth
C_broth = m_broth * 4184.0
m_pcm = m_pcm_g / 1000.0
Cp_pcm_sensible = 2000.0
delta_T_melt = 1.0
R_b_pcm = 2.0
R_contact = 0.5
R_sink_air = 2.0
m_sink = 0.15
C_sink = m_sink * 900.0
R_int = 2.5
t_hours = 20.0
t_span = (0.0, t_hours * 3600.0)
t_eval = np.linspace(0.0, t_hours * 3600.0, 1000)

def cardinal_growth_rate(T):
    if T <= 10.0 or T >= 38.5:
        return 0.0
    num = (T - 38.5) * (T - 10.0)**2
    den = (32.0 - 10.0) * ((32.0 - 10.0)*(T - 32.0) - (32.0 - 38.5)*(32.0 + 10.0 - 2.0*T))
    if den == 0.0:
        return 0.0
    return 0.48 * max(0.0, num / den)

def odes_buf(t, y):
    X, S, Tb, Tpcm, Tc = y
    if S > 0.001 and X < 22.0:
        mu = cardinal_growth_rate(Tb) * (S / (2.0 + S))
        dX_dt = (mu * X * (1.0 - X/22.0)) / 3600.0
        dS_dt = (-(1.0 / 0.16) * (dX_dt * 3600.0) - 0.015 * X) / 3600.0
    else:
        dX_dt, dS_dt = 0.0, 0.0
    Q_bio = V_broth * 655.0 * (-dS_dt)
    Q_loss = (Tb - T_amb) / R_insul
    Q_to_pcm = (Tb - Tpcm) / R_b_pcm
    Q_teg = (Tpcm - Tc) / (R_th_teg + R_contact)
    Q_sink = (Tc - T_amb) / R_sink_air
    
    sigma = delta_T_melt / 2.0
    dbeta_dT = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((Tpcm - T_pcm_melt)/sigma)**2)
    C_pcm_dyn = m_pcm * (Cp_pcm_sensible + L_f * dbeta_dT)
    
    dTb = (Q_bio - Q_loss - Q_to_pcm) / C_broth
    dTpcm = (Q_to_pcm - Q_teg) / C_pcm_dyn
    dTc = (Q_teg - Q_sink) / C_sink
    return [dX_dt, dS_dt, dTb, dTpcm, dTc]

def odes_unbuf(t, y):
    X, S, Tb, Tc = y
    if S > 0.001 and X < 22.0:
        mu = cardinal_growth_rate(Tb) * (S / (2.0 + S))
        dX_dt = (mu * X * (1.0 - X/22.0)) / 3600.0
        dS_dt = (-(1.0 / 0.16) * (dX_dt * 3600.0) - 0.015 * X) / 3600.0
    else:
        dX_dt, dS_dt = 0.0, 0.0
    Q_bio = V_broth * 655.0 * (-dS_dt)
    Q_loss = (Tb - T_amb) / R_insul
    Q_teg = (Tb - Tc) / (R_b_pcm + R_th_teg + R_contact)
    Q_sink = (Tc - T_amb) / R_sink_air
    dTb = (Q_bio - Q_loss - Q_teg) / C_broth
    dTc = (Q_teg - Q_sink) / C_sink
    return [dX_dt, dS_dt, dTb, dTc]

# --- Run Solvers ---
sol_unbuf = solve_ivp(odes_unbuf, t_span, [X_initial, S_initial, T_pitch, T_amb], t_eval=t_eval, method='Radau')
sol_buf = solve_ivp(odes_buf, t_span, [X_initial, S_initial, T_pitch, T_amb, T_amb], t_eval=t_eval, method='Radau')

time_h = t_eval / 3600.0
deltaT_unbuf = np.maximum(0.0, sol_unbuf.y[2] - sol_unbuf.y[3])
P_unbuf = ((S_teg * deltaT_unbuf)**2) / (4.0 * R_int)
E_unbuf = np.cumsum(P_unbuf * np.gradient(t_eval))[-1]

deltaT_buf = np.maximum(0.0, sol_buf.y[3] - sol_buf.y[4])
P_buf = ((S_teg * deltaT_buf)**2) / (4.0 * R_int)
E_buf = np.cumsum(P_buf * np.gradient(t_eval))[-1]

# --- Metrics Banner ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Unbuffered Peak Broth Temp", f"{np.max(sol_unbuf.y[2]):.2f} °C")
col2.metric("PCM-Buffered Peak Broth Temp", f"{np.max(sol_buf.y[2]):.2f} °C", f"-{np.max(sol_unbuf.y[2]) - np.max(sol_buf.y[2]):.2f} °C")
col3.metric("Harvested Energy (Unbuffered)", f"{E_unbuf:.2f} J")
col4.metric("Harvested Energy (Buffered)", f"{E_buf:.2f} J")

# --- Interactive Visualizations ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(time_h, sol_unbuf.y[2], 'r--', label='Broth (Unbuffered)')
axes[0, 0].plot(time_h, sol_buf.y[2], 'b-', label='Broth (PCM-Buffered)')
axes[0, 0].plot(time_h, sol_buf.y[3], 'g-.', label='PCM Buffer')
axes[0, 0].axhline(T_pcm_melt, color='gray', linestyle=':', label=f'PCM Melt ({T_pcm_melt}°C)')
axes[0, 0].set_title("Thermal Pulse Shaping [°C]")
axes[0, 0].set_xlabel("Time [h]")
axes[0, 0].grid(True, alpha=0.4)
axes[0, 0].legend()

axes[0, 1].plot(time_h, deltaT_unbuf, 'r--', label='ΔT Unbuffered')
axes[0, 1].plot(time_h, deltaT_buf, 'b-', label='ΔT Buffered')
axes[0, 1].set_title("Thermal Gradient Across TEG [°C]")
axes[0, 1].set_xlabel("Time [h]")
axes[0, 1].grid(True, alpha=0.4)
axes[0, 1].legend()

axes[1, 0].plot(time_h, P_unbuf*1000, 'r--', label='Unbuffered')
axes[1, 0].plot(time_h, P_buf*1000, 'b-', label='Buffered')
axes[1, 0].set_title("Instantaneous Power [mW]")
axes[1, 0].set_xlabel("Time [h]")
axes[1, 0].grid(True, alpha=0.4)
axes[1, 0].legend()

axes[1, 1].plot(time_h, np.cumsum(P_unbuf * np.gradient(t_eval)), 'r--', label='Unbuffered Energy')
axes[1, 1].plot(time_h, np.cumsum(P_buf * np.gradient(t_eval)), 'b-', label='Buffered Energy')
axes[1, 1].set_title("Cumulative Harvested Energy [J]")
axes[1, 1].set_xlabel("Time [h]")
axes[1, 1].grid(True, alpha=0.4)
axes[1, 1].legend()

plt.tight_layout()
st.pyplot(fig)
