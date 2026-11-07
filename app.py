import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(page_title="Yeast PCM-TEG Harvester Simulation", layout="wide")

st.title("Yeast PCM-TEG Energy Harvester")
st.markdown("""
Simulation comparing unbuffered, series PCM, and parallel hybrid setups for 
harvesting thermoelectric energy from yeast fermentation while preventing stuck fermentation.
""")

st.sidebar.header("Fermentation Parameters")
S_initial = st.sidebar.slider("Initial Glucose Concentration [g/L]", 60.0, 220.0, 160.0, 5.0)
X_initial = st.sidebar.slider("Initial Yeast Biomass Pitch [g/L]", 1.0, 10.0, 4.0, 0.5)
V_broth = st.sidebar.slider("Broth Volume [L]", 0.25, 1.0, 0.50, 0.05)
T_pitch = st.sidebar.slider("Initial Pitching Temperature [deg C]", 22.0, 32.0, 28.0, 0.5)

st.sidebar.header("PCM Thermal Buffer")
m_pcm_g = st.sidebar.slider("PCM Buffer Mass [grams]", 20.0, 180.0, 90.0, 5.0)
T_pcm_melt = st.sidebar.slider("PCM Melting Temperature [deg C]", 28.0, 35.0, 31.5, 0.5)
L_f = st.sidebar.slider("Latent Heat of Fusion [kJ/kg]", 120.0, 250.0, 200.0, 10.0) * 1000.0

st.sidebar.header("Thermal Network & TEG")
R_insul = st.sidebar.slider("Insulation Thermal Resistance R_insul [K/W]", 30.0, 120.0, 85.0, 5.0)
R_th_teg = st.sidebar.slider("TEG Thermal Resistance R_teg [K/W]", 1.0, 4.0, 2.4, 0.2)
S_teg_mV = st.sidebar.slider("Seebeck Coefficient [mV/K]", 25.0, 60.0, 45.0, 1.0)
S_teg = S_teg_mV / 1000.0
T_amb = st.sidebar.slider("Ambient Temperature [deg C]", 18.0, 28.0, 23.0, 0.5)

m_broth = V_broth
C_broth = m_broth * 4184.0
m_pcm = m_pcm_g / 1000.0
Cp_pcm = 2100.0
delta_T_melt = 1.0

R_b_teg = 4.0
R_contact = 0.6
R_b_pcm = 0.8
R_sink_air = 2.0
m_sink = 0.15
C_sink = m_sink * 900.0
R_int = 2.5

T_min = 10.0
T_opt = 31.0
T_safe = 33.0
T_lethal = 36.0
mu_opt = 0.50
k_inact = 2.0
kd_mult = 1.5

X_max = 24.0
Ks = 2.5
Y_xs = 0.16
q_etoh = 0.48
m_s = 0.015
delta_H = 655.0

t_total_h = 24.0
t_span = (0.0, t_total_h * 3600.0)
t_eval = np.linspace(0.0, t_total_h * 3600.0, 800)
time_h = t_eval / 3600.0

def cardinal_growth_rate(T):
    if T <= T_min or T >= T_lethal:
        return 0.0
    num = (T - T_lethal) * (T - T_min)**2
    den = (T_opt - T_min) * ((T_opt - T_min)*(T - T_opt) - (T_opt - T_lethal)*(T_opt + T_min - 2.0*T))
    if den == 0.0:
        return 0.0
    return mu_opt * max(0.0, num / den)

def pcm_dyn_C(Tpcm):
    sigma = delta_T_melt / 2.0
    dbeta = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((Tpcm - T_pcm_melt)/sigma)**2)
    return m_pcm * (Cp_pcm + L_f * dbeta)

def odes_unbuffered(t, y):
    X, S, Tb, Tc, V = y
    dV = -k_inact * (max(0.0, Tb - T_safe)**1.8) * V / 3600.0
    active_V = max(0.0, (V - 0.15) / 0.85) if V > 0.15 else 0.0

    mu = cardinal_growth_rate(Tb) * (S / (Ks + S)) * active_V if S > 0.01 else 0.0
    kd = kd_mult * (1.0 - active_V) + 1.0 * max(0.0, Tb - T_safe)**2
    growth = mu * X * (1.0 - X/X_max)
    dX = (growth - kd * X) / 3600.0 if X > 0.001 else 0.0

    ferm = q_etoh * active_V * X * (S / (Ks + S)) if S > 0.01 and X > 0.001 else 0.0
    maint = m_s * active_V * X if X > 0.001 and S > 0.01 else 0.0
    dS = (-(1.0 / Y_xs) * max(0.0, growth) - ferm - maint) / 3600.0 if S > 0.01 and X > 0.001 else 0.0

    Q_bio = V_broth * delta_H * (-dS)
    Q_loss = (Tb - T_amb) / R_insul
    Q_teg = (Tb - Tc) / (R_b_teg + R_th_teg + R_contact)
    Q_sink = (Tc - T_amb) / R_sink_air

    dTb = (Q_bio - Q_loss - Q_teg) / C_broth
    dTc = (Q_teg - Q_sink) / C_sink
    return [dX, dS, dTb, dTc, dV]

def odes_hybrid(t, y):
    X, S, Tb, Tpcm, Tc, V = y
    dV = -k_inact * (max(0.0, Tb - T_safe)**1.8) * V / 3600.0
    active_V = max(0.0, (V - 0.15) / 0.85) if V > 0.15 else 0.0

    mu = cardinal_growth_rate(Tb) * (S / (Ks + S)) * active_V if S > 0.01 else 0.0
    kd = kd_mult * (1.0 - active_V) + 1.0 * max(0.0, Tb - T_safe)**2
    growth = mu * X * (1.0 - X/X_max)
    dX = (growth - kd * X) / 3600.0 if X > 0.001 else 0.0

    ferm = q_etoh * active_V * X * (S / (Ks + S)) if S > 0.01 and X > 0.001 else 0.0
    maint = m_s * active_V * X if X > 0.001 and S > 0.01 else 0.0
    dS = (-(1.0 / Y_xs) * max(0.0, growth) - ferm - maint) / 3600.0 if S > 0.01 and X > 0.001 else 0.0

    Q_bio = V_broth * delta_H * (-dS)
    Q_loss = (Tb - T_amb) / R_insul
    Q_teg = (Tb - Tc) / (R_b_teg + R_th_teg + R_contact)
    Q_pcm = (Tb - Tpcm) / R_b_pcm
    Q_sink = (Tc - T_amb) / R_sink_air

    C_pcm = pcm_dyn_C(Tpcm)
    dTb = (Q_bio - Q_loss - Q_teg - Q_pcm) / C_broth
    dTpcm = Q_pcm / C_pcm
    dTc = (Q_teg - Q_sink) / C_sink
    return [dX, dS, dTb, dTpcm, dTc, dV]

sol_unbuf = solve_ivp(odes_unbuffered, t_span, [X_initial, S_initial, T_pitch, T_amb, 1.0], t_eval=t_eval, method='Radau')
sol_hyb = solve_ivp(odes_hybrid, t_span, [X_initial, S_initial, T_pitch, T_amb, T_amb, 1.0], t_eval=t_eval, method='Radau')

dT_unbuf = np.maximum(0.0, sol_unbuf.y[2] - sol_unbuf.y[3])
dT_hyb = np.maximum(0.0, sol_hyb.y[2] - sol_hyb.y[4])

P_unbuf = ((S_teg * dT_unbuf)**2) / (4.0 * R_int)
P_hyb = ((S_teg * dT_hyb)**2) / (4.0 * R_int)

dt = np.gradient(t_eval)
E_unbuf = np.cumsum(P_unbuf * dt)[-1]
E_hyb = np.cumsum(P_hyb * dt)[-1]
gain = ((E_hyb - E_unbuf) / E_unbuf) * 100.0 if E_unbuf > 0.01 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Unbuffered Peak Broth Temp", f"{np.max(sol_unbuf.y[2]):.2f} °C")
col2.metric("Parallel Hybrid Peak Broth Temp", f"{np.max(sol_hyb.y[2]):.2f} °C")
col3.metric("Harvested Energy (Unbuffered)", f"{E_unbuf:.2f} J")
col4.metric("Harvested Energy (Hybrid)", f"{E_hyb:.2f} J", f"{gain:+.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(time_h, sol_unbuf.y[2], 'r--', label='Unbuffered')
axes[0, 0].plot(time_h, sol_hyb.y[2], 'b-', label='Parallel Hybrid')
axes[0, 0].axhline(T_pcm_melt, color='purple', linestyle=':', label=f'PCM Melt ({T_pcm_melt}°C)')
axes[0, 0].axhline(33.0, color='darkorange', linestyle='-.', label='Thermal Stress Limit (33°C)')
axes[0, 0].set_title("Broth Culture Temperature [°C]")
axes[0, 0].set_xlabel("Time [h]")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(time_h, sol_unbuf.y[1], 'r--', label=f'Unbuffered ({sol_unbuf.y[1][-1]:.1f} g/L left)')
axes[0, 1].plot(time_h, sol_hyb.y[1], 'b-', label=f'Hybrid ({sol_hyb.y[1][-1]:.1f} g/L left)')
axes[0, 1].set_title("Substrate (Glucose) Concentration [g/L]")
axes[0, 1].set_xlabel("Time [h]")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

axes[1, 0].plot(time_h, P_unbuf * 1000, 'r--', label='Unbuffered')
axes[1, 0].plot(time_h, P_hyb * 1000, 'b-', label='Parallel Hybrid')
axes[1, 0].set_title("Instantaneous Power Output [mW]")
axes[1, 0].set_xlabel("Time [h]")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

axes[1, 1].plot(time_h, np.cumsum(P_unbuf * dt), 'r--', label=f'Unbuffered: {E_unbuf:.1f} J')
axes[1, 1].plot(time_h, np.cumsum(P_hyb * dt), 'b-', label=f'Hybrid: {E_hyb:.1f} J ({gain:+.1f}%)')
axes[1, 1].set_title("Cumulative Harvested Electrical Energy [J]")
axes[1, 1].set_xlabel("Time [h]")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
st.pyplot(fig)
