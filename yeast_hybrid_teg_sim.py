import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class SystemParameters:
    def __init__(self):
        self.V_broth = 0.50
        self.m_broth = 0.50
        self.Cp_broth = 4184.0
        self.C_broth = self.m_broth * self.Cp_broth

        self.S_initial = 160.0
        self.X_initial = 4.0
        self.X_max = 24.0
        self.Ks = 2.5
        self.Y_xs = 0.16
        self.q_etoh = 0.48
        self.m_s = 0.015
        self.delta_H_glyc = 655.0

        self.T_min = 10.0
        self.T_opt = 31.0
        self.T_safe = 33.0
        self.T_lethal = 36.0
        self.mu_opt = 0.50
        self.k_inact = 2.0
        self.kd_mult = 1.5

        self.m_pcm = 0.090
        self.T_pcm_melt = 31.5
        self.delta_T_melt = 1.0
        self.L_f = 200000.0
        self.Cp_pcm = 2100.0

        self.R_insul = 85.0
        self.R_b_teg = 4.0
        self.R_th_teg = 2.4
        self.R_contact = 0.6
        self.R_b_pcm = 0.8
        self.R_sink_air = 2.0

        self.m_sink = 0.15
        self.Cp_sink = 900.0
        self.C_sink = self.m_sink * self.Cp_sink

        self.S_teg = 0.045
        self.R_int = 2.5

        self.T_amb = 23.0
        self.T_pitch = 28.0
        self.t_total_hours = 24.0

def cardinal_growth_rate(T, p):
    if T <= p.T_min or T >= p.T_lethal:
        return 0.0
    num = (T - p.T_lethal) * (T - p.T_min)**2
    den = (p.T_opt - p.T_min) * (
        (p.T_opt - p.T_min)*(T - p.T_opt) - (p.T_opt - p.T_lethal)*(p.T_opt + p.T_min - 2.0*T)
    )
    if den == 0.0:
        return 0.0
    return p.mu_opt * max(0.0, num / den)

def pcm_effective_capacitance(T_pcm, p):
    sigma = p.delta_T_melt / 2.0
    dbeta_dT = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((T_pcm - p.T_pcm_melt)/sigma)**2)
    return p.m_pcm * (p.Cp_pcm + p.L_f * dbeta_dT)

def odes_unbuffered(t, y, p):
    X, S, Tb, Tc, V_viab = y
    
    dV = -p.k_inact * (max(0.0, Tb - p.T_safe)**1.8) * V_viab / 3600.0
    active_V = max(0.0, (V_viab - 0.15) / 0.85) if V_viab > 0.15 else 0.0

    mu = cardinal_growth_rate(Tb, p) * (S / (p.Ks + S)) * active_V if S > 0.01 else 0.0
    kd = p.kd_mult * (1.0 - active_V) + 1.0 * max(0.0, Tb - p.T_safe)**2
    growth = mu * X * (1.0 - X/p.X_max)
    dX = (growth - kd * X) / 3600.0 if X > 0.001 else 0.0
    
    ferm_rate = p.q_etoh * active_V * X * (S / (p.Ks + S)) if S > 0.01 and X > 0.001 else 0.0
    maint_rate = p.m_s * active_V * X if X > 0.001 and S > 0.01 else 0.0
    dS = (-(1.0 / p.Y_xs) * max(0.0, growth) - ferm_rate - maint_rate) / 3600.0 if S > 0.01 and X > 0.001 else 0.0
    
    Q_bio = p.V_broth * p.delta_H_glyc * (-dS)
    Q_loss = (Tb - p.T_amb) / p.R_insul
    R_total_teg = p.R_b_teg + p.R_th_teg + p.R_contact
    Q_teg = (Tb - Tc) / R_total_teg
    Q_sink = (Tc - p.T_amb) / p.R_sink_air

    dTb = (Q_bio - Q_loss - Q_teg) / p.C_broth
    dTc = (Q_teg - Q_sink) / p.C_sink
    return [dX, dS, dTb, dTc, dV]

def odes_series_buffered(t, y, p):
    X, S, Tb, Tpcm, Tc, V_viab = y
    
    dV = -p.k_inact * (max(0.0, Tb - p.T_safe)**1.8) * V_viab / 3600.0
    active_V = max(0.0, (V_viab - 0.15) / 0.85) if V_viab > 0.15 else 0.0

    mu = cardinal_growth_rate(Tb, p) * (S / (p.Ks + S)) * active_V if S > 0.01 else 0.0
    kd = p.kd_mult * (1.0 - active_V) + 1.0 * max(0.0, Tb - p.T_safe)**2
    growth = mu * X * (1.0 - X/p.X_max)
    dX = (growth - kd * X) / 3600.0 if X > 0.001 else 0.0
    
    ferm_rate = p.q_etoh * active_V * X * (S / (p.Ks + S)) if S > 0.01 and X > 0.001 else 0.0
    maint_rate = p.m_s * active_V * X if X > 0.001 and S > 0.01 else 0.0
    dS = (-(1.0 / p.Y_xs) * max(0.0, growth) - ferm_rate - maint_rate) / 3600.0 if S > 0.01 and X > 0.001 else 0.0
    
    Q_bio = p.V_broth * p.delta_H_glyc * (-dS)
    Q_loss = (Tb - p.T_amb) / p.R_insul
    Q_to_pcm = (Tb - Tpcm) / p.R_b_pcm
    
    R_teg_out = p.R_th_teg + p.R_contact + 2.5
    Q_teg = (Tpcm - Tc) / R_teg_out
    Q_sink = (Tc - p.T_amb) / p.R_sink_air

    C_pcm_dyn = pcm_effective_capacitance(Tpcm, p)

    dTb = (Q_bio - Q_loss - Q_to_pcm) / p.C_broth
    dTpcm = (Q_to_pcm - Q_teg) / C_pcm_dyn
    dTc = (Q_teg - Q_sink) / p.C_sink
    return [dX, dS, dTb, dTpcm, dTc, dV]

def odes_parallel_hybrid(t, y, p):
    X, S, Tb, Tpcm, Tc, V_viab = y
    
    dV = -p.k_inact * (max(0.0, Tb - p.T_safe)**1.8) * V_viab / 3600.0
    active_V = max(0.0, (V_viab - 0.15) / 0.85) if V_viab > 0.15 else 0.0

    mu = cardinal_growth_rate(Tb, p) * (S / (p.Ks + S)) * active_V if S > 0.01 else 0.0
    kd = p.kd_mult * (1.0 - active_V) + 1.0 * max(0.0, Tb - p.T_safe)**2
    growth = mu * X * (1.0 - X/p.X_max)
    dX = (growth - kd * X) / 3600.0 if X > 0.001 else 0.0
    
    ferm_rate = p.q_etoh * active_V * X * (S / (p.Ks + S)) if S > 0.01 and X > 0.001 else 0.0
    maint_rate = p.m_s * active_V * X if X > 0.001 and S > 0.01 else 0.0
    dS = (-(1.0 / p.Y_xs) * max(0.0, growth) - ferm_rate - maint_rate) / 3600.0 if S > 0.01 and X > 0.001 else 0.0
    
    Q_bio = p.V_broth * p.delta_H_glyc * (-dS)
    Q_loss = (Tb - p.T_amb) / p.R_insul

    R_total_teg = p.R_b_teg + p.R_th_teg + p.R_contact
    Q_teg = (Tb - Tc) / R_total_teg

    Q_pcm = (Tb - Tpcm) / p.R_b_pcm
    Q_sink = (Tc - p.T_amb) / p.R_sink_air

    C_pcm_dyn = pcm_effective_capacitance(Tpcm, p)

    dTb = (Q_bio - Q_loss - Q_teg - Q_pcm) / p.C_broth
    dTpcm = Q_pcm / C_pcm_dyn
    dTc = (Q_teg - Q_sink) / p.C_sink
    return [dX, dS, dTb, dTpcm, dTc, dV]

def run_tri_system_simulation():
    p = SystemParameters()
    t_span = (0.0, p.t_total_hours * 3600.0)
    t_eval = np.linspace(0.0, p.t_total_hours * 3600.0, 1200)
    t_hours = t_eval / 3600.0

    sol_unbuf = solve_ivp(odes_unbuffered, t_span, [p.X_initial, p.S_initial, p.T_pitch, p.T_amb, 1.0], t_eval=t_eval, method='Radau', args=(p,))
    sol_series = solve_ivp(odes_series_buffered, t_span, [p.X_initial, p.S_initial, p.T_pitch, p.T_amb, p.T_amb, 1.0], t_eval=t_eval, method='Radau', args=(p,))
    sol_hybrid = solve_ivp(odes_parallel_hybrid, t_span, [p.X_initial, p.S_initial, p.T_pitch, p.T_amb, p.T_amb, 1.0], t_eval=t_eval, method='Radau', args=(p,))

    S_u, X_u = sol_unbuf.y[1], sol_unbuf.y[0]
    S_s, X_s = sol_series.y[1], sol_series.y[0]
    S_h, X_h = sol_hybrid.y[1], sol_hybrid.y[0]

    Tb_u, Tc_u = sol_unbuf.y[2], sol_unbuf.y[3]
    Tb_s, Tc_s = sol_series.y[2], sol_series.y[4]
    Tb_h, Tc_h = sol_hybrid.y[2], sol_hybrid.y[4]

    deltaT_u = np.maximum(0.0, Tb_u - Tc_u)
    deltaT_s = np.maximum(0.0, sol_series.y[3] - Tc_s)
    deltaT_h = np.maximum(0.0, Tb_h - Tc_h)

    P_u = ((p.S_teg * deltaT_u)**2) / (4.0 * p.R_int)
    P_s = ((p.S_teg * deltaT_s)**2) / (4.0 * p.R_int)
    P_h = ((p.S_teg * deltaT_h)**2) / (4.0 * p.R_int)

    dt = np.gradient(t_eval)
    E_u = np.cumsum(P_u * dt)
    E_s = np.cumsum(P_s * dt)
    E_h = np.cumsum(P_h * dt)

    dur_u = np.sum(deltaT_u >= 2.5) * (p.t_total_hours / len(t_eval))
    dur_s = np.sum(deltaT_s >= 2.5) * (p.t_total_hours / len(t_eval))
    dur_h = np.sum(deltaT_h >= 2.5) * (p.t_total_hours / len(t_eval))

    e_unbuf_final = E_u[-1]
    e_series_final = E_s[-1]
    e_hybrid_final = E_h[-1]
    gain_hybrid = ((e_hybrid_final - e_unbuf_final) / e_unbuf_final) * 100.0

    print(f"Unbuffered Peak Temp: {np.max(Tb_u):.2f} C | Final Energy: {e_unbuf_final:.2f} J")
    print(f"Series PCM Peak Temp: {np.max(Tb_s):.2f} C | Final Energy: {e_series_final:.2f} J")
    print(f"Hybrid Peak Temp:     {np.max(Tb_h):.2f} C | Final Energy: {e_hybrid_final:.2f} J (+{gain_hybrid:.1f}%)")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.patch.set_facecolor('#ffffff')

    ax = axes[0, 0]
    ax.plot(t_hours, Tb_u, 'r--', label=f'Unbuffered (Peak: {np.max(Tb_u):.1f}°C)', linewidth=2.0)
    ax.plot(t_hours, Tb_s, 'g:', label=f'Series PCM (Peak: {np.max(Tb_s):.1f}°C)', linewidth=2.0)
    ax.plot(t_hours, Tb_h, 'b-', label=f'Parallel Hybrid (Peak: {np.max(Tb_h):.1f}°C)', linewidth=2.5)
    ax.axhline(p.T_pcm_melt, color='purple', linestyle='--', alpha=0.7, label=f'PCM Melt Point ({p.T_pcm_melt}°C)')
    ax.axhline(p.T_safe, color='darkorange', linestyle='-.', alpha=0.8, label=f'Thermal Stress Limit ({p.T_safe}°C)')
    ax.axhline(p.T_lethal, color='darkred', linestyle=':', alpha=0.8, label=f'Lethal Limit ({p.T_lethal}°C)')
    ax.set_title('(a) Culture Temperature Regulation', fontsize=11, fontweight='bold')
    ax.set_xlabel('Fermentation Time (h)', fontsize=9.5)
    ax.set_ylabel('Broth Temperature (°C)', fontsize=9.5)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='lower left', fontsize=8)

    ax = axes[0, 1]
    ax.plot(t_hours, S_u, 'r--', label=f'Unbuffered (Stalled: {S_u[-1]:.1f} g/L)', linewidth=2.0)
    ax.plot(t_hours, S_s, 'g:', label=f'Series PCM ({S_s[-1]:.1f} g/L)', linewidth=2.0)
    ax.plot(t_hours, S_h, 'b-', label=f'Parallel Hybrid ({S_h[-1]:.1f} g/L)', linewidth=2.5)
    ax.set_title('(b) Substrate Consumption vs Time', fontsize=11, fontweight='bold')
    ax.set_xlabel('Fermentation Time (h)', fontsize=9.5)
    ax.set_ylabel('Glucose Concentration (g/L)', fontsize=9.5)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[0, 2]
    ax.plot(t_hours, X_u, 'r--', label='Unbuffered (Thermal Cell Death)', linewidth=2.0)
    ax.plot(t_hours, X_s, 'g:', label='Series PCM (Protected)', linewidth=2.0)
    ax.plot(t_hours, X_h, 'b-', label='Parallel Hybrid (Protected)', linewidth=2.5)
    ax.set_title('(c) Active Biomass Population', fontsize=11, fontweight='bold')
    ax.set_xlabel('Fermentation Time (h)', fontsize=9.5)
    ax.set_ylabel('Biomass Concentration (g/L)', fontsize=9.5)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='lower right', fontsize=8)

    ax = axes[1, 0]
    ax.plot(t_hours, deltaT_u, 'r--', label=f'Unbuffered ({dur_u:.1f}h)', linewidth=2.0)
    ax.plot(t_hours, deltaT_s, 'g:', label=f'Series PCM ({dur_s:.1f}h)', linewidth=2.0)
    ax.plot(t_hours, deltaT_h, 'b-', label=f'Parallel Hybrid ({dur_h:.1f}h)', linewidth=2.5)
    ax.axhline(2.5, color='orange', linestyle=':', label='Usable Harvester Threshold (ΔT ≥ 2.5°C)')
    ax.set_title('(d) Thermal Gradient Across TEG', fontsize=11, fontweight='bold')
    ax.set_xlabel('Fermentation Time (h)', fontsize=9.5)
    ax.set_ylabel('ΔT = Th - Tc (°C)', fontsize=9.5)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1, 1]
    ax.plot(t_hours, P_u * 1000.0, 'r--', label=f'Unbuffered (Peak: {np.max(P_u)*1000:.1f} mW)', linewidth=2.0)
    ax.plot(t_hours, P_s * 1000.0, 'g:', label=f'Series PCM (Peak: {np.max(P_s)*1000:.1f} mW)', linewidth=2.0)
    ax.plot(t_hours, P_h * 1000.0, 'b-', label=f'Parallel Hybrid (Peak: {np.max(P_h)*1000:.1f} mW)', linewidth=2.5)
    ax.set_title('(e) Instantaneous Electrical Power Output', fontsize=11, fontweight='bold')
    ax.set_xlabel('Fermentation Time (h)', fontsize=9.5)
    ax.set_ylabel('Electrical Power (mW)', fontsize=9.5)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1, 2]
    ax.plot(t_hours, E_u, 'r--', label=f'Unbuffered: {e_unbuf_final:.1f} J', linewidth=2.0)
    ax.plot(t_hours, E_s, 'g:', label=f'Series PCM: {e_series_final:.1f} J', linewidth=2.0)
    ax.plot(t_hours, E_h, 'b-', label=f'Parallel Hybrid: {e_hybrid_final:.1f} J (+{gain_hybrid:.0f}%)', linewidth=2.5)
    ax.set_title('(f) Cumulative Harvested Electrical Energy', fontsize=11, fontweight='bold')
    ax.set_xlabel('Fermentation Time (h)', fontsize=9.5)
    ax.set_ylabel('Cumulative Energy (J)', fontsize=9.5)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='lower right', fontsize=8.5)

    plt.tight_layout()
    plt.savefig("tri_system_comprehensive_comparison.png", dpi=300)

if __name__ == '__main__':
    run_tri_system_simulation()
