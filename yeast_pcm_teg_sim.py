import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class SimulationParameters:
    def __init__(self):
        self.V_broth = 0.50
        self.m_broth = 0.50
        self.Cp_broth = 4184.0
        self.C_broth = self.m_broth * self.Cp_broth
        
        self.S_initial = 100.0
        self.X_initial = 4.0
        self.X_max = 22.0
        self.Ks = 2.0
        self.Y_xs = 0.16
        self.m_s = 0.015
        self.delta_H_glycolysis = 655.0
        
        self.T_min = 10.0
        self.T_opt = 32.0
        self.T_max = 38.5
        self.mu_opt = 0.48

        self.m_pcm = 0.06
        self.T_pcm_melt = 28.0
        self.delta_T_melt = 1.0
        self.L_f = 180000.0
        self.Cp_pcm_sensible = 2000.0
        
        self.R_insul = 35.0
        self.R_b_pcm = 2.0
        self.R_th_teg = 2.2
        self.R_contact = 0.5
        self.R_sink_air = 2.0
        
        self.m_sink = 0.15
        self.Cp_sink = 900.0
        self.C_sink = self.m_sink * self.Cp_sink

        self.S_teg = 0.045
        self.R_int = 2.5
        
        self.T_amb = 22.0
        self.T_pitch = 26.0
        self.t_total_hours = 20.0

def cardinal_growth_rate(T, p):
    if T <= p.T_min or T >= p.T_max:
        return 0.0
    num = (T - p.T_max) * (T - p.T_min)**2
    den = (p.T_opt - p.T_min) * (
        (p.T_opt - p.T_min)*(T - p.T_opt) - (p.T_opt - p.T_max)*(p.T_opt + p.T_min - 2.0*T)
    )
    if den == 0.0:
        return 0.0
    return p.mu_opt * max(0.0, num / den)

def pcm_effective_heat_capacity(T_pcm, p):
    sigma = p.delta_T_melt / 2.0
    dbeta_dT = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((T_pcm - p.T_pcm_melt) / sigma)**2)
    return p.m_pcm * (p.Cp_pcm_sensible + p.L_f * dbeta_dT)

def odes_buffered(t_sec, y, p):
    X, S, T_b, T_pcm, T_c = y
    
    if S > 0.001 and X < p.X_max:
        mu_T = cardinal_growth_rate(T_b, p)
        mu_eff = mu_T * (S / (p.Ks + S))
        dX_dt_h = mu_eff * X * (1.0 - (X / p.X_max))
        dS_dt_h = -(1.0 / p.Y_xs) * dX_dt_h - p.m_s * X
    else:
        dX_dt_h = 0.0
        dS_dt_h = 0.0
        
    dX_dt = dX_dt_h / 3600.0
    dS_dt = dS_dt_h / 3600.0
    
    Q_bio = p.V_broth * p.delta_H_glycolysis * (-dS_dt)
    Q_loss = (T_b - p.T_amb) / p.R_insul
    Q_to_pcm = (T_b - T_pcm) / p.R_b_pcm
    R_total_teg = p.R_th_teg + p.R_contact
    Q_through_teg = (T_pcm - T_c) / R_total_teg
    Q_sink_out = (T_c - p.T_amb) / p.R_sink_air
    
    dTb_dt = (Q_bio - Q_loss - Q_to_pcm) / p.C_broth
    C_pcm_dyn = pcm_effective_heat_capacity(T_pcm, p)
    dTpcm_dt = (Q_to_pcm - Q_through_teg) / C_pcm_dyn
    dTc_dt = (Q_through_teg - Q_sink_out) / p.C_sink
    
    return [dX_dt, dS_dt, dTb_dt, dTpcm_dt, dTc_dt]

def odes_unbuffered(t_sec, y, p):
    X, S, T_b, T_c = y
    
    if S > 0.001 and X < p.X_max:
        mu_T = cardinal_growth_rate(T_b, p)
        mu_eff = mu_T * (S / (p.Ks + S))
        dX_dt_h = mu_eff * X * (1.0 - (X / p.X_max))
        dS_dt_h = -(1.0 / p.Y_xs) * dX_dt_h - p.m_s * X
    else:
        dX_dt_h = 0.0
        dS_dt_h = 0.0
        
    dX_dt = dX_dt_h / 3600.0
    dS_dt = dS_dt_h / 3600.0
    
    Q_bio = p.V_broth * p.delta_H_glycolysis * (-dS_dt)
    Q_loss = (T_b - p.T_amb) / p.R_insul
    R_total_direct = p.R_b_pcm + p.R_th_teg + p.R_contact
    Q_through_teg = (T_b - T_c) / R_total_direct
    Q_sink_out = (T_c - p.T_amb) / p.R_sink_air
    
    dTb_dt = (Q_bio - Q_loss - Q_through_teg) / p.C_broth
    dTc_dt = (Q_through_teg - Q_sink_out) / p.C_sink
    
    return [dX_dt, dS_dt, dTb_dt, dTc_dt]

def run_comparative_simulation():
    p = SimulationParameters()
    t_span = (0.0, p.t_total_hours * 3600.0)
    t_eval = np.linspace(0.0, p.t_total_hours * 3600.0, 1500)
    
    y0_buf = [p.X_initial, p.S_initial, p.T_pitch, p.T_amb, p.T_amb]
    y0_unbuf = [p.X_initial, p.S_initial, p.T_pitch, p.T_amb]
    
    sol_buf = solve_ivp(odes_buffered, t_span, y0_buf, t_eval=t_eval, method='Radau', args=(p,))
    sol_unbuf = solve_ivp(odes_unbuffered, t_span, y0_unbuf, t_eval=t_eval, method='Radau', args=(p,))
    
    t_hours = sol_buf.t / 3600.0
    
    Tb_buf = sol_buf.y[2]
    Tpcm_buf = sol_buf.y[3]
    Tc_buf = sol_buf.y[4]
    delta_T_buf = np.maximum(0.0, Tpcm_buf - Tc_buf)
    
    Voc_buf = p.S_teg * delta_T_buf
    P_elec_buf = (Voc_buf**2) / (4.0 * p.R_int)
    E_elec_buf = np.cumsum(P_elec_buf * np.gradient(sol_buf.t))
    
    Tb_unbuf = sol_unbuf.y[2]
    Tc_unbuf = sol_unbuf.y[3]
    delta_T_unbuf = np.maximum(0.0, Tb_unbuf - Tc_unbuf)
    
    Voc_unbuf = p.S_teg * delta_T_unbuf
    P_elec_unbuf = (Voc_unbuf**2) / (4.0 * p.R_int)
    E_elec_unbuf = np.cumsum(P_elec_unbuf * np.gradient(sol_unbuf.t))
    
    final_E_buf = E_elec_buf[-1]
    final_E_unbuf = E_elec_unbuf[-1]
    energy_gain_pct = ((final_E_buf - final_E_unbuf) / max(1e-9, final_E_unbuf)) * 100.0
    
    duration_buf = np.sum(delta_T_buf >= 3.0) * (p.t_total_hours / len(t_eval))
    duration_unbuf = np.sum(delta_T_unbuf >= 3.0) * (p.t_total_hours / len(t_eval))

    print(f"Unbuffered Peak Temp: {np.max(Tb_unbuf):.2f} C | Final Energy: {final_E_unbuf:.3f} J | Duration: {duration_unbuf:.2f} h")
    print(f"PCM-Buffered Peak Temp: {np.max(Tb_buf):.2f} C | Final Energy: {final_E_buf:.3f} J (+{energy_gain_pct:.1f}%) | Duration: {duration_buf:.2f} h")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#ffffff')

    ax = axes[0, 0]
    ax.plot(t_hours, Tb_unbuf, 'r--', label='Broth $T_b$ (Unbuffered Control)', linewidth=2.0)
    ax.plot(t_hours, Tb_buf, 'b-', label='Broth $T_b$ (PCM-Buffered)', linewidth=2.2)
    ax.plot(t_hours, Tpcm_buf, 'g-.', label='PCM Buffer $T_{pcm}$', linewidth=1.8)
    ax.axhline(p.T_pcm_melt, color='gray', linestyle=':', label='PCM Melt Point (31.5°C)')
    ax.axhline(p.T_max, color='darkred', linestyle=':', label='Yeast Inhibition Limit (38.5°C)')
    ax.set_title('(a) Thermal Pulse Shaping & Temperature Regulation', fontsize=12, fontweight='bold')
    ax.set_xlabel('Fermentation Time [hours]', fontsize=11)
    ax.set_ylabel('Temperature [°C]', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)

    ax = axes[0, 1]
    ax.plot(t_hours, delta_T_unbuf, 'r--', label=r'$\Delta T$ (Unbuffered)', linewidth=2.0)
    ax.plot(t_hours, delta_T_buf, 'b-', label=r'$\Delta T$ (PCM-Buffered)', linewidth=2.2)
    ax.axhline(3.0, color='orange', linestyle=':', label=r'Usable Threshold ($\Delta T \geq 3.0^\circ$C)')
    ax.set_title('(b) Dynamic Thermal Gradient Across TEG', fontsize=12, fontweight='bold')
    ax.set_xlabel('Fermentation Time [hours]', fontsize=11)
    ax.set_ylabel(r'$\Delta T = T_h - T_c$ [°C]', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)

    ax = axes[1, 0]
    ax.plot(t_hours, P_elec_unbuf * 1e3, 'r--', label='Power (Unbuffered)', linewidth=2.0)
    ax.plot(t_hours, P_elec_buf * 1e3, 'b-', label='Power (PCM-Buffered)', linewidth=2.2)
    ax.set_title('(c) Instantaneous Matched Electrical Power', fontsize=12, fontweight='bold')
    ax.set_xlabel('Fermentation Time [hours]', fontsize=11)
    ax.set_ylabel('Harvested Power [mW]', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)

    ax = axes[1, 1]
    ax.plot(t_hours, E_elec_unbuf, 'r--', label=f'Unbuffered: {final_E_unbuf:.2f} J', linewidth=2.0)
    ax.plot(t_hours, E_elec_buf, 'b-', label=f'PCM-Buffered: {final_E_buf:.2f} J (+{energy_gain_pct:.0f}%)', linewidth=2.2)
    ax.set_title('(d) Cumulative Harvested Electrical Energy [J]', fontsize=12, fontweight='bold')
    ax.set_xlabel('Fermentation Time [hours]', fontsize=11)
    ax.set_ylabel('Cumulative Energy [Joules]', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig("yeast_pcm_teg_simulation_results.png", dpi=300)

if __name__ == '__main__':
    run_comparative_simulation()
