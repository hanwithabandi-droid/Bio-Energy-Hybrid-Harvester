"""
================================================================================
MULTI-PHYSICS SIMULATION FRAMEWORK: YEAST-PCM-TEG BIO-THERMAL HARVESTER
Title: Multi-Physics Modeling and Exergy Optimization of Phase-Change Material
       Buffer Systems for Biological Thermoelectric Energy Harvesting
================================================================================
This script implements the complete coupled differential equation system:
1. Microbial Kinetics: Temperature-dependent Monod growth & substrate-heat coupling (von Stockar biocalorimetry)
2. Latent Heat Buffering: Apparent heat capacity formulation for Phase Change Material (PCM)
3. Lumped Thermal Circuit: Multi-node dynamic heat transfer network
4. Thermoelectric Transduction: Seebeck generation, internal resistance, and dynamic matched power
5. Exergy Analysis: Second-Law thermodynamic efficiency and comparative benchmarking against unbuffered control
================================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PARAMETERS & SYSTEM SPECIFICATIONS
# ==============================================================================

class SimulationParameters:
    def __init__(self):
        # --- Biological / Bioreactor Parameters ---
        self.V_broth = 0.50             # Broth volume [L] (500 mL)
        self.m_broth = 0.50             # Mass of broth [kg] (approx. water density)
        self.Cp_broth = 4184.0          # Specific heat of broth [J/(kg*K)]
        self.C_broth = self.m_broth * self.Cp_broth  # Thermal mass [J/K] (~2092 J/K)
        
        self.S_initial = 100.0          # Initial glucose concentration [g/L]
        self.X_initial = 4.0            # Initial yeast biomass pitch [g/L]
        self.X_max = 22.0               # Maximum biomass carrying capacity [g/L]
        self.Ks = 2.0                   # Monod half-saturation constant [g/L]
        self.Y_xs = 0.16                # Biomass yield from substrate [g dry mass / g glucose]
        self.m_s = 0.015                # Maintenance substrate coefficient [g/(g*h)]
        
        # Enthalpy of glucose glycolysis to ethanol (von Stockar, -118 kJ/mol)
        # 118 kJ / 180.16 g/mol = 655 J / g glucose consumed
        self.delta_H_glycolysis = 655.0 # [J/g glucose]
        
        # Cardinal Temperature Model Parameters (Rosso / Ratkowsky model for S. cerevisiae)
        self.T_min = 10.0               # Minimum growth temperature [deg C]
        self.T_opt = 32.0               # Optimum growth temperature [deg C]
        self.T_max = 38.5               # Maximum thermal inhibition limit [deg C]
        self.mu_opt = 0.48              # Maximum specific growth rate at T_opt [1/h]

        # --- Phase Change Material (PCM) Parameters ---
        # Organic Bio-PCM / Eicosane-Octadecane Blend (e.g., PureTemp 27 or RT-28HC)
        self.m_pcm = 0.06               # Mass of PCM buffer [kg] (60 grams)
        self.T_pcm_melt = 28.0          # Phase change central melting temperature [deg C]
        self.delta_T_melt = 1.0         # Half-width of phase transition range [deg C]
        self.L_f = 180000.0             # Latent heat of fusion [J/kg] (180 kJ/kg)
        self.Cp_pcm_sensible = 2000.0   # Sensible specific heat capacity [J/(kg*K)]
        
        # --- Thermal Resistance Network Parameters ---
        self.R_insul = 35.0             # High-grade PU / Vacuum flask wall insulation [K/W]
        self.R_b_pcm = 2.0              # Reactor wall to PCM interface thermal resistance [K/W]
        self.R_th_teg = 2.2             # Internal thermal resistance of TEG module [K/W]
        self.R_contact = 0.5            # Thermal contact paste resistance [K/W]
        self.R_sink_air = 2.0           # Aluminum finned heatsink to air convection resistance [K/W]
        
        self.m_sink = 0.15              # Mass of aluminum heatsink [kg]
        self.Cp_sink = 900.0            # Specific heat of aluminum [J/(kg*K)]
        self.C_sink = self.m_sink * self.Cp_sink # Thermal mass of heatsink [J/K]

        # --- Thermoelectric Generator (TEG) Electrical Parameters ---
        self.S_teg = 0.045              # Seebeck coefficient of module [V/K]
        self.R_int = 2.5                # Internal electrical resistance [Ohms]
        
        # --- Environmental Boundary Conditions ---
        self.T_amb = 22.0               # Ambient room temperature [deg C]
        self.T_pitch = 26.0             # Initial pitching broth temperature [deg C]
        self.t_total_hours = 20.0       # Total simulation duration [hours]


# ==============================================================================
# 2. CORE MULTI-PHYSICS CONSTITUTIVE EQUATIONS
# ==============================================================================

def cardinal_growth_rate(T, p):
    """
    Computes temperature-dependent specific growth rate mu(T) [1/h]
    using the Cardinal Temperature Model with Inflection (Rosso et al.).
    """
    if T <= p.T_min or T >= p.T_max:
        return 0.0
    
    num = (T - p.T_max) * (T - p.T_min)**2
    den = (p.T_opt - p.T_min) * (
        (p.T_opt - p.T_min)*(T - p.T_opt) - (p.T_opt - p.T_max)*(p.T_opt + p.T_min - 2.0*T)
    )
    if den == 0.0:
        return 0.0
    
    phi = num / den
    mu = p.mu_opt * max(0.0, phi)
    return mu

def pcm_effective_heat_capacity(T_pcm, p):
    """
    Apparent Heat Capacity Method for Phase Change Material:
    Includes sensible heat plus Gaussian-smoothed latent heat absorption.
    """
    # Gaussian distribution of latent heat absorption around T_pcm_melt
    sigma = p.delta_T_melt / 2.0
    dbeta_dT = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((T_pcm - p.T_pcm_melt) / sigma)**2)
    Cp_effective = p.Cp_pcm_sensible + p.L_f * dbeta_dT
    return p.m_pcm * Cp_effective  # Total dynamic thermal mass [J/K]


# ==============================================================================
# 3. COUPLED ODE SYSTEMS (PCM-BUFFERED VS. UNBUFFERED CONTROL)
# ==============================================================================

def odes_buffered(t_sec, y, p):
    """
    State vector y:
    y[0] = X       (Biomass concentration, g/L)
    y[1] = S       (Substrate concentration, g/L)
    y[2] = T_b     (Broth temperature, °C)
    y[3] = T_pcm   (PCM buffer temperature, °C)
    y[4] = T_c     (TEG cold side / heatsink temperature, °C)
    """
    X, S, T_b, T_pcm, T_c = y
    
    # 1. Biological Kinetics
    if S > 0.001 and X < p.X_max:
        mu_T = cardinal_growth_rate(T_b, p)
        mu_eff = mu_T * (S / (p.Ks + S))
        dX_dt_h = mu_eff * X * (1.0 - (X / p.X_max))
        dS_dt_h = -(1.0 / p.Y_xs) * dX_dt_h - p.m_s * X
    else:
        dX_dt_h = 0.0
        dS_dt_h = 0.0
        
    dX_dt = dX_dt_h / 3600.0  # Convert to per second
    dS_dt = dS_dt_h / 3600.0  # Convert to per second
    
    # Biological metabolic heat generation [Watts]
    Q_bio = p.V_broth * p.delta_H_glycolysis * (-dS_dt)
    
    # 2. Thermal Energy Balance
    # Heat loss to ambient through vessel insulation
    Q_loss = (T_b - p.T_amb) / p.R_insul
    
    # Heat transfer from broth to PCM buffer
    Q_to_pcm = (T_b - T_pcm) / p.R_b_pcm
    
    # Heat transfer from PCM through TEG to heat sink
    R_total_teg = p.R_th_teg + p.R_contact
    Q_through_teg = (T_pcm - T_c) / R_total_teg
    
    # Heat rejection from heatsink to ambient
    Q_sink_out = (T_c - p.T_amb) / p.R_sink_air
    
    # 3. Dynamic Temperature Rates
    dTb_dt = (Q_bio - Q_loss - Q_to_pcm) / p.C_broth
    
    C_pcm_dyn = pcm_effective_heat_capacity(T_pcm, p)
    dTpcm_dt = (Q_to_pcm - Q_through_teg) / C_pcm_dyn
    
    dTc_dt = (Q_through_teg - Q_sink_out) / p.C_sink
    
    return [dX_dt, dS_dt, dTb_dt, dTpcm_dt, dTc_dt]


def odes_unbuffered(t_sec, y, p):
    """
    Unbuffered direct-coupled baseline control system.
    State vector y:
    y[0] = X       (Biomass concentration, g/L)
    y[1] = S       (Substrate concentration, g/L)
    y[2] = T_b     (Broth temperature, °C)
    y[3] = T_c     (TEG cold side / heatsink temperature, °C)
    """
    X, S, T_b, T_c = y
    
    # Biological Kinetics
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
    
    # Direct heat conduction through TEG from broth
    R_total_direct = p.R_b_pcm + p.R_th_teg + p.R_contact
    Q_through_teg = (T_b - T_c) / R_total_direct
    
    Q_sink_out = (T_c - p.T_amb) / p.R_sink_air
    
    dTb_dt = (Q_bio - Q_loss - Q_through_teg) / p.C_broth
    dTc_dt = (Q_through_teg - Q_sink_out) / p.C_sink
    
    return [dX_dt, dS_dt, dTb_dt, dTc_dt]


# ==============================================================================
# 4. SIMULATION EXECUTION & DATA POST-PROCESSING
# ==============================================================================

def run_comparative_simulation():
    p = SimulationParameters()
    t_span = (0.0, p.t_total_hours * 3600.0)
    t_eval = np.linspace(0.0, p.t_total_hours * 3600.0, 1500)
    
    # Initial conditions: broth starts at warm pitching temperature (26.0 deg C), other nodes at ambient
    y0_buf = [p.X_initial, p.S_initial, p.T_pitch, p.T_amb, p.T_amb]
    y0_unbuf = [p.X_initial, p.S_initial, p.T_pitch, p.T_amb]
    
    print(">>> Solving PCM-Buffered System ODEs...")
    sol_buf = solve_ivp(odes_buffered, t_span, y0_buf, t_eval=t_eval, method='Radau', args=(p,))
    
    print(">>> Solving Unbuffered Baseline System ODEs...")
    sol_unbuf = solve_ivp(odes_unbuffered, t_span, y0_unbuf, t_eval=t_eval, method='Radau', args=(p,))
    
    t_hours = sol_buf.t / 3600.0
    
    # --- Buffered Transduction & Electrical Metrics ---
    Tb_buf   = sol_buf.y[2]
    Tpcm_buf = sol_buf.y[3]
    Tc_buf   = sol_buf.y[4]
    delta_T_buf = np.maximum(0.0, Tpcm_buf - Tc_buf)
    
    Voc_buf = p.S_teg * delta_T_buf                          # Open circuit voltage [V]
    P_elec_buf = (Voc_buf**2) / (4.0 * p.R_int)              # Matched electrical power [Watts]
    E_elec_buf = np.cumsum(P_elec_buf * np.gradient(sol_buf.t)) # Cumulative energy [Joules]
    
    # Exergy rate entering TEG: Q_in * (1 - T_amb / T_h)
    Q_in_teg_buf = delta_T_buf / (p.R_th_teg + p.R_contact)
    Carnot_factor_buf = 1.0 - (p.T_amb + 273.15) / np.maximum(1.0, (Tpcm_buf + 273.15))
    Exergy_rate_buf = Q_in_teg_buf * np.maximum(0.0, Carnot_factor_buf)
    E_exergy_cum_buf = np.cumsum(Exergy_rate_buf * np.gradient(sol_buf.t))
    
    # --- Unbuffered Transduction & Electrical Metrics ---
    Tb_unbuf = sol_unbuf.y[2]
    Tc_unbuf = sol_unbuf.y[3]
    delta_T_unbuf = np.maximum(0.0, Tb_unbuf - Tc_unbuf)
    
    Voc_unbuf = p.S_teg * delta_T_unbuf
    P_elec_unbuf = (Voc_unbuf**2) / (4.0 * p.R_int)
    E_elec_unbuf = np.cumsum(P_elec_unbuf * np.gradient(sol_unbuf.t))
    
    Q_in_teg_unbuf = delta_T_unbuf / (p.R_b_pcm + p.R_th_teg + p.R_contact)
    Carnot_factor_unbuf = 1.0 - (p.T_amb + 273.15) / np.maximum(1.0, (Tb_unbuf + 273.15))
    Exergy_rate_unbuf = Q_in_teg_unbuf * np.maximum(0.0, Carnot_factor_unbuf)
    E_exergy_cum_unbuf = np.cumsum(Exergy_rate_unbuf * np.gradient(sol_unbuf.t))
    
    # --- Summary Metrics ---
    peak_Tb_unbuf = np.max(Tb_unbuf)
    peak_Tb_buf   = np.max(Tb_buf)
    temp_suppression = peak_Tb_unbuf - peak_Tb_buf
    
    final_E_buf   = E_elec_buf[-1]
    final_E_unbuf = E_elec_unbuf[-1]
    energy_gain_pct = ((final_E_buf - final_E_unbuf) / max(1e-9, final_E_unbuf)) * 100.0
    
    # Operational duration where delta_T >= 3.0 deg C
    duration_buf = np.sum(delta_T_buf >= 3.0) * (p.t_total_hours / len(t_eval))
    duration_unbuf = np.sum(delta_T_unbuf >= 3.0) * (p.t_total_hours / len(t_eval))
    
    eta_exergy_buf = (final_E_buf / max(1e-6, E_exergy_cum_buf[-1])) * 100.0
    eta_exergy_unbuf = (final_E_unbuf / max(1e-6, E_exergy_cum_unbuf[-1])) * 100.0

    print("\n" + "="*70)
    print("           COMPUTATIONAL MULTI-PHYSICS BENCHMARK RESULTS")
    print("="*70)
    print(f"Unbuffered Peak Culture Temperature : {peak_Tb_unbuf:.2f} deg C")
    print(f"PCM-Buffered Peak Culture Temp      : {peak_Tb_buf:.2f} deg C (Suppression: -{temp_suppression:.2f} deg C)")
    print(f"Effective TEG Operation (Delta-T >= 3 deg C) : Unbuffered: {duration_unbuf:.2f} h | Buffered: {duration_buf:.2f} h")
    print(f"Total Harvested Electrical Energy   : Unbuffered: {final_E_unbuf:.3f} J | Buffered: {final_E_buf:.3f} J")
    print(f"Relative Energy Output Gain         : +{energy_gain_pct:.1f} %")
    print(f"Second-Law Exergy Efficiency        : Unbuffered: {eta_exergy_unbuf:.2f} % | Buffered: {eta_exergy_buf:.2f} %")
    print("="*70 + "\n")
    
    # ==============================================================================
    # 5. SCIENTIFIC VISUALIZATION & FIGURE GENERATION
    # ==============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#ffffff')

    # Subplot (a): Core Temperature Profiles
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

    # Subplot (b): Temperature Gradient Across TEG (Delta-T)
    ax = axes[0, 1]
    ax.plot(t_hours, delta_T_unbuf, 'r--', label=r'$\Delta T$ (Unbuffered)', linewidth=2.0)
    ax.plot(t_hours, delta_T_buf, 'b-', label=r'$\Delta T$ (PCM-Buffered)', linewidth=2.2)
    ax.axhline(3.0, color='orange', linestyle=':', label=r'Usable Threshold ($\Delta T \geq 3.0^\circ$C)')
    ax.set_title('(b) Dynamic Thermal Gradient Across TEG', fontsize=12, fontweight='bold')
    ax.set_xlabel('Fermentation Time [hours]', fontsize=11)
    ax.set_ylabel(r'$\Delta T = T_h - T_c$ [°C]', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)

    # Subplot (c): Instantaneous Thermoelectric Power
    ax = axes[1, 0]
    ax.plot(t_hours, P_elec_unbuf * 1e3, 'r--', label='Power (Unbuffered)', linewidth=2.0)
    ax.plot(t_hours, P_elec_buf * 1e3, 'b-', label='Power (PCM-Buffered)', linewidth=2.2)
    ax.set_title('(c) Instantaneous Matched Electrical Power', fontsize=12, fontweight='bold')
    ax.set_xlabel('Fermentation Time [hours]', fontsize=11)
    ax.set_ylabel('Harvested Power [mW]', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)

    # Subplot (d): Cumulative Harvested Electrical Energy
    ax = axes[1, 1]
    ax.plot(t_hours, E_elec_unbuf, 'r--', label=f'Unbuffered: {final_E_unbuf:.2f} J', linewidth=2.0)
    ax.plot(t_hours, E_elec_buf, 'b-', label=f'PCM-Buffered: {final_E_buf:.2f} J (+{energy_gain_pct:.0f}%)', linewidth=2.2)
    ax.set_title('(d) Cumulative Harvested Electrical Energy [J]', fontsize=12, fontweight='bold')
    ax.set_xlabel('Fermentation Time [hours]', fontsize=11)
    ax.set_ylabel('Cumulative Energy [Joules]', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plot_filename = "yeast_pcm_teg_simulation_results.png"
    plt.savefig(plot_filename, dpi=300)
    print(f">>> High-resolution multi-physics figure exported: {plot_filename}")
    print(">>> Opening interactive plot window on your screen...")
    plt.show()

if __name__ == '__main__':
    run_comparative_simulation()

