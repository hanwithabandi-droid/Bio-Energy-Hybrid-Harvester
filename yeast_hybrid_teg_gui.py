import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

class BioTegSimulationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bio-TEG Harvester Simulation")
        self.geometry("1580x930")
        self.configure(bg="#111827")

        self.anim_running = False
        self.anim_index = 0
        self.anim_step_delay = 30

        self.params = {
            "S_initial": 160.0,
            "X_initial": 4.0,
            "T_pcm_melt": 31.5,
            "m_pcm_g": 90.0,
            "R_insul": 85.0,
            "T_amb": 23.0,
            "T_pitch": 28.0,
            "S_teg": 0.045,
            "R_int": 2.5
        }

        self._setup_ui()
        self.run_simulation_data()
        self.draw_plots()

    def _setup_ui(self):
        self.sidebar = tk.Frame(self, width=400, bg="#1f2937", padx=16, pady=12)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.main_area = tk.Frame(self, bg="#ffffff")
        self.main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        title_lbl = tk.Label(self.sidebar, text="Bio-TEG Harvester", font=("Helvetica", 14, "bold"), fg="#f9fafb", bg="#1f2937")
        title_lbl.pack(anchor="w", pady=(0, 2))
        sub_lbl = tk.Label(self.sidebar, text="Yeast-PCM-TEG Model Comparison", font=("Helvetica", 9), fg="#9ca3af", bg="#1f2937")
        sub_lbl.pack(anchor="w", pady=(0, 8))

        self.sliders = {}
        def add_slider(key, label_text, min_v, max_v, default_v, step_v, unit_text):
            frame = tk.Frame(self.sidebar, bg="#1f2937")
            frame.pack(fill=tk.X, pady=2.5)
            
            lbl_row = tk.Frame(frame, bg="#1f2937")
            lbl_row.pack(fill=tk.X)
            lbl = tk.Label(lbl_row, text=label_text, font=("Helvetica", 9, "bold"), fg="#f3f4f6", bg="#1f2937")
            lbl.pack(side=tk.LEFT)
            val_lbl = tk.Label(lbl_row, text=f"{default_v} {unit_text}", font=("Helvetica", 9), fg="#10b981", bg="#1f2937")
            val_lbl.pack(side=tk.RIGHT)
            
            def on_slide(val):
                v = round(float(val), 1)
                val_lbl.config(text=f"{v} {unit_text}")
                self.params[key] = v
                if not self.anim_running:
                    self.update_simulation_live()

            slider = ttk.Scale(frame, from_=min_v, to=max_v, orient=tk.HORIZONTAL, value=default_v, command=on_slide)
            slider.pack(fill=tk.X, pady=(1, 0))
            self.sliders[key] = (slider, val_lbl, unit_text)

        add_slider("S_initial", "Glucose Concentration", 60.0, 220.0, 160.0, 5.0, "g/L")
        add_slider("X_initial", "Initial Yeast Pitch", 1.0, 10.0, 4.0, 0.5, "g/L")
        add_slider("T_pcm_melt", "PCM Melting Point", 28.0, 35.0, 31.5, 0.5, "°C")
        add_slider("m_pcm_g", "PCM Buffer Mass", 20.0, 180.0, 90.0, 5.0, "g")
        add_slider("R_insul", "Vessel Insulation (R_th)", 30.0, 120.0, 85.0, 5.0, "K/W")
        add_slider("T_amb", "Ambient Room Temp", 18.0, 28.0, 23.0, 0.5, "°C")

        btn_frame = tk.Frame(self.sidebar, bg="#1f2937")
        btn_frame.pack(fill=tk.X, pady=(8, 4))

        self.btn_run = tk.Button(btn_frame, text="Run Simulation", font=("Helvetica", 10, "bold"), bg="#3b82f6", fg="white", activebackground="#2563eb", relief=tk.FLAT, pady=5, command=self.update_simulation_live)
        self.btn_run.pack(fill=tk.X, pady=2)

        self.btn_animate = tk.Button(btn_frame, text="Play 24h Animation", font=("Helvetica", 10, "bold"), bg="#10b981", fg="white", activebackground="#059669", relief=tk.FLAT, pady=5, command=self.toggle_animation)
        self.btn_animate.pack(fill=tk.X, pady=2)

        kpi_frame = tk.LabelFrame(self.sidebar, text="Simulation Summary", font=("Helvetica", 9, "bold"), fg="#f3f4f6", bg="#374151", padx=10, pady=6)
        kpi_frame.pack(fill=tk.X, pady=(6, 0))

        self.lbl_unbuf_kpi = tk.Label(kpi_frame, text="Unbuffered: -- J (Peak: -- °C)", font=("Helvetica", 9), fg="#f87171", bg="#374151", anchor="w")
        self.lbl_unbuf_kpi.pack(fill=tk.X, pady=2)

        self.lbl_series_kpi = tk.Label(kpi_frame, text="Series PCM: -- J (Peak: -- °C)", font=("Helvetica", 9), fg="#34d399", bg="#374151", anchor="w")
        self.lbl_series_kpi.pack(fill=tk.X, pady=2)

        self.lbl_hybrid_kpi = tk.Label(kpi_frame, text="Hybrid: -- J (Peak: -- °C)", font=("Helvetica", 10, "bold"), fg="#60a5fa", bg="#374151", anchor="w")
        self.lbl_hybrid_kpi.pack(fill=tk.X, pady=2)

        self.lbl_gain_kpi = tk.Label(kpi_frame, text="Hybrid vs Unbuffered: -- %", font=("Helvetica", 9, "bold"), fg="#38bdf8", bg="#374151", anchor="w")
        self.lbl_gain_kpi.pack(fill=tk.X, pady=2)

        self.lbl_sugar_kpi = tk.Label(kpi_frame, text="Residual Sugar: Unbuffered = -- | Hybrid = --", font=("Helvetica", 9), fg="#fbbf24", bg="#374151", anchor="w")
        self.lbl_sugar_kpi.pack(fill=tk.X, pady=2)

        self.lbl_dur_kpi = tk.Label(kpi_frame, text="Duration (Delta-T >= 2.5°C): Unbuf = -- h | Hybrid = -- h", font=("Helvetica", 9), fg="#e5e7eb", bg="#374151", anchor="w")
        self.lbl_dur_kpi.pack(fill=tk.X, pady=2)

        self.fig = Figure(figsize=(11.5, 7.8), dpi=100)
        self.fig.patch.set_facecolor('#ffffff')
        self.ax1 = self.fig.add_subplot(2, 3, 1)
        self.ax2 = self.fig.add_subplot(2, 3, 2)
        self.ax3 = self.fig.add_subplot(2, 3, 3)
        self.ax4 = self.fig.add_subplot(2, 3, 4)
        self.ax5 = self.fig.add_subplot(2, 3, 5)
        self.ax6 = self.fig.add_subplot(2, 3, 6)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self.main_area)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def run_simulation_data(self):
        S_init = self.params["S_initial"]
        X_init = self.params["X_initial"]
        T_melt = self.params["T_pcm_melt"]
        m_pcm = self.params["m_pcm_g"] / 1000.0
        R_insul = self.params["R_insul"]
        T_amb = self.params["T_amb"]
        T_pitch = self.params["T_pitch"]
        S_teg = self.params["S_teg"]
        R_int = self.params["R_int"]

        V_broth = 0.50
        C_broth = 0.50 * 4184.0
        delta_H = 655.0
        L_f = 200000.0
        Cp_pcm = 2100.0
        delta_T_melt = 1.0

        T_min = 10.0
        T_opt = 31.0
        T_safe = 33.0
        T_lethal = 36.0
        mu_opt = 0.50
        k_inact = 2.0
        kd_mult = 1.5

        R_b_teg = 4.0
        R_th_teg = 2.4
        R_contact = 0.6
        R_b_pcm = 0.8
        R_sink_air = 2.0
        C_sink = 0.15 * 900.0

        X_max = 24.0
        Ks = 2.5
        Y_xs = 0.16
        q_etoh = 0.48
        m_s = 0.015

        t_total_h = 24.0
        t_span = (0.0, t_total_h * 3600.0)
        self.t_eval = np.linspace(0.0, t_total_h * 3600.0, 600)
        self.t_hours = self.t_eval / 3600.0

        def cardinal_mu(T):
            if T <= T_min or T >= T_lethal:
                return 0.0
            num = (T - T_lethal) * (T - T_min)**2
            den = (T_opt - T_min) * ((T_opt - T_min)*(T - T_opt) - (T_opt - T_lethal)*(T_opt + T_min - 2.0*T))
            if den == 0.0:
                return 0.0
            return mu_opt * max(0.0, num / den)

        def pcm_dyn_C(Tpcm):
            sigma = delta_T_melt / 2.0
            dbeta = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((Tpcm - T_melt)/sigma)**2)
            return m_pcm * (Cp_pcm + L_f * dbeta)

        def odes_u(t, y):
            X, S, Tb, Tc, V = y
            dV = -k_inact * (max(0.0, Tb - T_safe)**1.8) * V / 3600.0
            active_V = max(0.0, (V - 0.15) / 0.85) if V > 0.15 else 0.0

            mu = cardinal_mu(Tb) * (S / (Ks + S)) * active_V if S > 0.01 else 0.0
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

        def odes_s(t, y):
            X, S, Tb, Tpcm, Tc, V = y
            dV = -k_inact * (max(0.0, Tb - T_safe)**1.8) * V / 3600.0
            active_V = max(0.0, (V - 0.15) / 0.85) if V > 0.15 else 0.0

            mu = cardinal_mu(Tb) * (S / (Ks + S)) * active_V if S > 0.01 else 0.0
            kd = kd_mult * (1.0 - active_V) + 1.0 * max(0.0, Tb - T_safe)**2
            growth = mu * X * (1.0 - X/X_max)
            dX = (growth - kd * X) / 3600.0 if X > 0.001 else 0.0
            
            ferm = q_etoh * active_V * X * (S / (Ks + S)) if S > 0.01 and X > 0.001 else 0.0
            maint = m_s * active_V * X if X > 0.001 and S > 0.01 else 0.0
            dS = (-(1.0 / Y_xs) * max(0.0, growth) - ferm - maint) / 3600.0 if S > 0.01 and X > 0.001 else 0.0
            
            Q_bio = V_broth * delta_H * (-dS)
            Q_loss = (Tb - T_amb) / R_insul
            Q_to_pcm = (Tb - Tpcm) / R_b_pcm
            Q_teg = (Tpcm - Tc) / (R_th_teg + R_contact + 2.5)
            Q_sink = (Tc - T_amb) / R_sink_air
            
            C_pcm = pcm_dyn_C(Tpcm)
            dTb = (Q_bio - Q_loss - Q_to_pcm) / C_broth
            dTpcm = (Q_to_pcm - Q_teg) / C_pcm
            dTc = (Q_teg - Q_sink) / C_sink
            return [dX, dS, dTb, dTpcm, dTc, dV]

        def odes_h(t, y):
            X, S, Tb, Tpcm, Tc, V = y
            dV = -k_inact * (max(0.0, Tb - T_safe)**1.8) * V / 3600.0
            active_V = max(0.0, (V - 0.15) / 0.85) if V > 0.15 else 0.0

            mu = cardinal_mu(Tb) * (S / (Ks + S)) * active_V if S > 0.01 else 0.0
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

        sol_u = solve_ivp(odes_u, t_span, [X_init, S_init, T_pitch, T_amb, 1.0], t_eval=self.t_eval, method='Radau')
        sol_s = solve_ivp(odes_s, t_span, [X_init, S_init, T_pitch, T_amb, T_amb, 1.0], t_eval=self.t_eval, method='Radau')
        sol_h = solve_ivp(odes_h, t_span, [X_init, S_init, T_pitch, T_amb, T_amb, 1.0], t_eval=self.t_eval, method='Radau')

        self.sol_u = sol_u
        self.sol_s = sol_s
        self.sol_h = sol_h

        self.dT_u = np.maximum(0.0, sol_u.y[2] - sol_u.y[3])
        self.dT_s = np.maximum(0.0, sol_s.y[3] - sol_s.y[4])
        self.dT_h = np.maximum(0.0, sol_h.y[2] - sol_h.y[4])

        self.P_u = ((S_teg * self.dT_u)**2) / (4.0 * R_int)
        self.P_s = ((S_teg * self.dT_s)**2) / (4.0 * R_int)
        self.P_h = ((S_teg * self.dT_h)**2) / (4.0 * R_int)

        dt = np.gradient(self.t_eval)
        self.E_u = np.cumsum(self.P_u * dt)
        self.E_s = np.cumsum(self.P_s * dt)
        self.E_h = np.cumsum(self.P_h * dt)

        e_u = self.E_u[-1]
        e_s = self.E_s[-1]
        e_h = self.E_h[-1]
        dur_u = np.sum(self.dT_u >= 2.5) * (t_total_h / len(self.t_eval))
        dur_h = np.sum(self.dT_h >= 2.5) * (t_total_h / len(self.t_eval))
        gain = ((e_h - e_u) / e_u) * 100.0 if e_u > 0.01 else 0.0

        self.lbl_unbuf_kpi.config(text=f"Unbuffered: {e_u:.1f} J (Peak: {np.max(sol_u.y[2]):.1f}°C)")
        self.lbl_series_kpi.config(text=f"Series PCM: {e_s:.1f} J (Peak: {np.max(sol_s.y[2]):.1f}°C)")
        self.lbl_hybrid_kpi.config(text=f"Hybrid: {e_h:.1f} J (Peak: {np.max(sol_h.y[2]):.1f}°C)")
        self.lbl_gain_kpi.config(text=f"Hybrid vs Unbuffered: {gain:+.1f}%")
        self.lbl_sugar_kpi.config(text=f"Residual Sugar: Unbuffered = {sol_u.y[1][-1]:.1f} g/L | Hybrid = {sol_h.y[1][-1]:.1f} g/L")
        self.lbl_dur_kpi.config(text=f"Duration (Delta-T >= 2.5°C): Unbuf = {dur_u:.1f} h | Hybrid = {dur_h:.1f} h")

    def draw_plots(self, max_idx=None):
        if max_idx is None:
            max_idx = len(self.t_hours)

        t_sub = self.t_hours[:max_idx]
        T_melt = self.params["T_pcm_melt"]

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.clear()

        self.ax1.plot(t_sub, self.sol_u.y[2][:max_idx], 'r--', label=f'Unbuffered ({np.max(self.sol_u.y[2]):.1f}°C)', linewidth=1.8)
        self.ax1.plot(t_sub, self.sol_s.y[2][:max_idx], 'g:', label=f'Series PCM ({np.max(self.sol_s.y[2]):.1f}°C)', linewidth=1.8)
        self.ax1.plot(t_sub, self.sol_h.y[2][:max_idx], 'b-', label=f'Parallel Hybrid ({np.max(self.sol_h.y[2]):.1f}°C)', linewidth=2.3)
        self.ax1.axhline(T_melt, color='purple', linestyle='--', alpha=0.6, label=f'PCM Melt ({T_melt}°C)')
        self.ax1.axhline(33.0, color='darkorange', linestyle='-.', alpha=0.7, label='Stress (33°C)')
        self.ax1.axhline(36.0, color='darkred', linestyle=':', alpha=0.7, label='Lethal (36°C)')
        self.ax1.set_title("(a) Broth Temperature (°C)", fontweight="bold", fontsize=9.5)
        self.ax1.set_xlabel("Time (h)", fontsize=8.5)
        self.ax1.set_xlim(0, 24)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc="lower left", fontsize=7.2)

        s_u_end = self.sol_u.y[1][-1]
        s_h_end = self.sol_h.y[1][-1]
        self.ax2.plot(t_sub, self.sol_u.y[1][:max_idx], 'r--', label=f'Unbuffered ({s_u_end:.1f} g/L left)', linewidth=1.8)
        self.ax2.plot(t_sub, self.sol_s.y[1][:max_idx], 'g:', label='Series PCM (Complete)', linewidth=1.8)
        self.ax2.plot(t_sub, self.sol_h.y[1][:max_idx], 'b-', label=f'Parallel Hybrid ({s_h_end:.1f} g/L left)', linewidth=2.3)
        self.ax2.set_title("(b) Sugar Concentration (g/L)", fontweight="bold", fontsize=9.5)
        self.ax2.set_xlabel("Time (h)", fontsize=8.5)
        self.ax2.set_xlim(0, 24)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc="upper right", fontsize=7.2)

        self.ax3.plot(t_sub, self.sol_u.y[0][:max_idx], 'r--', label='Unbuffered', linewidth=1.8)
        self.ax3.plot(t_sub, self.sol_s.y[0][:max_idx], 'g:', label='Series PCM', linewidth=1.8)
        self.ax3.plot(t_sub, self.sol_h.y[0][:max_idx], 'b-', label='Parallel Hybrid', linewidth=2.3)
        self.ax3.set_title("(c) Living Biomass (g/L)", fontweight="bold", fontsize=9.5)
        self.ax3.set_xlabel("Time (h)", fontsize=8.5)
        self.ax3.set_xlim(0, 24)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend(loc="lower right", fontsize=7.2)

        dur_u = np.sum(self.dT_u >= 2.5) * (24.0 / len(self.t_eval))
        dur_h = np.sum(self.dT_h >= 2.5) * (24.0 / len(self.t_eval))
        self.ax4.plot(t_sub, self.dT_u[:max_idx], 'r--', label=f'Unbuffered ({dur_u:.1f}h)', linewidth=1.8)
        self.ax4.plot(t_sub, self.dT_s[:max_idx], 'g:', label='Series PCM', linewidth=1.8)
        self.ax4.plot(t_sub, self.dT_h[:max_idx], 'b-', label=f'Parallel Hybrid ({dur_h:.1f}h)', linewidth=2.3)
        self.ax4.axhline(2.5, color='orange', linestyle=':', label='Usable Threshold (ΔT ≥ 2.5°C)')
        self.ax4.set_title("(d) TEG Gradient Delta-T (°C)", fontweight="bold", fontsize=9.5)
        self.ax4.set_xlabel("Time (h)", fontsize=8.5)
        self.ax4.set_xlim(0, 24)
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend(loc="upper right", fontsize=7.2)

        p_u_peak = np.max(self.P_u) * 1000.0
        p_h_peak = np.max(self.P_h) * 1000.0
        self.ax5.plot(t_sub, self.P_u[:max_idx] * 1000.0, 'r--', label=f'Unbuffered ({p_u_peak:.1f} mW)', linewidth=1.8)
        self.ax5.plot(t_sub, self.P_s[:max_idx] * 1000.0, 'g:', label='Series PCM', linewidth=1.8)
        self.ax5.plot(t_sub, self.P_h[:max_idx] * 1000.0, 'b-', label=f'Parallel Hybrid ({p_h_peak:.1f} mW)', linewidth=2.3)
        self.ax5.set_title("(e) Electrical Power (mW)", fontweight="bold", fontsize=9.5)
        self.ax5.set_xlabel("Time (h)", fontsize=8.5)
        self.ax5.set_xlim(0, 24)
        self.ax5.grid(True, alpha=0.3)
        self.ax5.legend(loc="upper right", fontsize=7.2)

        e_u = self.E_u[-1]
        e_s = self.E_s[-1]
        e_h = self.E_h[-1]
        gain = ((e_h - e_u) / e_u) * 100.0 if e_u > 0.01 else 0.0
        self.ax6.plot(t_sub, self.E_u[:max_idx], 'r--', label=f'Unbuffered: {e_u:.1f} J', linewidth=1.8)
        self.ax6.plot(t_sub, self.E_s[:max_idx], 'g:', label=f'Series PCM: {e_s:.1f} J', linewidth=1.8)
        self.ax6.plot(t_sub, self.E_h[:max_idx], 'b-', label=f'Parallel Hybrid: {e_h:.1f} J (+{gain:.0f}%)', linewidth=2.3)
        self.ax6.set_title("(f) Harvested Energy (J)", fontweight="bold", fontsize=9.5)
        self.ax6.set_xlabel("Time (h)", fontsize=8.5)
        self.ax6.set_xlim(0, 24)
        self.ax6.grid(True, alpha=0.3)
        self.ax6.legend(loc="lower right", fontsize=7.5)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_simulation_live(self):
        self.run_simulation_data()
        self.draw_plots()

    def toggle_animation(self):
        if self.anim_running:
            self.anim_running = False
            self.btn_animate.config(text="Play 24h Animation", bg="#10b981")
        else:
            self.anim_running = True
            self.anim_index = 5
            self.btn_animate.config(text="Pause Animation", bg="#f59e0b")
            self.run_simulation_data()
            self._step_animation()

    def _step_animation(self):
        if not self.anim_running:
            return
        self.anim_index += 4
        if self.anim_index > len(self.t_hours):
            self.anim_running = False
            self.btn_animate.config(text="Play 24h Animation", bg="#10b981")
            self.draw_plots()
            return
        
        self.draw_plots(max_idx=self.anim_index)
        self.after(self.anim_step_delay, self._step_animation)

if __name__ == "__main__":
    app = BioTegSimulationGUI()
    app.mainloop()
