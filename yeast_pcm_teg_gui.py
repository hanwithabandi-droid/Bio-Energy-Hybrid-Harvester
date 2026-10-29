"""
================================================================================
LIVE INTERACTIVE DESKTOP SIMULATOR: YEAST-PCM-TEG BIO-THERMAL HARVESTER
Run directly with: py yeast_pcm_teg_gui.py
================================================================================
Provides a native GUI application with real-time slider parameter tuning,
live ODE multi-physics recalculation, and animated time-series progression.
================================================================================
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

class LiveBioTegSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Yeast-PCM-TEG Bio-Thermal Harvester: Live Digital Twin Simulator")
        self.geometry("1400x850")
        self.configure(bg="#f4f6f9")
        
        # State variables for simulation & animation
        self.anim_running = False
        self.anim_index = 0
        self.anim_step_delay = 40  # ms
        
        # Default physical parameters
        self.params = {
            "S_initial": 100.0,
            "X_initial": 4.0,
            "T_pcm_melt": 28.0,
            "m_pcm_g": 60.0,
            "R_insul": 35.0,
            "T_amb": 22.0,
            "T_pitch": 26.0,
            "S_teg": 0.045,
            "R_int": 2.5
        }
        
        self._setup_ui()
        self.run_simulation_data()
        self.draw_plots()

    def _setup_ui(self):
        # Main layout: Left sidebar for controls, right frame for live plots
        self.sidebar = tk.Frame(self, width=360, bg="#2c3e50", padx=15, pady=15)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.main_area = tk.Frame(self, bg="#ffffff")
        self.main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Header Title
        title_lbl = tk.Label(self.sidebar, text="BIO-TEG DIGITAL TWIN", font=("Helvetica", 14, "bold"), fg="#ecf0f1", bg="#2c3e50")
        title_lbl.pack(anchor="w", pady=(0, 5))
        sub_lbl = tk.Label(self.sidebar, text="Live Multi-Physics Simulation", font=("Helvetica", 9), fg="#bdc3c7", bg="#2c3e50")
        sub_lbl.pack(anchor="w", pady=(0, 15))

        # Slider Factory Function
        self.sliders = {}
        def add_slider(key, label_text, min_v, max_v, default_v, step_v, unit_text):
            frame = tk.Frame(self.sidebar, bg="#2c3e50")
            frame.pack(fill=tk.X, pady=4)
            
            lbl_row = tk.Frame(frame, bg="#2c3e50")
            lbl_row.pack(fill=tk.X)
            lbl = tk.Label(lbl_row, text=label_text, font=("Helvetica", 9, "bold"), fg="#ecf0f1", bg="#2c3e50")
            lbl.pack(side=tk.LEFT)
            val_lbl = tk.Label(lbl_row, text=f"{default_v} {unit_text}", font=("Helvetica", 9), fg="#1abc9c", bg="#2c3e50")
            val_lbl.pack(side=tk.RIGHT)
            
            def on_slide(val):
                v = round(float(val), 1)
                val_lbl.config(text=f"{v} {unit_text}")
                self.params[key] = v
                if not self.anim_running:
                    self.update_simulation_live()

            slider = ttk.Scale(frame, from_=min_v, to=max_v, orient=tk.HORIZONTAL, value=default_v, command=on_slide)
            slider.pack(fill=tk.X, pady=(2, 0))
            self.sliders[key] = (slider, val_lbl, unit_text)

        # Build interactive parameter sliders
        add_slider("S_initial", "Glucose Concentration", 40.0, 180.0, 100.0, 5.0, "g/L")
        add_slider("X_initial", "Initial Yeast Pitch", 1.0, 10.0, 4.0, 0.5, "g/L")
        add_slider("T_pcm_melt", "PCM Melting Point", 25.0, 35.0, 28.0, 0.5, "°C")
        add_slider("m_pcm_g", "PCM Buffer Mass", 10.0, 150.0, 60.0, 5.0, "g")
        add_slider("R_insul", "Vessel Insulation (R_th)", 10.0, 60.0, 35.0, 2.5, "K/W")
        add_slider("T_amb", "Ambient Room Temp", 18.0, 28.0, 22.0, 0.5, "°C")

        # Action Buttons
        btn_frame = tk.Frame(self.sidebar, bg="#2c3e50")
        btn_frame.pack(fill=tk.X, pady=(15, 10))

        self.btn_run = tk.Button(btn_frame, text="⚡ Recompute Now", font=("Helvetica", 10, "bold"), bg="#3498db", fg="white", activebackground="#2980b9", activeforeground="white", relief=tk.FLAT, pady=6, command=self.update_simulation_live)
        self.btn_run.pack(fill=tk.X, pady=3)

        self.btn_animate = tk.Button(btn_frame, text="▶ Play Live Time Animation", font=("Helvetica", 10, "bold"), bg="#2ecc71", fg="white", activebackground="#27ae60", activeforeground="white", relief=tk.FLAT, pady=6, command=self.toggle_animation)
        self.btn_animate.pack(fill=tk.X, pady=3)

        # KPI Dashboard Cards
        kpi_frame = tk.LabelFrame(self.sidebar, text="Live Output Metrics", font=("Helvetica", 9, "bold"), fg="#ecf0f1", bg="#34495e", padx=10, pady=8)
        kpi_frame.pack(fill=tk.X, pady=(10, 0))

        self.lbl_peak_temp = tk.Label(kpi_frame, text="Peak Culture: -- °C", font=("Helvetica", 9), fg="#ecf0f1", bg="#34495e", anchor="w")
        self.lbl_peak_temp.pack(fill=tk.X, pady=2)

        self.lbl_temp_supp = tk.Label(kpi_frame, text="PCM Suppression: -- °C", font=("Helvetica", 9), fg="#1abc9c", bg="#34495e", anchor="w")
        self.lbl_temp_supp.pack(fill=tk.X, pady=2)

        self.lbl_energy_buf = tk.Label(kpi_frame, text="Harvested Energy: -- J", font=("Helvetica", 9), fg="#f39c12", bg="#34495e", anchor="w")
        self.lbl_energy_buf.pack(fill=tk.X, pady=2)

        self.lbl_exergy_eff = tk.Label(kpi_frame, text="Exergy Efficiency: -- %", font=("Helvetica", 9), fg="#e74c3c", bg="#34495e", anchor="w")
        self.lbl_exergy_eff.pack(fill=tk.X, pady=2)

        # Setup Matplotlib Canvas
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.fig.patch.set_facecolor('#ffffff')
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self.main_area)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def run_simulation_data(self):
        """Solves the multi-physics ODEs using currently selected parameters."""
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
        L_f = 180000.0
        Cp_pcm = 2000.0
        delta_T_melt = 1.0
        R_b_pcm = 2.0
        R_th_teg = 2.2
        R_contact = 0.5
        R_sink_air = 2.0
        C_sink = 0.15 * 900.0
        t_span = (0.0, 20.0 * 3600.0)
        self.t_eval = np.linspace(0.0, 20.0 * 3600.0, 400)
        self.t_hours = self.t_eval / 3600.0

        def cardinal_mu(T):
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
                mu = cardinal_mu(Tb) * (S / (2.0 + S))
                dX_dt = (mu * X * (1.0 - X/22.0)) / 3600.0
                dS_dt = (-(1.0 / 0.16) * (dX_dt * 3600.0) - 0.015 * X) / 3600.0
            else:
                dX_dt, dS_dt = 0.0, 0.0
            Q_bio = V_broth * delta_H * (-dS_dt)
            Q_loss = (Tb - T_amb) / R_insul
            Q_to_pcm = (Tb - Tpcm) / R_b_pcm
            Q_teg = (Tpcm - Tc) / (R_th_teg + R_contact)
            Q_sink = (Tc - T_amb) / R_sink_air
            
            sigma = delta_T_melt / 2.0
            dbeta_dT = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((Tpcm - T_melt)/sigma)**2)
            C_pcm_dyn = m_pcm * (Cp_pcm + L_f * dbeta_dT)
            
            dTb = (Q_bio - Q_loss - Q_to_pcm) / C_broth
            dTpcm = (Q_to_pcm - Q_teg) / C_pcm_dyn
            dTc = (Q_teg - Q_sink) / C_sink
            return [dX_dt, dS_dt, dTb, dTpcm, dTc]

        def odes_unbuf(t, y):
            X, S, Tb, Tc = y
            if S > 0.001 and X < 22.0:
                mu = cardinal_mu(Tb) * (S / (2.0 + S))
                dX_dt = (mu * X * (1.0 - X/22.0)) / 3600.0
                dS_dt = (-(1.0 / 0.16) * (dX_dt * 3600.0) - 0.015 * X) / 3600.0
            else:
                dX_dt, dS_dt = 0.0, 0.0
            Q_bio = V_broth * delta_H * (-dS_dt)
            Q_loss = (Tb - T_amb) / R_insul
            Q_teg = (Tb - Tc) / (R_b_pcm + R_th_teg + R_contact)
            Q_sink = (Tc - T_amb) / R_sink_air
            dTb = (Q_bio - Q_loss - Q_teg) / C_broth
            dTc = (Q_teg - Q_sink) / C_sink
            return [dX_dt, dS_dt, dTb, dTc]

        # Fast numerical integration
        sol_unbuf = solve_ivp(odes_unbuf, t_span, [X_init, S_init, T_pitch, T_amb], t_eval=self.t_eval, method='Radau')
        sol_buf = solve_ivp(odes_buf, t_span, [X_init, S_init, T_pitch, T_amb, T_amb], t_eval=self.t_eval, method='Radau')

        self.sol_unbuf = sol_unbuf
        self.sol_buf = sol_buf

        # Post process curves
        self.deltaT_unbuf = np.maximum(0.0, sol_unbuf.y[2] - sol_unbuf.y[3])
        self.deltaT_buf = np.maximum(0.0, sol_buf.y[3] - sol_buf.y[4])

        self.P_unbuf = ((S_teg * self.deltaT_unbuf)**2) / (4.0 * R_int)
        self.P_buf = ((S_teg * self.deltaT_buf)**2) / (4.0 * R_int)

        self.E_unbuf_cum = np.cumsum(self.P_unbuf * np.gradient(self.t_eval))
        self.E_buf_cum = np.cumsum(self.P_buf * np.gradient(self.t_eval))

        # Exergy calculation
        Q_teg_buf = self.deltaT_buf / (R_th_teg + R_contact)
        Carnot_buf = np.maximum(0.0, 1.0 - (T_amb + 273.15) / (sol_buf.y[3] + 273.15))
        Exergy_rate = Q_teg_buf * Carnot_buf
        E_exergy_total = max(1e-6, np.trapz(Exergy_rate, self.t_eval))
        eta_exergy = (self.E_buf_cum[-1] / E_exergy_total) * 100.0

        # Update KPI UI Cards
        peak_unbuf = np.max(sol_unbuf.y[2])
        peak_buf = np.max(sol_buf.y[2])
        suppression = peak_unbuf - peak_buf
        self.lbl_peak_temp.config(text=f"Peak Culture: {peak_buf:.2f} °C (Unbuf: {peak_unbuf:.2f} °C)")
        self.lbl_temp_supp.config(text=f"PCM Suppression: -{suppression:.2f} °C")
        self.lbl_energy_buf.config(text=f"Harvested Energy: {self.E_buf_cum[-1]:.2f} J")
        self.lbl_exergy_eff.config(text=f"Exergy Efficiency: {eta_exergy:.2f} %")

    def draw_plots(self, max_idx=None):
        """Draws or redraws the 4 Matplotlib plots."""
        if max_idx is None:
            max_idx = len(self.t_hours)

        t_sub = self.t_hours[:max_idx]
        T_melt = self.params["T_pcm_melt"]

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        # (1) Thermal Profiles
        self.ax1.plot(t_sub, self.sol_unbuf.y[2][:max_idx], 'r--', label='Broth (Unbuffered)')
        self.ax1.plot(t_sub, self.sol_buf.y[2][:max_idx], 'b-', label='Broth (Buffered)', linewidth=2)
        self.ax1.plot(t_sub, self.sol_buf.y[3][:max_idx], 'g-.', label='PCM Buffer')
        self.ax1.axhline(T_melt, color='gray', linestyle=':', label=f'PCM Melt ({T_melt}°C)')
        self.ax1.set_title("Thermal Profiles [°C]", fontweight="bold", fontsize=10)
        self.ax1.set_xlabel("Time [hours]", fontsize=9)
        self.ax1.set_ylabel("Temperature [°C]", fontsize=9)
        self.ax1.set_xlim(0, 20)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc="upper right", fontsize=8)

        # (2) Temperature Gradient Across TEG
        self.ax2.plot(t_sub, self.deltaT_unbuf[:max_idx], 'r--', label=r'$\Delta T$ Unbuffered')
        self.ax2.plot(t_sub, self.deltaT_buf[:max_idx], 'b-', label=r'$\Delta T$ Buffered', linewidth=2)
        self.ax2.set_title(r"Thermal Gradient Across TEG ($\Delta T$)", fontweight="bold", fontsize=10)
        self.ax2.set_xlabel("Time [hours]", fontsize=9)
        self.ax2.set_ylabel(r"$\Delta T = T_h - T_c$ [°C]", fontsize=9)
        self.ax2.set_xlim(0, 20)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc="upper right", fontsize=8)

        # (3) Instantaneous Power
        self.ax3.plot(t_sub, self.P_unbuf[:max_idx] * 1000.0, 'r--', label='Unbuffered')
        self.ax3.plot(t_sub, self.P_buf[:max_idx] * 1000.0, 'b-', label='Buffered', linewidth=2)
        self.ax3.set_title("Instantaneous Harvested Power [mW]", fontweight="bold", fontsize=10)
        self.ax3.set_xlabel("Time [hours]", fontsize=9)
        self.ax3.set_ylabel("Power [mW]", fontsize=9)
        self.ax3.set_xlim(0, 20)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend(loc="upper right", fontsize=8)

        # (4) Cumulative Energy
        self.ax4.plot(t_sub, self.E_unbuf_cum[:max_idx], 'r--', label='Unbuffered')
        self.ax4.plot(t_sub, self.E_buf_cum[:max_idx], 'b-', label='Buffered', linewidth=2)
        self.ax4.set_title("Cumulative Harvested Energy [Joules]", fontweight="bold", fontsize=10)
        self.ax4.set_xlabel("Time [hours]", fontsize=9)
        self.ax4.set_ylabel("Energy [J]", fontsize=9)
        self.ax4.set_xlim(0, 20)
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend(loc="lower right", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_simulation_live(self):
        """Immediately recomputes and redraws when a slider changes."""
        self.run_simulation_data()
        self.draw_plots()

    def toggle_animation(self):
        """Toggles real-time dynamic time animation."""
        if self.anim_running:
            self.anim_running = False
            self.btn_animate.config(text="▶ Play Live Time Animation", bg="#2ecc71")
        else:
            self.anim_running = True
            self.anim_index = 5
            self.btn_animate.config(text="⏸ Pause Animation", bg="#e67e22")
            self.run_simulation_data()
            self._step_animation()

    def _step_animation(self):
        if not self.anim_running:
            return
        self.anim_index += 4
        if self.anim_index > len(self.t_hours):
            self.anim_running = False
            self.btn_animate.config(text="▶ Play Live Time Animation", bg="#2ecc71")
            self.draw_plots()
            return
        
        self.draw_plots(max_idx=self.anim_index)
        self.after(self.anim_step_delay, self._step_animation)

if __name__ == "__main__":
    app = LiveBioTegSimulator()
    app.mainloop()
