#!/usr/bin/env python
"""
hydration_cp_recipe_optimizer.py
Interactive tool to evaluate the compromise between CP signal intensity, 
spectrometer parameters, and hydration layer integrity based on real-world 
experimental sample preparation weights and volumes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from rm_cp_physics import (
    N_A, k_B, Rb, EINSTEIN_COEFF, D_CH_RIGID, D_HH_RIGID,
    water_viscosity, perrin_shape_factors,
    compute_cp_efficiency_vec, ross_minton_tau_c, derive_hydrodynamics,
)

# PHI_M: maximum (random close) packing volume fraction in the Ross-Minton
# crowding model — the point at which the hydrated protein "jams" and
# viscosity/correlation-time formally diverge. It sets the jamming
# concentration cl0 = PHI_M / v_eff (the x-axis threshold drawn on the plots
# below). ~0.64 is the canonical value for random close packing of hard
# spheres; real proteins can range roughly 0.5-0.7 depending on shape and
# size polydispersity, so it's exposed as a slider (default 0.64) rather
# than hardcoded.
PHI_M_DEFAULT = 0.64

class HydrationCPOptimizer:
    def __init__(self):
        # Initialize Canvas window
        self.fig = plt.figure('Hydration vs CP Recipe Optimizer', figsize=(13.5, 8.5))
        self.fig.subplots_adjust(left=0.09, right=0.88, top=0.92, bottom=0.46, hspace=0.45)
        
        # Upper Plot: NMR Yield parameters
        self.ax_nmr = self.fig.add_subplot(2, 1, 1)
        self.ax_nmr_right = self.ax_nmr.twinx()
        
        # Lower Plot: Hydration Structural Integrity
        self.ax_hyd = self.fig.add_subplot(2, 1, 2)
        
        # Plot Styling
        for ax in (self.ax_nmr, self.ax_hyd):
            ax.set_facecolor('#ffffff')
            ax.grid(True, linestyle=':', linewidth=0.5, color='#d0d0d0')
            
        self.ax_nmr.set_title('Cross-Polarization Yield Profile', fontsize=11, fontweight='bold')
        self.ax_nmr.set_ylabel('Total CP Signal Intensity (c × E) [Arb]', color='#1f77b4')
        self.ax_nmr_right.set_ylabel('Absolute CP Efficiency (E)', color='#2ca02c')
        self.ax_hyd.set_title('Hydration Shell Thermodynamic Health', fontsize=11, fontweight='bold')
        self.ax_hyd.set_xlabel('Uniform Protein Concentration (mg/mL)', fontsize=10)
        self.ax_hyd.set_ylabel('Hydration Layer Integrity (%)', color='#aa1111')

        # Concentration domain space for calculation and plotting (up to 850 mg/mL)
        self.c_space = np.linspace(1.0, 850.0, 500)
        
        # Curve handles
        self.line_intensity, = self.ax_nmr.plot([], [], color='#1f77b4', lw=2.5, label='Net CP Intensity')
        self.line_efficiency, = self.ax_nmr_right.plot([], [], color='#2ca02c', lw=1.5, ls='--', label='Efficiency')
        self.line_hyd, = self.ax_hyd.plot([], [], color='#aa1111', lw=2.5)
        
        # Array to track dynamically drawn guide lines to avoid duplicate overlays
        self.active_vlines = []
        
        # Build UI Controls
        self._build_sliders()
        self.refresh()

    def _build_sliders(self):
        default_bg = '#f8f9fa'
        protein_bg = '#fff3cd'  # Soft pastel amber highlight
        water_bg = '#e3f2fd'    # Soft pastel sky-blue highlight
        model_bg = '#f3e5f5'    # Soft pastel lilac highlight for crowding/interaction-model knobs
        
        # Grid layout placement parameters (3 columns, 5 rows)
        col1_x = 0.06
        slider_specs_col1 = [
            ('smw',      'Protein MW (kDa)',       10.0,  1000.0,  100.0,   col1_x, 0.35, default_bg, '#2ca02c'), 
            ('stemp',    'Temperature (K)',        260.0, 310.0,   283.0,   col1_x, 0.28, default_bg, '#2ca02c'),
            ('smas',     'MAS Speed (Hz)',         100.0, 200000.0,12000.0, col1_x, 0.21, default_bg, '#2ca02c'), # Max limit expanded to 200kHz
            ('shyd',     'Nominal Hydr. (g/g)',    0.10,  0.80,    0.40,    col1_x, 0.14, default_bg, '#2ca02c'),
            ('sprot',    'Prot Density (g/ml)',    1.10,  1.50,    1.34,    col1_x, 0.07, default_bg, '#2ca02c')
        ]
        
        col2_x = 0.39
        slider_specs_col2 = [
            ('sprot_mass', 'Protein Dry Mass (µg)',  50.0,  1000.0,  500.0,   col2_x, 0.35, protein_bg, '#ff9800'), 
            ('swater_vol', 'Added Water Vol (µL)',   0.20,  5.0,     0.88,    col2_x, 0.28, water_bg,   '#2196f3'), 
            ('sw0H',       '1H Larmor Freq (MHz)',   300.0, 1300.0,  700.0,   col2_x, 0.21, default_bg, '#2ca02c'), # Max limit expanded to 1300MHz
            ('sw1H',       '1H Nutation Freq (kHz)', 10.0,  150.0,   41.0,    col2_x, 0.14, default_bg, '#2ca02c'),
            ('stau_cp',    'CP Contact Time (µs)',   20.0,  2000.0,  100.0,   col2_x, 0.07, default_bg, '#2ca02c')
        ]

        # Column 3: crowding/interaction-model parameters. PHI_M is the
        # Ross-Minton max-packing fraction (see module docstring/comment for
        # its physical meaning); b, S, tau_s and the Hartmann-Hahn match
        # index n feed directly into compute_cp_efficiency_vec and were
        # previously hardcoded with no way to explore their effect here.
        # scav (Cavity Vol) adds extra specific volume on top of the
        # hydration shell to represent a hollow-sphere/cage-like particle:
        # its empty interior adds excluded/tumbling volume (crowding +
        # tau_c0) without adding buoyant mass, so it is mixed into v_eff
        # only, not into rho_p. Default 0 recovers the original solid-
        # particle behaviour. n defaults to 0 (the un-shifted Hartmann-Hahn
        # condition), since that is the actual condition most commonly
        # used experimentally; it remains a free slider. saxial is the
        # Perrin equivalent-ellipsoid axial ratio p=a/b (prolate/dumbbell-
        # like for p>1, oblate/disc-like for p<1); it scales tau_c0 (via
        # the empirical rotational shape factor F_T**4) and, as a simple
        # Simha-type proxy, the crowding-exponent prefactor (via F_T).
        # Default 1.0 (sphere) recovers the original behaviour.
        col3_x = 0.72
        slider_specs_col3 = [
            ('sphi',     'Max Packing φ_m',      0.50,  0.74,  PHI_M_DEFAULT, col3_x, 0.43, model_bg, '#9c27b0'),
            ('scav',     'Cavity Vol (mL/g)',    0.0,   0.60,  0.0,           col3_x, 0.37, model_bg, '#9c27b0'),
            ('saxial',   'Axial Ratio p (a/b)',  0.10,  10.0,  1.0,           col3_x, 0.32, model_bg, '#9c27b0'),
            ('sdsmooth', 'Crowding Smooth δ',    0.0,   0.20,  0.03,          col3_x, 0.26, model_bg, '#9c27b0'),
            ('sb',       'Field b (mT)',         0.005, 0.50,  0.20,          col3_x, 0.21, model_bg, '#9c27b0'),
            ('sS',       'Order Param S',        0.0,   1.0,   0.0,           col3_x, 0.15, model_bg, '#9c27b0'),
            ('stau_s',   'Slow τ_s (ms)',        0.01,  10.0,  1.0,           col3_x, 0.10, model_bg, '#9c27b0'),
            ('sn',       'HH Match Index n',     0,     4,     0,             col3_x, 0.04, model_bg, '#9c27b0', 1),
        ]
        
        self.sliders = {}
        all_specs = slider_specs_col1 + slider_specs_col2 + slider_specs_col3
        
        for spec in all_specs:
            key, label, vmin, vmax, vinit, x, y, bg_color, bar_color = spec[:9]
            step = spec[9] if len(spec) > 9 else None
            sax = self.fig.add_axes([x, y, 0.22, 0.03], facecolor=bg_color)
            if step is not None:
                sl = Slider(sax, label, vmin, vmax, valinit=vinit, valstep=step, color=bar_color)
            else:
                sl = Slider(sax, label, vmin, vmax, valinit=vinit, color=bar_color)
            sl.label.set_fontsize(9)
            sl.valtext.set_fontsize(9)
            sl.on_changed(self._on_slider_update)
            self.sliders[key] = sl

    def _on_slider_update(self, val):
        self.refresh()

    def refresh(self):
        # Clear previous indicator lines across subplots to prevent line accumulation
        for line in self.active_vlines:
            try:
                line.remove()
            except ValueError:
                pass
        self.active_vlines.clear()

        # 1. Read/Extract experimental parameters
        mw = self.sliders['smw'].val
        temp = self.sliders['stemp'].val
        w_R = self.sliders['smas'].val
        delta_hyd_nominal = self.sliders['shyd'].val
        rho_p = self.sliders['sprot'].val
        
        m_p = self.sliders['sprot_mass'].val     # Mass in micrograms
        V_w = self.sliders['swater_vol'].val     # Water volume in microliters
        
        w0H_mhz = self.sliders['sw0H'].val
        w1H_khz = self.sliders['sw1H'].val
        tau_cp_us = self.sliders['stau_cp'].val

        # Crowding/interaction-model parameters (previously hardcoded, now adjustable)
        phi_m = self.sliders['sphi'].val
        cavity_vol = self.sliders['scav'].val
        p_axial = self.sliders['saxial'].val
        delta_smooth = self.sliders['sdsmooth'].val
        b = self.sliders['sb'].val
        S = self.sliders['sS'].val
        tau_s = self.sliders['stau_s'].val * 1e-3   # ms -> s
        n_match = int(round(self.sliders['sn'].val))
        
        # 2. Derive true preparation concentration (mg/mL) accounting for protein excluded volume
        c_current = m_p / (V_w + (m_p * 1e-3) / rho_p)
        
        # Variable transformations into standard SI units
        w0H = w0H_mhz * 1e6
        w0C = w0H * (67.262e6 / 267.513e6) 
        w1H = w1H_khz * 1e3
        tau_CP = tau_cp_us * 1e-6

        # 3. Structural/Hydrodynamic scaling definitions (Ross-Minton)
        v_bar = 1.0 / rho_p
        v_eff = v_bar + delta_hyd_nominal + cavity_vol
        cl0 = (phi_m / v_eff) * 1000.0   # Mechanical Jamming Limit (mg/mL)
        
        tau_c0, F_T, F_R = derive_hydrodynamics(mw, temp, v_eff, p_axial)

        # 4. Evaluate concentration-dependent viscosity crowding and CP efficiency matrix
        # Ross-Minton: ln(eta_rel) = EINSTEIN_COEFF * v_eff * c / (1 - c/cl0); rewritten in
        # terms of c_ratio = c/cl0 the v_eff dependence cancels (v_eff*cl0 == phi_m*1000),
        # leaving a prefactor of EINSTEIN_COEFF * phi_m — recomputed live since phi_m is
        # now a slider rather than a fixed constant.
        # Ross-Minton: ln(eta_rel) = EINSTEIN_COEFF * v_eff * c / (1 - c/cl0); rewritten in
        # terms of c_ratio = c/cl0 the v_eff dependence cancels (v_eff*cl0 == phi_m*1000),
        # leaving a prefactor of EINSTEIN_COEFF * phi_m — recomputed live since phi_m is
        # now a slider rather than a fixed constant. F_T folded in as a simple Simha-type
        # proxy: non-spherical particles raise viscosity faster per unit packing fraction.
        c_ratio = np.clip(self.c_space / cl0, 0, 0.99)
        einstein_coeff_eff = EINSTEIN_COEFF * F_T * phi_m
        tau_c_space = ross_minton_tau_c(c_ratio, tau_c0, einstein_coeff_eff, phi_m, delta_smooth)
        
        w1C_hz = w1H - n_match * w_R
        cp_eff_space = compute_cp_efficiency_vec(
            tau_c_space, w_R, w0H, w0C, w1H, w1C_hz, b, S, tau_s, tau_CP
        )
        cp_eff_space = np.nan_to_num(cp_eff_space, nan=0.0)
        net_intensity = self.c_space * cp_eff_space

        # 5. Evaluate compression and dehydration of the primary hydration layer
        v_available = phi_m / (self.c_space / 1000.0)
        delta_hyd_actual = np.clip(v_available - v_bar - cavity_vol, 0, delta_hyd_nominal)
        hydration_integrity = (delta_hyd_actual / delta_hyd_nominal) * 100.0

        # Update trace lines
        self.line_intensity.set_data(self.c_space, net_intensity)
        self.line_efficiency.set_data(self.c_space, cp_eff_space)
        self.line_hyd.set_data(self.c_space, hydration_integrity)
        
        # Redraw vertical threshold lines
        line1 = self.ax_nmr.axvline(cl0, color='#ff7f0e', ls=':', lw=2)
        line2 = self.ax_hyd.axvline(cl0, color='#ff7f0e', ls=':', lw=2, label=f'Jamming Limit ({cl0:.1f} mg/mL)')
        
        line3 = self.ax_nmr.axvline(c_current, color='#9467bd', ls='-', lw=1.5)
        line4 = self.ax_hyd.axvline(c_current, color='#9467bd', ls='-', lw=1.5, 
                                     label=f'Your Recipe Yields: {c_current:.1f} mg/mL')
        
        self.active_vlines.extend([line1, line2, line3, line4])
        
        # Rescale plot frames dynamically
        self.ax_nmr.set_xlim(0, 850)
        self.ax_nmr.set_ylim(0, max(10, np.max(net_intensity) * 1.1))
        self.ax_nmr_right.set_ylim(0, max(1.0, np.max(cp_eff_space) * 1.1))
        
        self.ax_hyd.set_xlim(0, 850)
        self.ax_hyd.set_ylim(-5, 105)
        
        self.ax_hyd.legend(loc='lower left', fontsize=8)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

if __name__ == '__main__':
    optimizer = HydrationCPOptimizer()
    optimizer.show()