#!/usr/bin/env python
"""
hydration_cp_recipe_optimizer.py
Interactive tool to evaluate the compromise between CP signal intensity, 
spectrometer parameters, and hydration layer integrity based on real-world 
experimental sample preparation weights and volumes.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Physical and Thermodynamic Constants
Rb = 8.31

# PHI_M: maximum (random close) packing volume fraction in the Ross-Minton
# crowding model — the point at which the hydrated protein "jams" and
# viscosity/correlation-time formally diverge. It sets the jamming
# concentration cl0 = PHI_M / v_eff (the x-axis threshold drawn on the plots
# below). ~0.64 is the canonical value for random close packing of hard
# spheres; real proteins can range roughly 0.5-0.7 depending on shape and
# size polydispersity, so it's exposed as a slider (default 0.64) rather
# than hardcoded.
PHI_M_DEFAULT = 0.64

# EINSTEIN_COEFF: the Einstein intrinsic-viscosity coefficient for dilute
# hard spheres. Together with PHI_M it sets how steeply tau_c (and hence CP
# efficiency) blows up as concentration approaches the jamming limit:
# ln(eta_rel) = EINSTEIN_COEFF * v_eff * c / (1 - c/cl0).
EINSTEIN_COEFF = 2.5

def perrin_shape_factors(p):
    """Perrin translational frictional ratio F_T, and HullRad's empirical
    rotational shape factor F_R = F_T**4, for a prolate (p>1) or oblate
    (p<1) ellipsoid of revolution of axial ratio p = a/b, relative to a
    sphere of equal volume. Returns (1.0, 1.0) for p == 1 (sphere). A single
    number, no coordinates needed -- this is the classic Perrin (1936)
    whole-body approximation, also used for "equivalent ellipsoid" fits of
    dumbbell/elongated multi-domain assemblies. Verified against tabulated
    frictional ratios (F_T ~ 1.044 at p=2, ~1.25 at p=5, ~1.54 at p=10):
        prolate: F_T = sqrt(p^2-1) / (p^(1/3) * ln(p + sqrt(p^2-1)))
        oblate:  F_T = sqrt(1-p^2) / (p^(1/3) * arctan(sqrt(1-p^2)/p))
    F_R = F_T**4 stands in for the full Perrin rotational tensor (HullRad,
    Fleming & Fleming 2018), valid well beyond axial ratios typically seen
    for folded proteins.
    """
    if abs(p - 1.0) < 1e-6:
        return 1.0, 1.0
    if p > 1.0:
        F_T = math.sqrt(p**2 - 1.0) / (p**(1.0/3.0) * math.log(p + math.sqrt(p**2 - 1.0)))
    else:
        F_T = math.sqrt(1.0 - p**2) / (p**(1.0/3.0) * math.atan(math.sqrt(1.0 - p**2) / p))
    return F_T, F_T**4

# D_CH_RIGID / D_HH_RIGID: static (rigid-lattice) one-bond 1H-13C
# heteronuclear, and effective 1H-1H homonuclear, dipolar coupling
# constants (angular frequency, rad/s). D_CH_RIGID is the standard
# ~1.09 A C-H bond-length value (~23 kHz/2pi) quoted throughout the CPMAS
# literature. D_HH_RIGID is a representative root-second-moment proton-
# proton coupling for a densely protonated, fully rigid solid (~30-50 kHz
# FWHM proton linewidths are typical; 40 kHz used here as a round default).
# These feed a MAS- and tau_c-INDEPENDENT "solid-like" CP channel (see
# compute_cp_efficiency_vec): in a truly rigid lattice the heteronuclear
# coupling is not motionally averaged at all, and it is the homonuclear
# (not the tumbling) fluctuation that lets cross-polarisation proceed, even
# exactly on the n=0 Hartmann-Hahn condition. Kept as fixed constants
# (not sliders) since they describe generic CH/HH bond physics rather than
# a per-sample preparation choice.
D_CH_RIGID = 2 * np.pi * 23.0e3
D_HH_RIGID = 2 * np.pi * 40.0e3

N_A = 6.02214076e23             
k_B = 1.380649e-23              

def water_viscosity(T):
    """Calculates water viscosity as a function of Temperature (K)"""
    return 2.414e-5 * (10 ** (247.8 / (T - 140.0)))

def compute_cp_efficiency_vec(tau_c, omega_R, omega_0_H, omega_0_C, omega_1_H, omega_1_C, b, S, tau_s, tau_CP):
    """Vectorized calculation of local CP efficiency ratio (I_CP/I_DP)"""
    gamma_H = 267.513e6
    gamma_C = 67.262e6
    b_tesla = b * 1e-3
    w_R = 2 * np.pi * omega_R
    w_0_H = 2 * np.pi * omega_0_H
    w_1_H = 2 * np.pi * omega_1_H
    w_1_C = 2 * np.pi * omega_1_C
    
    def j_0(w):
        term1 = (1 - S**2) * (2 * tau_c) / (1 + (w * tau_c)**2)
        term2 = S**2 * (2 * tau_s) / (1 + (w * tau_s)**2)
        return term1 + term2

    # Rigid-lattice spectral density: same Lorentzian form as j_0, but the
    # fluctuation that broadens it is the homonuclear (1H-1H) dipolar
    # network (tau_HH = 1/D_HH_RIGID), not molecular tumbling. Unlike j_0,
    # this does NOT vanish as tau_c -> infinity, which is the "solid-like"
    # behaviour a fully jammed/rigid particle should show.
    tau_HH = 1.0 / D_HH_RIGID
    def j_0_rigid(w):
        return (2 * tau_HH) / (1 + (w * tau_HH)**2)

    def sidebands(j0_func, w):
        return (1/3) * (j0_func(w - w_R) + j0_func(w + w_R)) + (1/6) * (j0_func(w - 2*w_R) + j0_func(w + 2*w_R))

    def j(w):
        return sidebands(j_0, w)

    def j_rigid(w):
        return sidebands(j_0_rigid, w)

    R_1rho_H = ((gamma_H * b_tesla)**2 / 2.0) * (j(w_1_H) + j(w_0_H))

    # Heteronuclear CP rate = mobile (motionally-averaged) channel + rigid
    # (solid-like) channel. R_CH_mobile dominates while the particle still
    # tumbles fast enough for the BWR/motional-narrowing treatment to hold;
    # R_CH_rigid takes over once tau_c grows long enough that the mobile
    # channel's spectral density has collapsed. Both carry the same
    # first/second-order MAS-sideband weighting, which is what lets the
    # rigid channel remain MAS-frequency dependent even though it sits
    # exactly on the n=0 (MAS-independent) Hartmann-Hahn condition.
    R_CH_mobile = ((gamma_C * b_tesla)**2 / 2.0) * j(w_1_H - w_1_C)
    # Motional-narrowing gate: the static D_CH_RIGID coupling is only
    # "seen" once tumbling is slow enough that it is no longer averaged
    # away. Uses an exponential (Arrhenius/mollifier-type) onset rather
    # than a power law: f_rigid ~ exp(-1/x) vanishes to ALL polynomial
    # orders as x=D_CH_RIGID*tau_c -> 0, suppressing fast-tumbling/dilute
    # leakage much more strongly than a x^2/(1+x^2) gate, while still
    # saturating to 1 once tau_c grows past the 1/D_CH_RIGID threshold,
    # and with a gentler slope right at its half-max point.
    with np.errstate(divide='ignore', over='ignore'):
        f_rigid = np.where(tau_c > 0, np.exp(-1.0 / (D_CH_RIGID * tau_c)), 0.0)
    R_CH_rigid = f_rigid * (D_CH_RIGID**2 / 2.0) * j_rigid(w_1_H - w_1_C)
    R_CH = R_CH_mobile + R_CH_rigid
    
    with np.errstate(divide='ignore'):
        T_1rho_H = np.where(R_1rho_H > 0, 1.0 / R_1rho_H, np.inf)
        T_CH = np.where(R_CH > 0, 1.0 / R_CH, np.inf)
        
    efficiency_term = np.zeros_like(T_CH)
    inf_mask = np.isinf(T_1rho_H)
    close_mask = np.isclose(T_CH, T_1rho_H) & ~inf_mask
    normal_mask = ~inf_mask & ~close_mask
    
    if np.any(close_mask):
        efficiency_term[close_mask] = (tau_CP / T_1rho_H[close_mask]) * np.exp(-tau_CP / T_1rho_H[close_mask])
    if np.any(normal_mask):
        t1, tch = T_1rho_H[normal_mask], T_CH[normal_mask]
        ratio = tch / t1
        efficiency_term[normal_mask] = (np.exp(-tau_CP / t1) - np.exp(-tau_CP / tch)) / (1.0 - ratio)
        
    return (gamma_H / gamma_C) * efficiency_term


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
        F_T, F_R = perrin_shape_factors(p_axial)
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
        
        M_mol = mw * 1000.0                     
        eta0 = water_viscosity(temp)               
        V_h = M_mol * v_eff * 1e-6 / N_A * F_R
        tau_c0 = eta0 * V_h / (k_B * temp)

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
        exponent = np.clip(EINSTEIN_COEFF * F_T * phi_m * c_ratio / (1.0 - c_ratio + delta_smooth), 0, 50)
        tau_c_space = np.clip(tau_c0 * np.exp(exponent), 0, 1e6)
        
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