#!/usr/bin/env python
"""
sedplot_rm_nmr.py
Sedimentation Profiler with a standalone NMR panel for Expected CP Signal.
Includes dynamic Hartmann-Hahn match tracking, MAS profile integrations, 
and dynamic Rotor-Radius to MAS-Max limits affecting both sliders and plots.
"""

import math
import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button, RadioButtons

Rb = 8.31

# PHI_M_DEFAULT: maximum (random close) packing volume fraction in the
# Ross-Minton crowding model. This is the volume fraction at which the
# hydrated protein "jams" — the centrifugal/MAS-driven concentration profile
# c(r) asymptotes toward cl0 = PHI_M/v_eff as it approaches this limit, and
# the NMR correlation-time (tau_c) crowding correction diverges there too.
# ~0.64 is the textbook random-close-packing value for hard spheres; real
# proteins typically range ~0.5-0.7 depending on shape/polydispersity, so
# it is exposed as a control-panel slider (default 0.64) rather than a
# fixed constant.
PHI_M_DEFAULT = 0.64

# DELTA_HYDRATION_DEFAULT: bound-water hydration-shell thickness (g water /
# g protein), added to the protein's intrinsic specific volume to get the
# "effective" specific volume that gets crowded out at high concentration.
# Also now an adjustable slider rather than a fixed constant.
DELTA_HYDRATION_DEFAULT = 0.40

# EINSTEIN_COEFF: Einstein intrinsic-viscosity coefficient for dilute hard
# spheres. Together with PHI_M it sets the steepness of the Ross-Minton
# crowding exponent: ln(eta_rel) = EINSTEIN_COEFF * v_eff * c / (1 - c/cl0).
EINSTEIN_COEFF = 2.5

def perrin_shape_factors(p):
    """Perrin translational frictional ratio F_T, and HullRad's empirical
    rotational shape factor F_R = F_T**4, for a prolate (p>1) or oblate
    (p<1) ellipsoid of revolution of axial ratio p = a/b, relative to a
    sphere of equal volume. Returns (1.0, 1.0) for p == 1 (sphere). A single
    number, no coordinates needed -- the classic Perrin (1936) whole-body
    approximation, also used for "equivalent ellipsoid" fits of dumbbell/
    elongated multi-domain assemblies. Verified against tabulated
    frictional ratios (F_T ~ 1.044 at p=2, ~1.25 at p=5, ~1.54 at p=10):
        prolate: F_T = sqrt(p^2-1) / (p^(1/3) * ln(p + sqrt(p^2-1)))
        oblate:  F_T = sqrt(1-p^2) / (p^(1/3) * arctan(sqrt(1-p^2)/p))
    F_R = F_T**4 stands in for the full Perrin rotational tensor (HullRad,
    Fleming & Fleming 2018).
    """
    if abs(p - 1.0) < 1e-6:
        return 1.0, 1.0
    if p > 1.0:
        F_T = math.sqrt(p**2 - 1.0) / (p**(1.0/3.0) * math.log(p + math.sqrt(p**2 - 1.0)))
    else:
        F_T = math.sqrt(1.0 - p**2) / (p**(1.0/3.0) * math.atan(math.sqrt(1.0 - p**2) / p))
    return F_T, F_T**4

# V_CAVITY_DEFAULT: extra specific volume (mL/g) representing the empty
# interior of a hollow-sphere/cage-like particle, on top of the protein-
# shell specific volume (v_bar = 1/rp0) and the surface hydration shell
# (DELTA_HYDRATION_DEFAULT). A hollow shell tumbles, and jams against its
# neighbours, according to its full OUTER volume, not the volume implied
# by its mass and shell density -- the solvent-filled cavity adds no
# buoyant mass but adds just as much excluded/hydrodynamic volume as if it
# were solid protein. Mixed into v_eff only (cl0, tau_c0/V_h); kept out of
# rho_p/v_bar so the buoyancy term (1 - rs0/rp0) used by k2 in the
# sedimentation profile is untouched. Default 0 recovers the original
# solid-particle behaviour; exposed as a slider below.
V_CAVITY_DEFAULT = 0.0

# D_CH_RIGID / D_HH_RIGID: static (rigid-lattice) one-bond 1H-13C
# heteronuclear, and effective 1H-1H homonuclear, dipolar coupling
# constants (angular frequency, rad/s). D_CH_RIGID is the standard
# ~1.09 A C-H bond-length value (~23 kHz/2pi). D_HH_RIGID is a
# representative root-second-moment proton-proton coupling for a densely
# protonated, fully rigid solid (~30-50 kHz FWHM proton linewidths are
# typical; 40 kHz used as a round default). These feed a MAS- and
# tau_c-INDEPENDENT "solid-like" CP channel (see compute_cp_efficiency_vec):
# in a truly rigid lattice the heteronuclear coupling is not motionally
# averaged at all, and it is the homonuclear (not the tumbling) fluctuation
# that lets cross-polarisation proceed, even exactly on the n=0
# Hartmann-Hahn condition. Kept as fixed constants (generic CH/HH bond
# physics, not a per-sample preparation choice).
D_CH_RIGID = 2 * np.pi * 23.0e3
D_HH_RIGID = 2 * np.pi * 40.0e3

N_A = 6.02214076e23             
k_B = 1.380649e-23              

def water_viscosity(T):
    return 2.414e-5 * (10 ** (247.8 / (T - 140.0)))

_SECTION_COLORS = {'inner':'#aec6e8','cone':'#b2df8a','middle':'#fdbf6f','outer':'#fb9a99'}

# ── Low-level scalar integrands ──────────────────────────────────────────────

def c_cylindrical(rd, A, k2, cl0):
    e = A - k2 * rd * rd
    if e >  500.0: e =  500.0
    if e < -500.0: e = -500.0
    return cl0 / (1.0 + math.exp(e))

def c_cylindrical_vec(r, A, k2, cl0):
    return cl0 / (1.0 + np.exp(np.clip(A - k2*r*r, -500.0, 500.0)))

def c_pyramidal(h, A, k2, cl0, r1, hcc, b1):
    f = r1*(hcc - h + b1)/hcc
    return f*f*c_cylindrical(h, A, k2, cl0)

def pyra_area(h, r1, hcc, b1):
    f = r1*(hcc - h + b1)/hcc
    return f*f

def c_immobilized(rd, A, k2, cl0, k10):
    v = c_cylindrical(rd, A, k2, cl0)
    return v if v > cl0*k10 else 0.0

def build_geometry(be0, htot, h1, hfun, h2, h3, r1, r2):
    bc0 = be0 - htot
    b1  = bc0 + h1
    htc = hfun - h2 - h1
    hcc = htc / (1.0 - r2/r1)
    b2  = b1 + htc
    b3  = b2 + h2
    b4  = b3 + h3
    return bc0, b1, b2, b3, b4, hcc

def _rotor_mass_residual(A, k2, p):
    cl0, r1, r2, r3 = p['cl0'], p['r1'], p['r2'], p['r3']
    bc0, b1, b2, b3, b4, hcc = p['bc0'], p['b1'], p['b2'], p['b3'], p['b4'], p['hcc']
    theor = p['theor']
    mass = (r1**2 * integrate.quad(c_cylindrical, bc0,b1, args=(A,k2,cl0), limit=100)[0]
          + integrate.quad(c_pyramidal, b1,b2, args=(A,k2,cl0,r1,hcc,b1), limit=100)[0]
          + r2**2 * integrate.quad(c_cylindrical, b2,b3, args=(A,k2,cl0), limit=100)[0]
          + r3**2 * integrate.quad(c_cylindrical, b3,b4, args=(A,k2,cl0), limit=100)[0]
          ) * math.pi
    return (mass - theor) / theor

def _find_A(k2, p, tol=1e-6):
    A_lo = k2*p['b4']**2  + 500.0
    A_hi = k2*p['bc0']**2 - 500.0
    try:
        return optimize.brentq(_rotor_mass_residual, A_hi, A_lo, args=(k2, p), xtol=tol, maxiter=200)
    except ValueError:
        return (A_lo + A_hi) / 2.0

def compute_cylindrical_profile(r, p):
    k2    = p['m0']*(1.-p['rs0']/p['rp0'])*p['wr']**2/(2.*Rb*p['T'])
    alpha = k2*p['be']**2*(1.-p['c0']/p['cl0'])
    beta  = k2*p['be']**2*(p['c0']/p['cl0'])
    lnum  = alpha if alpha>100 else math.log(max(math.expm1(min(alpha,500.)),1e-300))
    lden  = 0.    if beta >100 else math.log(max(-math.expm1(-min(beta,500.)),1e-300))
    return c_cylindrical_vec(r, lnum-lden, k2, p['cl0'])

def compute_device_profile(r, p):
    k2 = p['m0']*(1.-p['rs0']/p['rp0'])*p['wr']**2/(2.*Rb*p['T'])
    return c_cylindrical_vec(r, _find_A(k2,p), k2, p['cl0'])

def compute_immobilized_cylindrical(mr, p):
    k3  = p['m0']*(1.-p['rs0']/p['rp0'])*(2.*np.pi*mr)**2/(2.*Rb*p['T'])
    al  = k3*p['be']**2*(1.-p['c0']/p['cl0'])
    be_ = k3*p['be']**2*(p['c0']/p['cl0'])
    lA  = (np.where(al>100, al, np.log(np.clip(np.expm1(np.clip(al,-500,500)),1e-300,None)))
         - np.where(be_>100,0., np.log(np.clip(-np.expm1(-np.clip(be_,0.,500)),1e-300,None))))
    def lse(lx,ly):
        mx=np.maximum(lx,ly)
        return mx+np.log(np.exp(np.clip(lx-mx,-500,0))+np.exp(np.clip(ly-mx,-500,0)))
    k3b2   = k3*p['be']**2
    lnt    = lA + np.log(p['k1']/(1.-p['k1']))
    l1pA   = np.where(lA>10,lA,np.log1p(np.exp(np.clip(lA,-500,500))))
    if (1.-p['rs0']/p['rp0'])>0.:
        ln,ld = lse(k3b2,lA), lse(lnt,lA)
    else:
        ln,ld = lse(lnt,lA), l1pA
    with np.errstate(divide='ignore',invalid='ignore'):
        pf = np.where(k3>0, p['cl0']/(k3*p['c0']*p['be']**2), 0.)
        u  = pf*np.where(ln>ld, ln-ld, 0.)
    return np.clip(u,0.,1.)

def compute_immobilized_device(mr, p):
    u = np.zeros(len(mr))
    for i,pp in enumerate(mr):
        k2 = p['m0']*(1.-p['rs0']/p['rp0'])*(2.*math.pi*pp)**2/(2.*Rb*p['T'])
        u[i] = max(math.pi*p['r3']**2*integrate.quad(c_immobilized,p['b3'],p['b4'],
                   args=(_find_A(k2, p),k2,p['cl0'],p['k1']),limit=100)[0]/p['theor'], 0.)
    return u


# ── Vectorized NMR CP Efficiency Model ───────────────────────────────────────

def compute_cp_efficiency_vec(tau_c, omega_R, omega_0_H, omega_0_C, omega_1_H, omega_1_C, b, S, tau_s, tau_CP):
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


# ── Standalone NMR Panel ─────────────────────────────────────────────────────

class NMRPanel:
    def __init__(self):
        self.fig = plt.figure('NMR CP Expected Signal', figsize=(8, 9))
        self.fig.patch.set_facecolor('#fefefe')
        
        self.ax1 = self.fig.add_axes([0.12, 0.58, 0.80, 0.35])
        self.ax2 = self.fig.add_axes([0.12, 0.35, 0.80, 0.18])
        
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor('#ffffff')
            ax.grid(True, linestyle=':', linewidth=0.5)
            
        self.ax1.set(ylabel='CP Efficiency (I_CP/I_DP)', title='Expected Topgaard CP Signal')
        self.ax2.set(ylabel='Weighted Avg CP', xlabel='MAS Frequency (Hz)')
        
        self.lcp, = self.ax1.plot([], [], lw=2, color='#2ca02c', label='Local Efficiency E(r)')
        self.lcp_avg, = self.ax1.plot([], [], lw=2, color='#d62728', ls='--', label='Weighted Avg: 0.00')
        self.ax1.legend(loc='upper left', fontsize=8)
        
        self.lcp_mas, = self.ax2.plot([], [], lw=2, color='#1f77b4')
        self.lcp_mas_mark, = self.ax2.plot([], [], 'ro', markersize=6, zorder=5)
        
        self.sliders = {}
        axc = '#e0e0e0'
        # NOTE: w0C (13C Larmor frequency) is intentionally *not* an
        # independent slider. For any real magnet, w0C = w0H * (gamma_C /
        # gamma_H) is fixed by the gyromagnetic ratios — letting the two be
        # set independently would allow physically impossible field
        # combinations. It is derived from w0H below and displayed as a
        # read-out instead.
        specs = [
            ('w0H', '1H ν₀ (MHz)', 100, 1000, 500,   0.10, 0.22, None),
            ('w1H', '1H ν₁ (kHz)', 10, 200, 100,     0.10, 0.10, None),
            ('n',   'HH match n', 0, 4, 0,           0.10, 0.04, 1),
            ('b',   'Field b (mT)', 0.005, 0.50, 0.20,0.55, 0.22, None),
            ('S',   'Order S', 0.0, 1.0, 0.0,        0.55, 0.16, None),
            ('ts',  'Slow τ_s (ms)', 0.01, 10, 1.0,  0.55, 0.10, None),
            ('tcp', 'τ_CP (ms)', 0.1, 10.0, 1.0,     0.55, 0.04, None),
        ]
        
        for key, lbl, vmin, vmax, vi, x, y, step in specs:
            sax = self.fig.add_axes([x, y, 0.30, 0.035], facecolor=axc)
            if step is not None:
                sl = Slider(sax, lbl, vmin, vmax, valinit=vi, valstep=step)
            else:
                sl = Slider(sax, lbl, vmin, vmax, valinit=vi)
            sl.label.set_fontsize(8); sl.valtext.set_fontsize(8)
            sl.on_changed(self._on_change)
            self.sliders[key] = sl

        # Read-out for the derived (not independently settable) 13C Larmor frequency
        self.w0C_label = self.fig.text(0.10, 0.17, '', fontsize=8.5, color='#555555')

        self.p = None
        self.r_m = None
        self.tau_c_r = None
        self.w_MAS = None
        self.c_r = None
        self.max_mas = 120000.0

    def _on_change(self, val):
        self.refresh()
        
    def update_from_sed(self, p, r_m, tau_c_r, w_MAS, c_r, max_mas):
        self.p = p
        self.r_m = r_m
        self.tau_c_r = tau_c_r
        self.w_MAS = w_MAS
        self.c_r = c_r
        self.max_mas = max_mas
        self.refresh()
        
    def _compute_mas_curve(self, nmr_p):
        p = self.p
        mr = np.linspace(100., self.max_mas, 120)  # Dynamically limit evaluation trace length
        cp_avg_arr = np.zeros_like(mr)
        
        r_m_coarse = np.linspace(0., p['be'], 150)
        r_weight = r_m_coarse
        
        w0H, w0C, w1H = nmr_p['w0H'], nmr_p['w0C'], nmr_p['w1H']
        n_match, b, S, ts, tcp = nmr_p['n'], nmr_p['b'], nmr_p['S'], nmr_p['ts'], nmr_p['tcp']
        
        for i, w_R in enumerate(mr):
            k2 = p['m0']*(1.-p['rs0']/p['rp0'])*(w_R*2*np.pi)**2/(2.*Rb*p['T'])
            alpha = k2*p['be']**2*(1.-p['c0']/p['cl0'])
            beta  = k2*p['be']**2*(p['c0']/p['cl0'])
            lnum  = alpha if alpha>100 else math.log(max(math.expm1(min(alpha,500.)),1e-300))
            lden  = 0.    if beta >100 else math.log(max(-math.expm1(-min(beta,500.)),1e-300))
            c_r_local = c_cylindrical_vec(r_m_coarse, lnum-lden, k2, p['cl0'])
            
            c_ratio = np.clip(c_r_local / p['cl0'], 0, 0.99)
            exponent = np.clip(EINSTEIN_COEFF * p['f_t_axial'] * p['phi_m'] * c_ratio / (1.0 - c_ratio + p['delta_smooth']), 0, 50)
            tau_c_local = np.clip(p['tau_c0'] * np.exp(exponent), 0, 1e6)
            
            w1C_hz = w1H - n_match * w_R
            cp_eff = compute_cp_efficiency_vec(
                tau_c_local, w_R, w0H, w0C, w1H, w1C_hz, b, S, ts, tcp
            )
            cp_eff = np.nan_to_num(cp_eff, nan=0.0)
            
            weights = c_r_local * r_weight
            tot_weight = np.sum(weights)
            if tot_weight > 0:
                cp_avg_arr[i] = np.sum(cp_eff * weights) / tot_weight
            else:
                cp_avg_arr[i] = 0.0
                
        return mr, cp_avg_arr
        
    def refresh(self):
        if self.r_m is None or self.tau_c_r is None: return
        
        sl = self.sliders
        w0H_hz = sl['w0H'].val * 1e6
        # w0C is locked to w0H by the 13C/1H gyromagnetic ratio, not an
        # independent slider (see __init__ note) — this is what any real
        # spectrometer enforces for a given magnet.
        w0C_hz = w0H_hz * (67.262e6 / 267.513e6)
        nmr_p = {
            'w0H': w0H_hz, 'w0C': w0C_hz,
            'w1H': sl['w1H'].val * 1e3, 'n': sl['n'].val,
            'b': sl['b'].val, 'S': sl['S'].val,
            'ts': sl['ts'].val * 1e-3, 'tcp': sl['tcp'].val * 1e-3
        }
        self.w0C_label.set_text(f"13C ν₀ (locked): {w0C_hz/1e6:.1f} MHz")
        
        w1C_hz = nmr_p['w1H'] - nmr_p['n'] * self.w_MAS 
        
        cp_eff = compute_cp_efficiency_vec(
            self.tau_c_r, self.w_MAS, nmr_p['w0H'], nmr_p['w0C'], nmr_p['w1H'], 
            w1C_hz, nmr_p['b'], nmr_p['S'], nmr_p['ts'], nmr_p['tcp']
        )
        cp_eff = np.nan_to_num(cp_eff, nan=0.0)
        
        weights = self.c_r * self.r_m
        tot_weight = np.sum(weights)
        current_cp_avg = np.sum(cp_eff * weights) / tot_weight if tot_weight > 0 else 0.0
        
        mr_mas, cp_avg_arr = self._compute_mas_curve(nmr_p)
        
        self.lcp.set_data(self.r_m * 1e3, cp_eff)
        self.lcp_avg.set_data([self.r_m[0]*1e3, self.r_m[-1]*1e3], [current_cp_avg, current_cp_avg])
        self.lcp_avg.set_label(f'Weighted Avg: {current_cp_avg:.2f}')
        self.ax1.legend(loc='upper left', fontsize=8) 
        self.ax1.set_xlim(self.r_m[0]*1e3, self.r_m[-1]*1e3)
        self.ax1.set_ylim(0, max(4.0, np.max(cp_eff)*1.1))
        
        self.lcp_mas.set_data(mr_mas, cp_avg_arr)
        self.lcp_mas_mark.set_data([self.w_MAS], [current_cp_avg])
        self.ax2.set_xlim(mr_mas[0], mr_mas[-1])
        self.ax2.set_ylim(0, max(4.0, np.max(cp_avg_arr)*1.1))
        
        self.fig.canvas.draw_idle()


# ── Interactive Sedimentation Application ────────────────────────────────────

class SedPlot:
    _SPECS = {
        'Cylindrical': [
            ('srad','Rotor Radius (mm)',         0.15, 4.0,   1.2),
            ('smas','MAS Frequency (Hz)',        100,  120000, 12000), 
            ('sds', 'Solvent Density (g/ml)',    0.90, 1.63,  0.992),
            ('sdp', 'Protein Density (g/ml)',    1.20, 1.45,  1.42),
            ('scin','Initial Conc. (mg/ml)',     1,    700,   30.0),
            ('sm',  'Protein MW (kDa)',           1,   1000,  100.0),
            ('st',  'Temperature (K)',            227,  373,   274.0),
            ('sphi','Max Packing φ_m',           0.50, 0.74,  PHI_M_DEFAULT),
            ('shyd','Hydration Shell (g/g)',     0.10, 0.80,  DELTA_HYDRATION_DEFAULT),
            ('scav','Cavity Vol (mL/g)',         0.0,  0.60,  V_CAVITY_DEFAULT),
            ('saxial','Axial Ratio p (a/b)',     0.10, 10.0,  1.0),
            ('sdsmooth','Crowding Smooth δ',      0.0,  0.20,  0.03),
        ],
        'Device': [
            ('srad','Rotor Radius (mm)',         50,   200,   152),
            ('smas','Rotor Speed (rpm)',         100,  32000, 533),
            ('sds', 'Solvent Density (g/ml)',    0.90, 1.20,  0.99),
            ('sdp', 'Protein Density (g/ml)',    1.20, 1.45,  1.23),
            ('scin','Initial Conc. (mg/ml)',     1,    700,   30.0),
            ('sm',  'Protein MW (kDa)',           1,   1000,  100.0),
            ('st',  'Temperature (K)',            227,  373,   274.0),
            ('sphi','Max Packing φ_m',           0.50, 0.74,  PHI_M_DEFAULT),
            ('shyd','Hydration Shell (g/g)',     0.10, 0.80,  DELTA_HYDRATION_DEFAULT),
            ('scav','Cavity Vol (mL/g)',         0.0,  0.60,  V_CAVITY_DEFAULT),
            ('saxial','Axial Ratio p (a/b)',     0.10, 10.0,  1.0),
            ('sdsmooth','Crowding Smooth δ',      0.0,  0.20,  0.03),
        ],
    }

    _DEV = dict(htot=0.063,h3=0.015,r3=0.0017,h2=0.001,r2=0.0001, hfun=0.03,h1=0.0,r1=0.00605)

    def __init__(self):
        d=self._DEV
        self.htot=d['htot']; self.h3=d['h3']; self.r3=d['r3']
        self.h2=d['h2'];     self.r2=d['r2']; self.hfun=d['hfun']
        self.h1=d['h1'];     self.r1=d['r1']
        self.geometry='Cylindrical'
        self.sliders={}
        self._patches=[]; self._busy=False
        
        self.nmr_panel = NMRPanel() 
        self._build_plots()
        self._build_controls()

    def _build_plots(self):
        self.fig_p = plt.figure('Sedimentation Profile', figsize=(7,6))
        self.fig_p.subplots_adjust(left=0.15, right=0.93, top=0.92, bottom=0.12, hspace=0.35)
        
        self.ax_c = self.fig_p.add_subplot(2,1,1)
        self.ax_f = self.fig_p.add_subplot(2,1,2)
        
        for ax in (self.ax_c, self.ax_f):
            ax.set_facecolor('#fefefe')
            ax.grid(True,linestyle=':',linewidth=0.5)
            
        self.ax_c.set(ylabel='Concentration (mg/ml)', title='Concentration Profile')
        self.ax_f.set(xlabel='Frequency', ylabel='Fraction', title='Immobilized Protein Fraction')

        self.lc, = self.ax_c.plot([],[],lw=2,color='#d62728',label='c(r)')
        self.lt, = self.ax_c.plot([],[],lw=1.5,ls='--',color='#1f77b4',label='Threshold')
        
        self.lf, = self.ax_f.plot([],[],lw=2,color='#d62728',label='Immobilized mass fraction')
        self.lx, = self.ax_f.plot([],[],lw=2,color='#1f77b4',label='Rigid pellet vol. fraction')
        
        self.ax_c.legend(loc='upper left',fontsize=8)
        self.ax_f.legend(loc='upper left',fontsize=8)

    def _build_controls(self):
        self.fig_ctrl = plt.figure('SedPlot Controls', figsize=(6,6.2))
        self.fig_ctrl.patch.set_facecolor('#f0f0f0')
        axc='#dcdcdc'
        ys = np.linspace(0.88, 0.16, len(self._SPECS['Cylindrical']))
        
        for (key,lbl,vmin,vmax,vi),y in zip(self._SPECS['Cylindrical'],ys):
            ax=self.fig_ctrl.add_axes([0.15,y,0.65,0.045],facecolor=axc)
            sl=Slider(ax,lbl,vmin,vmax,valinit=vi)
            sl.on_changed(self._on_slider)
            self.sliders[key]=sl

        ax_r=self.fig_ctrl.add_axes([0.15,0.02,0.40,0.12],facecolor=axc)
        self.radio=RadioButtons(ax_r,('Cylindrical','Device'),active=0)
        self.radio.on_clicked(self._on_geometry)
        
        ax_b=self.fig_ctrl.add_axes([0.65,0.05,0.15,0.08])
        self.btn=Button(ax_b,'Reset',color=axc,hovercolor='0.75')
        self.btn.on_clicked(self._on_reset)
        self._refresh()

    def _clear_patches(self):
        for a in self._patches:
            try: a.remove()
            except Exception: pass
        self._patches.clear()

    def _draw_patches(self, p):
        self._clear_patches()
        bc0,b1,b2,b3,b4 = p['bc0'],p['b1'],p['b2'],p['b3'],p['b4']
        for name,(x0,x1) in zip(['inner','cone','middle','outer'], [(bc0,b1),(b1,b2),(b2,b3),(b3,b4)]):
            self._patches.append(self.ax_c.axvspan(x0*1e3,x1*1e3,alpha=0.22,color=_SECTION_COLORS[name],zorder=1))

    def _params(self):
        sl=self.sliders
        be=sl['srad'].val/1000.
        freq=sl['smas'].val if self.geometry=='Cylindrical' else sl['smas'].val/60.
        wr=freq*2.*math.pi
        rs,rp=sl['sds'].val,sl['sdp'].val
        c0=sl['scin'].val
        m0,T=sl['sm'].val,sl['st'].val
        phi_m=sl['sphi'].val
        delta_hyd=sl['shyd'].val
        cavity_vol=sl['scav'].val
        p_axial=sl['saxial'].val
        f_t_axial, f_r_axial = perrin_shape_factors(p_axial)
        delta_smooth=sl['sdsmooth'].val

        v_bar = 1.0 / rp
        v_eff = v_bar + delta_hyd + cavity_vol
        cl0 = phi_m / v_eff * 1000.0

        M_mol = m0 * 1000.0                     
        eta0 = water_viscosity(T)               
        V_h = M_mol * v_eff * 1e-6 / N_A * f_r_axial
        tau_c0 = eta0 * V_h / (k_B * T)         

        if self.geometry == 'Cylindrical':
            eta_r_target = 0.001 / tau_c0 if tau_c0 > 0 else 1e30
            if eta_r_target <= 1.0: cr0 = 0.0
            else:
                c_star = cl0 / 1000.0               
                y = math.log(eta_r_target) / (EINSTEIN_COEFF * f_t_axial * v_eff * c_star)
                cr0 = min(max((y / (1.0 + y) if y > 0 else 0.0) * c_star * 1000.0, 0.0), cl0)
        else:
            cr0 = 0.9 * cl0

        bc0,b1,b2,b3,b4,hcc=build_geometry(be,self.htot,self.h1,self.hfun,self.h2,self.h3,self.r1,self.r2)
        Vdev=math.pi*(self.r1**2*(b1-bc0)+integrate.quad(pyra_area,b1,b2,args=(self.r1,hcc,b1))[0]+self.r2**2*(b3-b2)+self.r3**2*(b4-b3))
        
        return dict(be=be,be0=be,wr=wr,freq=freq,rs0=rs,rp0=rp,c0=c0,cl0=cl0,cr0=cr0,m0=m0,T=T,
                    tau_c0=tau_c0,k1=cr0/cl0,r1=self.r1,r2=self.r2,r3=self.r3,
                    bc0=bc0,b1=b1,b2=b2,b3=b3,b4=b4,hcc=hcc,theor=Vdev*c0,phi_m=phi_m,
                    f_t_axial=f_t_axial, delta_smooth=delta_smooth)

    def _get_max_mas(self, srad):
        radii = np.array([0.15, 0.25, 0.45, 0.57, 0.75, 0.85, 1.0,  1.5,  2.0,  3.0,  4.0])
        mas   = np.array([110,  80,   67,   45,   42,   35,   24,   15,   10,   7,    5]) * 1000
        return np.interp(srad, radii, mas) * 1.20

    def _refresh(self):
        if self._busy: return
        self._busy=True
        p=self._params()
        
        if self.geometry=='Cylindrical':
            max_mas = self._get_max_mas(p['be'] * 1000.0)
            r_m=np.linspace(0.,p['be'],2000)
            s=compute_cylindrical_profile(r_m,p)
            mr=np.linspace(100., max_mas, 700) # Dynamic MAS limits for Sedplot
            u=compute_immobilized_cylindrical(mr,p)
            w_MAS = p['freq']
            
            c_ratio = np.clip(s / p['cl0'], 0, 0.99)
            exponent = np.clip(EINSTEIN_COEFF * p['f_t_axial'] * p['phi_m'] * c_ratio / (1.0 - c_ratio + p['delta_smooth']), 0, 50)
            tau_c_r = np.clip(p['tau_c0'] * np.exp(exponent), 0, 1e6)
            self.nmr_panel.fig.set_visible(True)
            self.nmr_panel.update_from_sed(p, r_m, tau_c_r, w_MAS, s, max_mas) # Pass max_mas to NMR panel
            
        else:
            max_rpm = self.sliders['smas'].valmax
            r_m=np.linspace(p['bc0'],p['b4'],2000)
            s=compute_device_profile(r_m,p)
            mr=np.linspace(16., max_rpm, 260)
            u=compute_immobilized_device(mr,p)
            self.nmr_panel.fig.set_visible(False)

        uc = np.array(u)*p['c0']/p['cr0']
            
        self.lc.set_data(r_m*1e3,s)
        self.lt.set_data(r_m*1e3,np.full(len(r_m), p['cr0']))
        self.lf.set_data(mr,u)
        self.lx.set_data(mr,uc)
        
        self.ax_c.set_xlim(r_m[0]*1e3,r_m[-1]*1e3)
        self.ax_c.set_ylim(0,p['cl0']*1.1)
        self.ax_f.set_xlim(mr[0],mr[-1]) # Dynamically apply xlim 
        self.ax_f.set_ylim(0,1.1)
        
        xlabel = 'MAS frequency (Hz)' if self.geometry=='Cylindrical' else 'Rotor speed (rpm)'
        self.ax_f.set_xlabel(xlabel, fontsize=10)
        
        if self.geometry=='Device': self._draw_patches(p)
        else: self._clear_patches()
            
        self.fig_p.canvas.draw_idle()
        self._busy=False

    def _on_slider(self, _=None):
        if self.geometry == 'Cylindrical':
            sl_mas = self.sliders['smas']
            sl_rad = self.sliders['srad']
            max_mas = self._get_max_mas(sl_rad.val)
            
            if sl_mas.valmax != max_mas:
                sl_mas.valmax = max_mas
                sl_mas.ax.set_xlim(sl_mas.valmin, max_mas)
                if sl_mas.val > max_mas:
                    sl_mas.set_val(max_mas)
                    return 
        self._refresh()

    def _on_geometry(self, label):
        if label==self.geometry: return
        self.geometry=label
        for key,lbl,vmin,vmax,vi in self._SPECS[label]:
            sl=self.sliders[key]
            with sl._observers.blocked():
                sl.label.set_text(lbl)
                sl.valmin=vmin; sl.valmax=vmax; sl.valinit=vi
                sl.ax.set_xlim(vmin,vmax)
                sl.set_val(vi)
        self.fig_ctrl.canvas.draw_idle()
        self._refresh()

    def _on_reset(self, _=None):
        for key,_l,_n,_x,vi in self._SPECS[self.geometry]:
            sl=self.sliders[key]
            with sl._observers.blocked(): sl.set_val(vi)
        self.fig_ctrl.canvas.draw_idle()
        self._refresh()

    def show(self): plt.show()

if __name__=='__main__':
    app=SedPlot()
    app.show()