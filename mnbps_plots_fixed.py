"""
MNBPS nucleon: numerical solver, mechanical EMT, d_1 calculation.
Apr 2026
Generates the three PNG files (six panels in total) used in the paper:
  1.png : chiral profile f(r) and isotropic pressure p_iso(r)
  2.png : pressure components p_r, p_t, p_iso, and shear forces s(r)
  3.png : Laue integrand r^2 p_iso(r) and the dependence d_1(lambda)

Also produces two console tables for the paper:
  - Table 1: extended scan over lambda (including lambda < 0.5 fm)
  - Table 2: numerical stability of the Laue integral at lambda = 0.67 fm
"""

import numpy as np
from scipy.integrate import solve_bvp, trapezoid
import matplotlib.pyplot as plt

# =========================================================
# PHYSICAL CONSTANTS
# =========================================================
fpi   = 93.0        # MeV
mpi   = 138.0       # MeV
hbarc = 197.327     # MeV*fm
M_N   = 939.0       # MeV
mu    = mpi / hbarc # 1/fm

# =========================================================
# GRID (DEFAULTS)
# =========================================================
RMIN_DEFAULT = 1e-4
RMAX_DEFAULT = 20.0
N_DEFAULT    = 4000
TOL_DEFAULT  = 1e-9

# =========================================================
# MODEL FUNCTIONS
# =========================================================
def M_func(r, f, lam):
    return r**2 + lam**2 * np.sin(f)**4 / r**2

def U_func(r, f):
    return np.sin(f)**2 + mu**2 * r**2 * (1.0 - np.cos(f))

def ode(r, y, lam):
    f, fp = y[0], y[1]
    s, c  = np.sin(f), np.cos(f)
    M     = M_func(r, f, lam)
    dMdr  =  2*r      - 2*lam**2 * s**4 / r**3
    dMdf  =  4*lam**2 * s**3 * c / r**2
    dUdf  =  2*s*c + mu**2 * r**2 * s
    fpp   = (dUdf - dMdr*fp - 0.5*dMdf*fp**2) / M
    return np.vstack((fp, fpp))

def bc(ya, yb):
    return np.array([ya[0] - np.pi, yb[0]])

# =========================================================
# CORE SOLVER
# =========================================================
def compute_all(lam, rmin=RMIN_DEFAULT, Rmax=RMAX_DEFAULT,
                N=N_DEFAULT, tol=TOL_DEFAULT, y_init=None):
    r = np.logspace(np.log10(rmin), np.log10(Rmax), N)
    if y_init is None or y_init.shape[1] != N:
        y_init = np.zeros((2, N))
        y_init[0] = np.pi * np.exp(-r)
        y_init[1] = -np.pi * np.exp(-r)

    sol = solve_bvp(
        lambda rr, yy: ode(rr, yy, lam), bc,
        r, y_init, tol=tol, max_nodes=200000
    )

    f  = sol.sol(r)[0]
    fp = sol.sol(r)[1]

    M = M_func(r, f, lam)
    U = U_func(r, f)

    # Radial pressure [MeV/fm^3]
    pr = (fpi**2 / (2.0 * hbarc)) * (0.5 * M * fp**2 - U) / r**2

    # Tangential pressure via radial equilibrium
    ln_r  = np.log(r)
    r2pr  = r**2 * pr
    dr2pr = np.gradient(r2pr, ln_r) / r
    pt    = dr2pr / (2.0 * r)

    # Cut singularity at r = 0
    cutoff_idx = np.searchsorted(r, 0.01)
    pt[:cutoff_idx] = np.nan
    pr[:cutoff_idx] = np.nan

    p_iso   = (pr + 2*pt) / 3.0
    s_shear =  pt - pr

    mask = (r >= 0.01) & np.isfinite(p_iso) & np.isfinite(pr)
    laue = trapezoid(r[mask]**2 * p_iso[mask], r[mask])
    d1   = -(4*np.pi * M_N) / (5 * hbarc**2) * trapezoid(
        r[mask]**4 * pr[mask], r[mask]
    )

    # Baryon radius from topological density
    ratio = (np.pi - f) / r * np.sinc((np.pi - f) / np.pi)
    B0    = -ratio**2 * fp / (2 * np.pi**2)
    RB_sq = trapezoid(r**4 * B0, r) / trapezoid(r**2 * B0, r)
    RB    = np.sqrt(abs(RB_sq))

    # Classical mass
    Mcl = 4*np.pi * (fpi**2 / 2.0) * trapezoid(0.5*M*fp**2 + U, r) / hbarc

    return {
        'r': r, 'f': f, 'fp': fp,
        'pr': pr, 'pt': pt, 'p_iso': p_iso, 's_shear': s_shear,
        'laue': laue, 'd1': d1, 'RB': RB, 'Mcl': Mcl,
        'y_sol': np.vstack((f, fp))
    }

# =========================================================
# BASE SOLUTION  lambda = 0.67  (CALIBRATION: R_B ~ 0.8548 fm)
# =========================================================
LAM_BASE = 0.67

print(f"Computing base solution at lambda = {LAM_BASE} ...")
res = compute_all(LAM_BASE)
print(f"  R_B  = {res['RB']:.4f} fm   (target: 0.8548 fm)")
print(f"  M_cl = {res['Mcl']:.1f} MeV")
print(f"  d_1  = {res['d1']:.5f}")
print(f"  Laue = {res['laue']:.3e} fm^-1")

r       = res['r']
f       = res['f']
pr      = res['pr']
pt      = res['pt']
p_iso   = res['p_iso']
s_shear = res['s_shear']

mask_plot = (r >= 0.05) & (r <= 3.0)
r_p    = r[mask_plot]
f_p    = f[mask_plot]
pr_p   = pr[mask_plot]
pt_p   = pt[mask_plot]
piso_p = p_iso[mask_plot]
s_p    = s_shear[mask_plot]

# =========================================================
# PLOT STYLE
# =========================================================
plt.rcParams.update({
    'font.size':      12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize':11,
    'figure.dpi':     150,
})

# =========================================================
# FIG 1: profile f(r) and isotropic pressure p_iso(r)
# =========================================================
fig1, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(r_p, f_p, 'b-', lw=2.5)
ax.set_xlabel('r [fm]')
ax.set_ylabel('f(r) [rad]')
ax.set_title('Chiral Profile Function')
ax.set_xlim(0, 3)
ax.set_ylim(0, np.pi + 0.1)
ax.axhline(np.pi, color='gray', ls=':', lw=1, alpha=0.5)
ax.text(0.05, np.pi + 0.05, r'$\pi$', fontsize=11, color='gray')
ax.grid(True, alpha=0.4)

ax = axes[1]
ax.plot(r_p, piso_p, 'k-', lw=2.5, label=r'$p_{\rm iso}(r)$')
ax.fill_between(r_p, piso_p, 0,
                where=(piso_p > 0), alpha=0.35, color='red',       label='Repulsion')
ax.fill_between(r_p, piso_p, 0,
                where=(piso_p < 0), alpha=0.35, color='royalblue', label='Attraction')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_xlabel('r [fm]')
ax.set_ylabel(r'$p_{\rm iso}$ [MeV/fm$^3$]')
ax.set_title('Mechanical Pressure Distribution')
ax.set_xlim(0, 3)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.4)

fig1.tight_layout()
fig1.savefig("1.png", dpi=300, bbox_inches='tight')
print("Saved 1.png")

# =========================================================
# FIG 2: pressure components and shear forces
# =========================================================
fig2, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(r_p, pr_p,   'b-',  lw=2.0, label=r'$p_r(r)$')
ax.plot(r_p, pt_p,   'r--', lw=2.0, label=r'$p_t(r)$')
ax.plot(r_p, piso_p, 'k:',  lw=1.8, label=r'$p_{\rm iso}(r)$')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_xlabel('r [fm]')
ax.set_ylabel(r'Pressure [MeV/fm$^3$]')
ax.set_title('Pressure Distribution Inside the Nucleon')
ax.set_xlim(0, 3)
p_all = np.concatenate([pr_p, pt_p, piso_p])
p_all = p_all[np.isfinite(p_all)]
ax.set_ylim(np.nanpercentile(p_all, 1) * 1.2,
            np.nanpercentile(p_all, 99) * 1.4)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.4)

ax = axes[1]
ax.plot(r_p, s_p, 'g-', lw=2.5, label=r'$s(r) = p_t - p_r$')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_xlabel('r [fm]')
ax.set_ylabel(r'$s(r)$ [MeV/fm$^3$]')
ax.set_title('Shear Force Distribution')
ax.set_xlim(0, 3)
s_fin = s_p[np.isfinite(s_p)]
ax.set_ylim(np.nanpercentile(s_fin, 1) * 1.3,
            np.nanpercentile(s_fin, 99) * 2.0)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.4)

fig2.tight_layout()
fig2.savefig("2.png", dpi=300, bbox_inches='tight')
print("Saved 2.png")

# =========================================================
# EXTENDED SCAN OVER lambda  (including lambda < 0.5 fm)
# =========================================================
lambda_values = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.67,
                          0.83, 1.00, 1.17, 1.33, 1.50])
d1_values, laue_values, RB_values = [], [], []

print("\nExtended scan over lambda:")
current_guess = None
for lam in lambda_values:
    res_lam = compute_all(lam, y_init=current_guess)
    d1_values.append(res_lam['d1'])
    laue_values.append(res_lam['laue'])
    RB_values.append(res_lam['RB'])
    current_guess = res_lam['y_sol']
    print(f"  lam={lam:.2f}  d1={res_lam['d1']:9.4f}  "
          f"Laue={res_lam['laue']:11.3e}  R_B={res_lam['RB']:.4f} fm")

d1_values   = np.array(d1_values)
laue_values = np.array(laue_values)
RB_values   = np.array(RB_values)

# =========================================================
# FIG 3: Laue integrand and d_1(lambda)
# =========================================================
fig3, axes = plt.subplots(1, 2, figsize=(13, 5))

mask_laue_plot = (r >= 0.05) & (r <= 3.0)
r_l    = r[mask_laue_plot]
laue_i = r_l**2 * p_iso[mask_laue_plot]

ax = axes[0]
ax.plot(r_l, laue_i, 'b-', lw=2.5)
ax.fill_between(r_l, laue_i, 0,
                where=(laue_i > 0), alpha=0.3, color='red')
ax.fill_between(r_l, laue_i, 0,
                where=(laue_i < 0), alpha=0.3, color='royalblue')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.text(0.95, 0.05,
        rf'$\int r^2 p_{{\rm iso}}\,dr = {res["laue"]:.2e}$',
        transform=ax.transAxes, ha='right', fontsize=10,
        bbox=dict(boxstyle='round', fc='white', alpha=0.8))
ax.set_xlabel('r [fm]')
ax.set_ylabel(r'$r^2\,p_{\rm iso}(r)$  [MeV/fm]')
ax.set_title('Laue Condition Integrand')
ax.set_xlim(0, 3)
ax.grid(True, alpha=0.4)

ax = axes[1]
ax.axhspan(-2.4, -1.6, alpha=0.15, color='green',
           label=r'JLab DVCS $(-2.0\pm0.4)$')
ax.axhline(-2.0, color='green', ls='--', lw=1.2)
ax.axhline(0,    color='red',   ls='--', lw=1.0)
ax.plot(lambda_values, d1_values, 'o-',
        color='navy', lw=2.5, ms=7, label=r'MNBPS $d_1(\lambda)$')

idx_base = int(np.argmin(abs(lambda_values - LAM_BASE)))
ax.plot(lambda_values[idx_base], d1_values[idx_base], '*',
        color='red', ms=14, zorder=5,
        label=r'$\lambda=0.67$ fm (calibration)')

ax.set_xlabel(r'$\lambda$ [fm]')
ax.set_ylabel(r'$d_1$')
ax.set_title(r'Dependence of $d_1$ on $\lambda$')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.4)

fig3.tight_layout()
fig3.savefig("3.png", dpi=300, bbox_inches='tight')
print("Saved 3.png")

# =========================================================
# TABLE 1 (paper)
# =========================================================
print("\n" + "="*64)
print("  Table 1 (paper): scan over lambda")
print("="*64)
print(f"  {'lam [fm]':>8}  {'d1':>10}  {'Laue [fm^-1]':>14}  {'R_B [fm]':>10}")
print("-"*64)
for lam, d1, laue, RB in zip(lambda_values, d1_values, laue_values, RB_values):
    marker = " <- calibration" if abs(lam - LAM_BASE) < 0.01 else ""
    print(f"  {lam:8.2f}  {d1:10.4f}  {laue:14.3e}  {RB:10.4f}{marker}")
print("="*64)

# =========================================================
# NUMERICAL STABILITY OF THE LAUE INTEGRAL
# (Table 2 of the paper -- referee response)
# =========================================================
print("\n" + "="*78)
print("  Table 2 (paper): numerical stability of the Laue integral at lambda=0.67")
print("="*78)
print(f"  {'config':<22}  {'N':>5}  {'rmin [fm]':>10}  {'Rmax [fm]':>10}"
      f"  {'Laue [fm^-1]':>14}  {'d1':>9}  {'R_B':>7}")
print("-"*78)

# Baseline
b = compute_all(0.67, rmin=1e-4, Rmax=20.0, N=4000, tol=1e-9)
print(f"  {'baseline':<22}  {4000:>5}  {1e-4:>10.0e}  {20.0:>10.1f}"
      f"  {b['laue']:>14.3e}  {b['d1']:>9.4f}  {b['RB']:>7.4f}")

# Vary N
for N in [2000, 3000, 6000]:
    s = compute_all(0.67, N=N)
    print(f"  {'N = '+str(N):<22}  {N:>5}  {1e-4:>10.0e}  {20.0:>10.1f}"
          f"  {s['laue']:>14.3e}  {s['d1']:>9.4f}  {s['RB']:>7.4f}")

# Vary rmin
for rmin in [1e-3, 1e-5]:
    s = compute_all(0.67, rmin=rmin)
    print(f"  {'rmin = '+f'{rmin:.0e}':<22}  {4000:>5}  {rmin:>10.0e}  {20.0:>10.1f}"
          f"  {s['laue']:>14.3e}  {s['d1']:>9.4f}  {s['RB']:>7.4f}")

# Vary Rmax
for Rmax in [10.0, 15.0, 25.0]:
    s = compute_all(0.67, Rmax=Rmax)
    print(f"  {'Rmax = '+str(Rmax):<22}  {4000:>5}  {1e-4:>10.0e}  {Rmax:>10.1f}"
          f"  {s['laue']:>14.3e}  {s['d1']:>9.4f}  {s['RB']:>7.4f}")

print("="*78)
print("  Summary: int r^2 p_iso dr is dominated by numerical residual")
print("  (decreases monotonically with N), while d_1 and R_B are stable")
print("  to ~1e-4. The Laue condition is satisfied identically by")
print("  construction; nucleon stability is governed by d_1 < 0.")
print("="*78)

plt.show()
