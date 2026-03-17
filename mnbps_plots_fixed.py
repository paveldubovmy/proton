import numpy as np
from scipy.integrate import solve_bvp, trapezoid
import matplotlib.pyplot as plt

# =========================================================
# ФИЗИЧЕСКИЕ КОНСТАНТЫ
# =========================================================
fpi   = 93.0        # MeV
mpi   = 138.0       # MeV
hbarc = 197.327     # MeV·fm
M_N   = 939.0       # MeV
mu    = mpi / hbarc # 1/fm

# =========================================================
# СЕТКА
# =========================================================
rmin = 1e-4
Rmax = 20.0
N    = 4000

r = np.logspace(np.log10(rmin), np.log10(Rmax), N)

# =========================================================
# ФУНКЦИИ МОДЕЛИ
# =========================================================
def M_func(r, f, lam):
    return r**2 + lam**2 * np.sin(f)**4 / r**2

def U_func(r, f):
    return np.sin(f)**2 + mu**2 * r**2 * (1.0 - np.cos(f))

def ode(r, y, lam):
    f,  fp = y[0], y[1]
    s,  c  = np.sin(f), np.cos(f)
    M      = M_func(r, f, lam)
    dMdr   =  2*r      - 2*lam**2 * s**4 / r**3
    dMdf   =  4*lam**2 * s**3 * c / r**2
    dUdf   =  2*s*c + mu**2 * r**2 * s
    fpp    = (dUdf - dMdr*fp - 0.5*dMdf*fp**2) / M
    return np.vstack((fp, fpp))

def bc(ya, yb):
    return np.array([ya[0] - np.pi, yb[0]])

y_guess      = np.zeros((2, N))
y_guess[0]   = np.pi * np.exp(-r)
y_guess[1]   = -np.pi * np.exp(-r)

# =========================================================
# РАСЧЁТ ВСЕХ МЕХАНИЧЕСКИХ ВЕЛИЧИН
# =========================================================
def compute_all(lam, y_init=None):
    if y_init is None:
        y_init = y_guess

    sol = solve_bvp(
        lambda rr, yy: ode(rr, yy, lam), bc,
        r, y_init, tol=1e-9, max_nodes=200000
    )
    if not sol.success:
        print(f"  [WARN] lambda={lam:.3f} --- solve_bvp did not fully converge")

    f  = sol.sol(r)[0]
    fp = sol.sol(r)[1]

    M = M_func(r, f, lam)
    U = U_func(r, f)

    # Радиальное давление [MeV/fm^3]
    pr = (fpi**2 / (2.0 * hbarc)) * (0.5 * M * fp**2 - U) / r**2

    # Тангенциальное давление через равновесие
    ln_r   = np.log(r)
    r2pr   = r**2 * pr
    dr2pr  = np.gradient(r2pr, ln_r) / r
    pt     = dr2pr / (2.0 * r)

    # Обрезаем сингулярность у r=0
    cutoff_idx = np.searchsorted(r, 0.01)
    pt[:cutoff_idx] = np.nan
    pr[:cutoff_idx] = np.nan

    p_iso   = (pr + 2*pt) / 3.0
    s_shear =  pt - pr

    mask_laue = r >= 0.01
    laue = trapezoid(r[mask_laue]**2 * p_iso[mask_laue], r[mask_laue])

    d1 = -(4*np.pi * M_N) / (5 * hbarc**2) * trapezoid(
        r[mask_laue]**4 * pr[mask_laue], r[mask_laue]
    )

    ratio = (np.pi - f) / r * np.sinc((np.pi - f) / np.pi)
    B0    = -ratio**2 * fp / (2 * np.pi**2)
    RB_sq = trapezoid(r**4 * B0, r) / trapezoid(r**2 * B0, r)
    RB    = np.sqrt(abs(RB_sq))

    return {
        'f': f, 'fp': fp, 'pr': pr, 'pt': pt,
        'p_iso': p_iso, 's_shear': s_shear,
        'laue': laue, 'd1': d1, 'RB': RB,
        'y_sol': np.vstack((f, fp))
    }

# =========================================================
# БАЗОВОЕ РЕШЕНИЕ  lambda = 0.67  (КАЛИБРОВКА: R_B = 0.8548 fm)
# =========================================================
LAM_BASE = 0.67   # <-- ИСПРАВЛЕНО: было 0.83

print(f"Считаем базовое решение lambda = {LAM_BASE} ...")
res = compute_all(LAM_BASE)
print(f"  R_B  = {res['RB']:.4f} fm   (цель: 0.8548 fm)")
print(f"  d1   = {res['d1']:.5f}")
print(f"  Laue = {res['laue']:.3e} fm^-1")

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
# СТИЛЬ
# =========================================================
plt.rcParams.update({
    'font.size':      12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize':11,
    'figure.dpi':     150,
})

# =========================================================
# РИС. 1: Профиль f(r) и изотропное давление p_iso(r)
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
# РИС. 2: pr, pt, p_iso + сдвиговые силы
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
# СКАН ПО lambda
# =========================================================
lambda_values = np.array([0.50, 0.67, 0.83, 1.00, 1.17, 1.33, 1.50])
d1_values, laue_values = [], []

print("\nScan over lambda:")
current_guess = y_guess.copy()

for lam in lambda_values:
    res_lam = compute_all(lam, current_guess)
    d1_values.append(res_lam['d1'])
    laue_values.append(res_lam['laue'])
    current_guess = res_lam['y_sol']
    print(f"  lam={lam:.2f}  d1={res_lam['d1']:9.5f}  "
          f"Laue={res_lam['laue']:10.3e}  R_B={res_lam['RB']:.4f} fm")

d1_values   = np.array(d1_values)
laue_values = np.array(laue_values)

# =========================================================
# РИС. 3: Лауэ-интегранд + d1(lambda)
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

# ИСПРАВЛЕНО: калибровочная точка lambda = 0.67, было 0.83
idx_base = np.argmin(abs(lambda_values - LAM_BASE))
ax.plot(lambda_values[idx_base], d1_values[idx_base], '*',
        color='red', ms=14, zorder=5,
        label=r'$\lambda=0.67$ fm (calibration)')   # ИСПРАВЛЕНО

ax.set_xlabel(r'$\lambda$ [fm]')
ax.set_ylabel(r'$d_1$')
ax.set_title(r'Dependence of $d_1$ on $\lambda$')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.4)

fig3.tight_layout()
fig3.savefig("3.png", dpi=300, bbox_inches='tight')
print("Saved 3.png")

# =========================================================
# ИТОГОВАЯ ТАБЛИЦА
# =========================================================
print("\n" + "="*60)
print("  Table for paper")
print("="*60)
print(f"  {'lam [fm]':>8}  {'d1':>10}  {'Laue [fm^-1]':>14}  {'R_B [fm]':>10}")
print("-"*60)
for lam, d1, laue in zip(lambda_values, d1_values, laue_values):
    marker = " <- calibration" if abs(lam - LAM_BASE) < 0.01 else ""
    print(f"  {lam:8.2f}  {d1:10.5f}  {laue:14.3e}{marker}")
print("="*60)

plt.show()
