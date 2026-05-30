# -*- coding: utf-8 -*-
# =========================================================
#  MNBPS-нуклон: механическая структура и моменты давления
#  Версия v17-clean (синхронизирована со статьей)
#
#  Что считает скрипт:
#    * краевую задачу для профиля f(r)  (solve_bvp);
#    * барионное число B и классическую массу M_cl;
#    * радиальное p_r, тангенциальное p_t, изотропное p_iso давления;
#    * условие Лауэ (второй момент);
#    * радиальный эффективный момент d_r^eff (через p_r);
#    * канонический D-член d1^can (через p_iso, Polyakov–Cebulla-нормировка);
#    * барионный радиус R_B;
#    * скан по lambda_tilde и все рисунки статьи (RU + EN, vector PDF/EPS).
#
#  ВАЖНО про обозначения и размерности:
#    В ковариантном лагранжиане L6 = -lambda^2 B_mu B^mu константа связи
#    lambda имеет размерность ДЛИНЫ (fm). В редуцированном одномерном
#    функционале E[f] удобно работать с ЭФФЕКТИВНЫМ параметром
#    lambda_tilde (lt) размерности fm^2, входящим в M(r,f) с весом 1/r^2.
#    Калибруется и сканируется именно lambda_tilde [fm^2]; в коде это
#    переменная `lt`. Не путать с lambda [fm] из лагранжиана.
#
#  ВАЖНО про моменты:
#    * d_r^eff -- радиальный эффективный момент (через p_r). Это та величина,
#      которая в v15 ошибочно называлась "каноническим d1" и сравнивалась
#      с окном JLab. Сама по себе корректна, но НЕ есть канонический D-член.
#    * d1^can -- канонический D-член (через p_iso, полный гильбертовский ТЭИ).
#      В радиально-согласованной редукции выполняется тождество
#          p_iso = p_r + (r/3) p_r'   =>   int r^4 p_iso dr = -(2/3) int r^4 p_r dr,
#      откуда d1^can = (25/6) d_r^eff. При lt = 0.67 fm^2 это -2.194 -> -9.14.
#      Прямое сопоставление d_r^eff с окном JLab снято.
# =========================================================

import numpy as np
from scipy.integrate import solve_bvp, trapezoid
import matplotlib.pyplot as plt

# ---- векторный вывод: текст как TrueType (Cyrillic-safe), без растеризации ----
plt.rcParams.update({
    'pdf.fonttype':   42,
    'ps.fonttype':    42,
    'font.family':    'DejaVu Sans',   # имеет кириллицу -> корректный RU vector PDF
    'font.size':      12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'svg.fonttype':   'none',
})

# =========================================================
# ФИЗИЧЕСКИЕ КОНСТАНТЫ
# =========================================================
fpi   = 93.0        # MeV
mpi   = 138.0       # MeV
hbarc = 197.327     # MeV*fm
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
# ФУНКЦИИ МОДЕЛИ   (lt = lambda_tilde, [fm^2])
# =========================================================
def M_func(r, f, lt):
    return r**2 + lt**2 * np.sin(f)**4 / r**2

def U_func(r, f):
    return np.sin(f)**2 + mu**2 * r**2 * (1.0 - np.cos(f))

def ode(r, y, lt):
    f, fp = y[0], y[1]
    s, c  = np.sin(f), np.cos(f)
    M     = M_func(r, f, lt)
    dMdr  = 2*r - 2*lt**2 * s**4 / r**3
    dMdf  = 4*lt**2 * s**3 * c / r**2
    dUdf  = 2*s*c + mu**2 * r**2 * s
    fpp   = (dUdf - dMdr*fp - 0.5*dMdf*fp**2) / M
    return np.vstack((fp, fpp))

def bc(ya, yb):
    return np.array([ya[0] - np.pi, yb[0]])

y_guess    = np.zeros((2, N))
y_guess[0] = np.pi * np.exp(-r)
y_guess[1] = -np.pi * np.exp(-r)

# =========================================================
# РАСЧеТ ВСЕХ ВЕЛИЧИН
# =========================================================
def compute_all(lt, y_init=None):
    if y_init is None:
        y_init = y_guess

    sol = solve_bvp(lambda rr, yy: ode(rr, yy, lt), bc,
                    r, y_init, tol=1e-8, max_nodes=100000)
    if not sol.success:
        print(f"  [WARN] lt={lt:.3f} --- solve_bvp не сошелся полностью")

    f  = sol.sol(r)[0]
    fp = sol.sol(r)[1]
    M  = M_func(r, f, lt)
    U  = U_func(r, f)

    # ---- барионная плотность B0 и барионное число B ----
    # sin f / r  с регуляризацией: (pi-f)/r * sinc((pi-f)/pi) == sin f / r
    ratio = (np.pi - f) / r * np.sinc((np.pi - f) / np.pi)
    B0    = -ratio**2 * fp / (2 * np.pi**2)
    Bnum  = 4*np.pi * trapezoid(r**2 * B0, r)

    # ---- классическая масса M_cl [MeV] (функционал энергии, ур. (5)) ----
    energy_integrand = 0.5 * M * fp**2 + U
    Mcl = 4*np.pi * fpi**2 / (2*hbarc) * trapezoid(energy_integrand, r)

    # ---- барионный радиус R_B ----
    RB_sq = trapezoid(r**4 * B0, r) / trapezoid(r**2 * B0, r)
    RB    = np.sqrt(abs(RB_sq))

    # ---- радиальное давление [MeV/fm^3] ----
    pr = (fpi**2 / (2.0 * hbarc)) * (0.5 * M * fp**2 - U) / r**2

    # ---- тангенциальное давление через равновесие ----
    ln_r  = np.log(r)
    r2pr  = r**2 * pr
    dr2pr = np.gradient(r2pr, ln_r) / r
    pt    = dr2pr / (2.0 * r)

    # обрезаем сингулярность у r=0
    cutoff_idx = np.searchsorted(r, 0.01)
    pt[:cutoff_idx] = np.nan
    pr[:cutoff_idx] = np.nan

    p_iso   = (pr + 2*pt) / 3.0
    s_shear = pt - pr

    mask = r >= 0.01
    laue = trapezoid(r[mask]**2 * p_iso[mask], r[mask])

    # ---- радиальный эффективный момент (через p_r) ----
    d_r_eff = -(4*np.pi * M_N) / (5 * hbarc**2) * trapezoid(
        r[mask]**4 * pr[mask], r[mask])

    # ---- канонический D-член (через p_iso, Polyakov–Cebulla) ----
    #   d1^can = 5 pi M_N/(hbarc)^2 * int r^4 p_iso dr
    #   тождественно = (25/6) d_r^eff в данной редукции
    d1_canon = 5 * np.pi * M_N / hbarc**2 * trapezoid(
        r[mask]**4 * p_iso[mask], r[mask])

    return {
        'f': f, 'fp': fp, 'pr': pr, 'pt': pt,
        'p_iso': p_iso, 's_shear': s_shear,
        'laue': laue, 'd_r_eff': d_r_eff, 'd1_canon': d1_canon,
        'RB': RB, 'B': Bnum, 'Mcl': Mcl,
        'y_sol': np.vstack((f, fp))
    }

# =========================================================
# БАЗОВОЕ РЕШЕНИЕ  lambda_tilde = 0.67 fm^2
#   (калибровка: целевое R_B = 0.8548 fm)
# =========================================================
LT_BASE = 0.67

print(f"Считаем базовое решение lambda_tilde = {LT_BASE} fm^2 ...")
res = compute_all(LT_BASE)
print(f"  B         = {res['B']:.12f}")
print(f"  M_cl      = {res['Mcl']:.3f} MeV")
print(f"  R_B       = {res['RB']:.4f} fm   (цель: 0.8548 fm)")
print(f"  d_r^eff   = {res['d_r_eff']:.5f}   (радиальный момент, через p_r)")
print(f"  d1_canon  = {res['d1_canon']:.5f}   (канонический D-член, через p_iso)")
print(f"  ratio d1_canon/d_r^eff = {res['d1_canon']/res['d_r_eff']:.5f}  "
      f"(теория: 25/6 = {25/6:.5f})")
print(f"  Laue      = {res['laue']:.3e} fm^-1")

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
# СКАН ПО lambda_tilde
# =========================================================
lt_values = np.array([0.20, 0.30, 0.40, 0.50, 0.60,
                      0.67, 0.83, 1.00, 1.17, 1.33, 1.50])
dreff_values, d1canon_values, laue_values, RB_values = [], [], [], []

print("\nСкан по lambda_tilde:")
current_guess = y_guess.copy()
for lt in lt_values:
    rl = compute_all(lt, current_guess)
    dreff_values.append(rl['d_r_eff'])
    d1canon_values.append(rl['d1_canon'])
    laue_values.append(rl['laue'])
    RB_values.append(rl['RB'])
    current_guess = rl['y_sol']
    print(f"  lt={lt:.2f}  d_r^eff={rl['d_r_eff']:9.5f}  "
          f"d1_canon={rl['d1_canon']:9.5f}  "
          f"Laue={rl['laue']:10.3e}  R_B={rl['RB']:.4f} fm")

dreff_values   = np.array(dreff_values)
d1canon_values = np.array(d1canon_values)
laue_values    = np.array(laue_values)
RB_values      = np.array(RB_values)

# =========================================================
# ПОДПИСИ: русская и английская версии
# =========================================================
L = {
    'ru': {
        'r':        'r [фм]',
        'frad':     'f(r) [рад]',
        'prof_t':   'Профильная функция f(r)',
        'piso':     r'$p_{\rm iso}$ [МэВ/фм$^3$]',
        'piso_t':   'Изотропное давление',
        'piso_lbl': r'$p_{\rm iso}(r)$',
        'rep':      'Отталкивание',
        'att':      'Притяжение',
        'press':    r'Давление [МэВ/фм$^3$]',
        'press_t':  'Компоненты давления внутри нуклона',
        'shear':    r'$s(r)$ [МэВ/фм$^3$]',
        'shear_t':  'Распределение сдвиговых сил',
        'shear_l':  r'$s(r) = p_t - p_r$',
        'laue_y':   r'$r^2\,p_{\rm iso}(r)$  [МэВ/фм]',
        'laue_t':   'Интегранд условия Лауэ',
        'lt':       r'$\tilde\lambda$ [фм$^2$]',
        'mom':      'механический момент',
        'dvs_t':    r'$d_r^{\rm eff}$ и канонический $d_1$ от $\tilde\lambda$',
        'jlab':     r'JLab DVCS, канон. $d_1$ ($-2.0\pm0.4$)',
        'dreff_l':  r'$d_r^{\rm eff}(\tilde\lambda)$ (радиальный момент)',
        'd1_l':     r'$d_1^{\rm can}(\tilde\lambda)$ (через $p_{\rm iso}$)',
        'star':     r'$\tilde\lambda=0.67$ фм$^2$ (калибровка)',
        'laue_box': r'$\int r^2 p_{\rm iso}\,dr = %.2e$',
    },
    'en': {
        'r':        'r [fm]',
        'frad':     'f(r) [rad]',
        'prof_t':   'Chiral Profile Function',
        'piso':     r'$p_{\rm iso}$ [MeV/fm$^3$]',
        'piso_t':   'Mechanical Pressure Distribution',
        'piso_lbl': r'$p_{\rm iso}(r)$',
        'rep':      'Repulsion',
        'att':      'Attraction',
        'press':    r'Pressure [MeV/fm$^3$]',
        'press_t':  'Pressure Distribution Inside the Nucleon',
        'shear':    r'$s(r)$ [MeV/fm$^3$]',
        'shear_t':  'Shear Force Distribution',
        'shear_l':  r'$s(r) = p_t - p_r$',
        'laue_y':   r'$r^2\,p_{\rm iso}(r)$  [MeV/fm]',
        'laue_t':   'Laue Condition Integrand',
        'lt':       r'$\tilde\lambda$ [fm$^2$]',
        'mom':      'mechanical moment',
        'dvs_t':    r'$d_r^{\rm eff}$ and canonical $d_1$ vs $\tilde\lambda$',
        'jlab':     r'JLab DVCS, canon. $d_1$ ($-2.0\pm0.4$)',
        'dreff_l':  r'$d_r^{\rm eff}(\tilde\lambda)$ (radial moment)',
        'd1_l':     r'$d_1^{\rm can}(\tilde\lambda)$ ($p_{\rm iso}$)',
        'star':     r'$\tilde\lambda=0.67$ fm$^2$ (calibration)',
        'laue_box': r'$\int r^2 p_{\rm iso}\,dr = %.2e$',
    },
}

def save(fig, stem, lang):
    suffix = '' if lang == 'ru' else '_en'
    for ext in ('pdf', 'eps'):
        fig.savefig(f"{stem}{suffix}.{ext}", bbox_inches='tight')
    print(f"  Saved {stem}{suffix}.pdf / .eps")

# =========================================================
# ПОСТРОЕНИЕ ВСЕХ РИСУНКОВ ДЛЯ ОДНОГО ЯЗЫКА
# =========================================================
def make_figures(lang):
    t = L[lang]
    print(f"\nРисунки [{lang}]:")

    # ---- РИС. 1: профиль + p_iso ----
    fig1, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(r_p, f_p, 'b-', lw=2.5)
    ax[0].set_xlabel(t['r']); ax[0].set_ylabel(t['frad'])
    ax[0].set_title(t['prof_t'])
    ax[0].set_xlim(0, 3); ax[0].set_ylim(0, np.pi + 0.1)
    ax[0].axhline(np.pi, color='gray', ls=':', lw=1, alpha=0.5)
    ax[0].text(0.05, np.pi + 0.05, r'$\pi$', fontsize=11, color='gray')
    ax[0].grid(True, alpha=0.4)

    ax[1].plot(r_p, piso_p, 'k-', lw=2.5, label=t['piso_lbl'])
    ax[1].fill_between(r_p, piso_p, 0, where=(piso_p > 0),
                       alpha=0.35, color='red', label=t['rep'])
    ax[1].fill_between(r_p, piso_p, 0, where=(piso_p < 0),
                       alpha=0.35, color='royalblue', label=t['att'])
    ax[1].axhline(0, color='gray', ls='--', lw=0.8)
    ax[1].set_xlabel(t['r']); ax[1].set_ylabel(t['piso'])
    ax[1].set_title(t['piso_t']); ax[1].set_xlim(0, 3)
    ax[1].legend(loc='upper right'); ax[1].grid(True, alpha=0.4)
    fig1.tight_layout(); save(fig1, '1', lang); plt.close(fig1)

    # ---- РИС. 2: p_r, p_t, p_iso + shear ----
    fig2, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(r_p, pr_p,   'b-',  lw=2.0, label=r'$p_r(r)$')
    ax[0].plot(r_p, pt_p,   'r--', lw=2.0, label=r'$p_t(r)$')
    ax[0].plot(r_p, piso_p, 'k:',  lw=1.8, label=r'$p_{\rm iso}(r)$')
    ax[0].axhline(0, color='gray', ls='--', lw=0.8)
    ax[0].set_xlabel(t['r']); ax[0].set_ylabel(t['press'])
    ax[0].set_title(t['press_t']); ax[0].set_xlim(0, 3)
    p_all = np.concatenate([pr_p, pt_p, piso_p]); p_all = p_all[np.isfinite(p_all)]
    ax[0].set_ylim(np.nanpercentile(p_all, 1) * 1.2,
                   np.nanpercentile(p_all, 99) * 1.4)
    ax[0].legend(loc='upper right'); ax[0].grid(True, alpha=0.4)

    ax[1].plot(r_p, s_p, 'g-', lw=2.5, label=t['shear_l'])
    ax[1].axhline(0, color='gray', ls='--', lw=0.8)
    ax[1].set_xlabel(t['r']); ax[1].set_ylabel(t['shear'])
    ax[1].set_title(t['shear_t']); ax[1].set_xlim(0, 3)
    s_fin = s_p[np.isfinite(s_p)]
    ax[1].set_ylim(np.nanpercentile(s_fin, 1) * 1.3,
                   np.nanpercentile(s_fin, 99) * 2.0)
    ax[1].legend(loc='upper right'); ax[1].grid(True, alpha=0.4)
    fig2.tight_layout(); save(fig2, '2', lang); plt.close(fig2)

    # ---- РИС. 3: Лауэ-интегранд + d_r^eff и канонический d1 ----
    fig3, ax = plt.subplots(1, 2, figsize=(13, 5))
    mlp  = (r >= 0.05) & (r <= 3.0)
    r_l  = r[mlp]
    li   = r_l**2 * p_iso[mlp]
    ax[0].plot(r_l, li, 'b-', lw=2.5)
    ax[0].fill_between(r_l, li, 0, where=(li > 0), alpha=0.3, color='red')
    ax[0].fill_between(r_l, li, 0, where=(li < 0), alpha=0.3, color='royalblue')
    ax[0].axhline(0, color='gray', ls='--', lw=0.8)
    ax[0].text(0.95, 0.05, t['laue_box'] % res['laue'],
               transform=ax[0].transAxes, ha='right', fontsize=10,
               bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax[0].set_xlabel(t['r']); ax[0].set_ylabel(t['laue_y'])
    ax[0].set_title(t['laue_t']); ax[0].set_xlim(0, 3)
    ax[0].grid(True, alpha=0.4)

    # полоса JLab относится к КАНОНИЧЕСКОМУ d1
    ax[1].axhspan(-2.4, -1.6, alpha=0.15, color='green', label=t['jlab'])
    ax[1].axhline(0, color='red', ls='--', lw=0.8)
    ax[1].plot(lt_values, dreff_values, 'o-', color='navy',
               lw=2.2, ms=6, label=t['dreff_l'])
    ax[1].plot(lt_values, d1canon_values, 's--', color='darkred',
               lw=2.2, ms=6, label=t['d1_l'])
    ib = np.argmin(abs(lt_values - LT_BASE))
    ax[1].plot(lt_values[ib], dreff_values[ib], '*', color='red',
               ms=15, zorder=5, label=t['star'])
    ax[1].set_xlabel(t['lt']); ax[1].set_ylabel(t['mom'])
    ax[1].set_title(t['dvs_t']); ax[1].set_ylim(-27, 1.5)
    ax[1].legend(loc='lower left', fontsize=9); ax[1].grid(True, alpha=0.4)
    fig3.tight_layout(); save(fig3, '3', lang); plt.close(fig3)

make_figures('ru')
make_figures('en')

# =========================================================
# ИТОГОВАЯ ТАБЛИЦА
# =========================================================
print("\n" + "="*78)
print("  Таблица для статьи (lambda_tilde в fm^2)")
print("="*78)
print(f"  {'lt[fm^2]':>8}  {'d_r^eff':>10}  {'d1_can':>10}  "
      f"{'Laue[fm^-1]':>13}  {'R_B[fm]':>8}")
print("-"*78)
for lt, dre, dca, lau, rb in zip(lt_values, dreff_values,
                                 d1canon_values, laue_values, RB_values):
    mark = "  <- калибровка" if abs(lt - LT_BASE) < 0.01 else ""
    print(f"  {lt:8.2f}  {dre:10.5f}  {dca:10.5f}  {lau:13.3e}  {rb:8.4f}{mark}")
print("="*78)
print("d_r^eff -- радиальный момент (через p_r); d1_can -- канонический D-член")
print("(через p_iso) = (25/6) d_r^eff. Прямое сопоставление d_r^eff с JLab снято.")
