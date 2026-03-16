import numpy as np
from scipy.integrate import solve_bvp, simpson
import matplotlib.pyplot as plt

# -----------------------------
# ФИЗИЧЕСКИЕ КОНСТАНТЫ
# -----------------------------
fpi = 93.0       # MeV
mpi = 138.0      # MeV
HBAR_C = 197.327
mu = mpi / HBAR_C

MN = 939.0       # масса нуклона (нормировка D)

# -----------------------------
# ЧИСЛЕННАЯ СЕТКА
# -----------------------------
rmin = 1e-4
Rmax = 40.0
N = 800

r = np.logspace(np.log10(rmin), np.log10(Rmax), N)

# -----------------------------
# РЕГУЛЯРИЗАЦИЯ sin(f)/r
# -----------------------------
def ratio(f, r):
    x = (np.pi - f)/np.pi
    return (np.pi - f)/r * np.sinc(x)

# -----------------------------
# ФУНКЦИИ M(r,f) И U(r,f)
# -----------------------------
def M_func(r, f, lam):
    return r**2 + lam**2 * (np.sin(f)**4)/(r**2)

def U_func(r, f):
    return np.sin(f)**2 + mu**2 * r**2 * (1 - np.cos(f))

# -----------------------------
# СИСТЕМА ODE
# -----------------------------
def ode(r, y, lam):

    f = y[0]
    fp = y[1]

    M = M_func(r, f, lam)
    U = U_func(r, f)

    dMdr = 2*r - 2*lam**2*(np.sin(f)**4)/(r**3)
    dMdf = lam**2*(4*np.sin(f)**3*np.cos(f))/(r**2)
    dUdf = 2*np.sin(f)*np.cos(f) + mu**2*r**2*np.sin(f)

    fpp = (dUdf - dMdr*fp - 0.5*dMdf*(fp**2))/M

    return np.vstack((fp, fpp))

# -----------------------------
# ГРАНИЧНЫЕ УСЛОВИЯ
# -----------------------------
def bc(ya, yb):
    return np.array([ya[0] - np.pi, yb[0]])

# -----------------------------
# НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ
# -----------------------------
y_guess = np.zeros((2, r.size))
y_guess[0] = np.pi * np.exp(-r)

# -----------------------------
# РАСЧЁТ D-TERM
# -----------------------------
def compute_D(lam):

    sol = solve_bvp(lambda r,y: ode(r,y,lam), bc, r, y_guess)

    f = sol.sol(r)[0]
    fp = sol.sol(r)[1]

    M = M_func(r, f, lam)
    U = U_func(r, f)

    pr = (fpi**2/2)*(0.5*M*(fp**2) - U)/(r**2)

    pt = (1/(2*r))*np.gradient(r**2 * pr, r)

    piso = (pr + 2*pt)/3

    D = -(4*np.pi*MN/3)*simpson(r**4 * piso, r)

    return D

# -----------------------------
# СКАН λ
# -----------------------------
lambda_values = np.linspace(0.5, 1.5, 7)

D_values = []

for lam in lambda_values:

    print("λ =", lam)

    D = compute_D(lam)

    D_values.append(D)

# -----------------------------
# ГРАФИК
# -----------------------------
plt.figure(figsize=(7,5))

plt.plot(lambda_values, D_values, 'o-', linewidth=2)

plt.axhline(0, color='red', linestyle='--')

plt.xlabel(r'$\lambda$', fontsize=14)
plt.ylabel('D-term', fontsize=14)

plt.title('Dependence of D-term on core parameter λ')

plt.grid(True)

plt.tight_layout()

plt.savefig("lambda_scan_Dterm.png", dpi=300)

plt.show()

# -----------------------------
# ВЫВОД
# -----------------------------
print("\nResults:")
for lam, D in zip(lambda_values, D_values):
    print(f"λ = {lam:.2f}   D = {D:.5e}")