import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# извлекаем решение
f = sol.sol(r)[0]
fp = sol.sol(r)[1]

# функции M и U (как в основном коде)
M = M_func(r, f, lam)
U = U_func(r, f)

# радиальное давление
pr = (fpi**2/2)*(0.5*M*(fp**2) - U)/(r**2)

# тангенциальное давление
pt = (1/(2*r))*np.gradient(r**2 * pr, r)

# изотропное давление
p_iso = (pr + 2*pt)/3

# -------------------------
# Проверка условия Лауэ
# -------------------------

laue_integrand = r**2 * p_iso

Laue = simpson(laue_integrand, r)

print("\nLaue condition:")
print("Integral ∫ r² p(r) dr =", Laue)

# -------------------------
# График
# -------------------------

plt.figure(figsize=(7,5))

plt.plot(r, laue_integrand, linewidth=2)

plt.axhline(0, linestyle='--')

plt.xscale('log')

plt.xlabel("r [fm]")
plt.ylabel(r"$r^2 p(r)$")

plt.title("Laue condition integrand")

plt.grid(True)

plt.tight_layout()

plt.savefig("laue_integrand.png", dpi=300)

plt.show()