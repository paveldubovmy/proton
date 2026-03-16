# ----------------------------
# Pressure и Shear forces
# ----------------------------

p = (pr + 2*pt)/3
s = pt - pr

# ----------------------------
# График давления
# ----------------------------

plt.figure(figsize=(7,5))

plt.plot(r, p, linewidth=2)

plt.axhline(0, linestyle='--')

plt.xscale('log')

plt.xlabel("r [fm]")
plt.ylabel("p(r)")

plt.title("Pressure distribution inside the nucleon")

plt.grid(True)

plt.tight_layout()

plt.savefig("pressure_distribution.png", dpi=300)

plt.show()


# ----------------------------
# График shear forces
# ----------------------------

plt.figure(figsize=(7,5))

plt.plot(r, s, linewidth=2)

plt.axhline(0, linestyle='--')

plt.xscale('log')

plt.xlabel("r [fm]")
plt.ylabel("s(r)")

plt.title("Shear force distribution")

plt.grid(True)

plt.tight_layout()

plt.savefig("shear_forces.png", dpi=300)

plt.show()