"""
Taller CMB - Parte 2
Funciones de Neumann N_n(x) (funciones de Bessel de segunda especie),
para n = 0, 1, 2 en el intervalo 0.01 <= x <= 20.
Autor: Samuel Hoyos
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

x = np.linspace(0.01, 20, 2000)

N0 = special.yn(0, x)
N1 = special.yn(1, x)
N2 = special.yn(2, x)

env = np.sqrt(2.0 / (np.pi * x))

fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(x, N0, label=r'$N_0(x)$', lw=1.8)
ax.plot(x, N1, label=r'$N_1(x)$', lw=1.8)
ax.plot(x, N2, label=r'$N_2(x)$', lw=1.8)
ax.plot(x,  env, 'k--', lw=1.0, alpha=0.6, label=r'$\pm\sqrt{2/(\pi x)}$')
ax.plot(x, -env, 'k--', lw=1.0, alpha=0.6)

ax.axhline(0, color='gray', lw=0.6)
ax.set_xlim(0, 20)
ax.set_ylim(-1.1, 0.65)
ax.set_xlabel(r'$x$', fontsize=12)
ax.set_ylabel(r'$N_n(x)$', fontsize=12)
ax.set_title('Funciones de Neumann $N_n(x)$', fontsize=12)
ax.legend(loc='upper right', fontsize=10, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('neumann.png', dpi=200, bbox_inches='tight')
print("Figura de Neumann lista.")
# Valores cerca del origen para evidenciar la divergencia
for n, Nn in [(0, N0), (1, N1), (2, N2)]:
    print(f"N_{n}(0.01) = {Nn[0]:.3f}")
