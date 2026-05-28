"""
Taller CMB - Parte 1.3
Graficar combinaciones de armonicos esfericos y comparar
patrones dominados por multipolos bajos, altos y muy altos.
Autor: Samuel Hoyos
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import special

# --- Paleta tipo Planck (azul frio -> crema -> rojo caliente) ---
_anchors = [(0.00, (0.00, 0.00, 0.45)), (0.18, (0.00, 0.35, 0.95)),
            (0.38, (0.35, 0.80, 1.00)), (0.50, (1.00, 1.00, 0.88)),
            (0.62, (1.00, 0.82, 0.20)), (0.82, (1.00, 0.40, 0.05)),
            (1.00, (0.50, 0.00, 0.00))]
cmap = LinearSegmentedColormap.from_list(
    "planck_like", [(p, c) for p, c in _anchors], N=256)


def Ylm(l, m, pol, az):
    try:
        return special.sph_harm_y(l, m, pol, az)      # scipy >= 1.15
    except AttributeError:
        return special.sph_harm(m, l, az, pol)        # scipy < 1.15


nlon, nlat = 500, 250
lon = np.linspace(-np.pi, np.pi, nlon)
lat = np.linspace(-np.pi/2, np.pi/2, nlat)
LON, LAT = np.meshgrid(lon, lat)
POL2D = np.pi/2 - LAT        
AZ2D = LON + np.pi         

modos = [(1, 0), (2, 1), (4, 2), (8, 4)]
fig1, axes = plt.subplots(2, 2, figsize=(8, 5.2),
                          subplot_kw={'projection': 'mollweide'})
for ax, (l, m) in zip(axes.ravel(), modos):
    ax.pcolormesh(LON, LAT, Ylm(l, m, POL2D, AZ2D).real, cmap=cmap, shading='auto')
    ax.set_title(rf'$\ell={l},\ m={m}$', fontsize=11, pad=6)
    ax.set_xticklabels([]); ax.set_yticklabels([])
fig1.suptitle(r'Armonicos esfericos individuales  Re[$Y_{\ell m}$]',
              fontsize=12, y=0.99)
fig1.subplots_adjust(hspace=0.35, top=0.90)
fig1.savefig('armonicos_individuales.png', dpi=200, bbox_inches='tight')

pol_1d = np.pi/2 - lat        
az_1d = lon + np.pi           

def mapa_sintetico(l_min, l_max, seed=1):
    rng = np.random.default_rng(seed)
    campo = np.zeros((nlat, nlon))
    for l in range(l_min, l_max + 1):
        for m in range(0, l + 1):
           
            leg = Ylm(l, m, pol_1d, 0.0).real        
            if m == 0:
                campo += rng.normal() * leg[:, None]
            else:
                
                pref = np.sqrt(2.0) * (-1)**m
                campo += rng.normal() * pref * (leg[:, None] * np.cos(m*az_1d)[None, :])
                campo += rng.normal() * pref * (leg[:, None] * np.sin(m*az_1d)[None, :])
    return campo

bajo  = mapa_sintetico(2, 10,   seed=1)   # multipolos bajos
alto  = mapa_sintetico(30, 45,  seed=1)   # multipolos altos
muy   = mapa_sintetico(90, 110, seed=1)   # multipolos muy altos

fig2, axs = plt.subplots(1, 3, figsize=(13, 3.4),
                         subplot_kw={'projection': 'mollweide'})
data = [(bajo, r'Multipolos bajos ($2\leq\ell\leq10$)'),
        (alto, r'Multipolos altos ($30\leq\ell\leq45$)'),
        (muy,  r'Multipolos muy altos ($90\leq\ell\leq110$)')]
for ax, (campo, tit) in zip(axs, data):
    vmax = np.max(np.abs(campo))
    ax.pcolormesh(LON, LAT, campo, cmap=cmap, shading='auto', vmin=-vmax, vmax=vmax)
    ax.set_title(tit, fontsize=11)
    ax.set_xticklabels([]); ax.set_yticklabels([])
fig2.tight_layout()
fig2.savefig('comparacion_multipolos.png', dpi=200, bbox_inches='tight')

plt.show()
