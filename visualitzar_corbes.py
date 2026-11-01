"""
Visualització de les corbes diàries (eòlica i preu) de cada font de dades.

Per a cada carpeta (dades_esios/, dades_energycharts/) genera una figura amb:
  - totes les corbes diàries (una línia per dia, colorejada per mes)
  - la corba mitjana en negre

Ús:
    python visualitzar_corbes.py
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

# Carpetes a visualitzar i nom de la font
CARPETES = {
    "dades_esios": "ESIOS (REE)",
    "dades_energycharts": "Energy-Charts (Fraunhofer ISE)",
}


def carregar(carpeta):
    """Llegeix el fitxer .npz i retorna matrius, malles (en hores) i dates."""
    d = np.load(os.path.join(carpeta, "corbes_mfpca.npz"))
    # ESIOS té una malla per variable; Energy-Charts en té una de comuna
    t_eol = d["t_eolica"] if "t_eolica" in d.files else d["t"]
    t_pre = d["t_preu"] if "t_preu" in d.files else d["t"]
    dies = pd.to_datetime(d["dies"])
    return d["X_eolica"], t_eol * 24, d["X_preu"], t_pre * 24, dies


def dibuixar_panell(ax, t, X, dies, colors, titol, unitat):
    """Dibuixa totes les corbes del panell i la mitjana."""
    for fila, color in zip(X, colors):
        ax.plot(t, fila, color=color, lw=0.5, alpha=0.35)
    ax.plot(t, np.nanmean(X, axis=0), color="black", lw=2.2, label="Mitjana")
    ax.set_title(f"{titol}  ({X.shape[0]} dies, {X.shape[1]} punts/dia)")
    ax.set_xlabel("Hora del dia (hora peninsular)")
    ax.set_ylabel(unitat)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")


def figura_font(carpeta, nom_font):
    """Genera i desa la figura d'una font de dades."""
    X_eol, t_eol, X_pre, t_pre, dies = carregar(carpeta)

    # Color segons la data (escala contínua al llarg del període)
    num = mdates.date2num(dies)
    norm = Normalize(vmin=num.min(), vmax=num.max())
    cmap = cm.viridis
    colors = cmap(norm(num))

    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    dibuixar_panell(axs[0], t_eol, X_eol / 1000, dies, colors, "Generació eòlica", "GW")
    dibuixar_panell(axs[1], t_pre, X_pre, dies, colors, "Preu mercat diari", "€/MWh")
    axs[1].axhline(0, color="grey", lw=0.8)

    # Barra de color amb les dates
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axs, shrink=0.9, pad=0.01)
    cb.ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
    cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    inici, fi = dies.min().strftime("%d/%m/%Y"), dies.max().strftime("%d/%m/%Y")
    fig.suptitle(f"{nom_font}: corbes diàries {inici} – {fi}", fontsize=14)

    sortida = os.path.join(carpeta, "corbes_diaries.png")
    fig.savefig(sortida, dpi=150)
    print(f"Figura desada: {sortida}")
    return fig


def main():
    trobades = 0
    for carpeta, nom in CARPETES.items():
        if os.path.exists(os.path.join(carpeta, "corbes_mfpca.npz")):
            figura_font(carpeta, nom)
            trobades += 1
        else:
            print(f"No s'ha trobat {carpeta}/corbes_mfpca.npz: s'omet")
    if trobades:
        plt.show()


if __name__ == "__main__":
    main()
