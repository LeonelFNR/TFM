"""
Descàrrega de generació eòlica i preu del mercat diari d'Espanya
des de l'API pública d'Energy-Charts (Fraunhofer ISE), sense token.

Genera:
  dades_energycharts/serie_eolica_preu.csv   -> sèrie temporal (15 min)
  dades_energycharts/corbes_eolica.csv       -> matriu dies x 96 punts
  dades_energycharts/corbes_preu.csv         -> matriu dies x 96 punts
  dades_energycharts/corbes_mfpca.npz        -> matrius + malles a [0, 1]

Ús:
    python energycharts_eolica_preu.py
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuració
# ---------------------------------------------------------------------------
BASE_URL = "https://api.energy-charts.info"
PAIS = "es"            # país per a la generació
ZONA_PREU = "ES"       # zona d'oferta per al preu
DATA_INICI = "2025-10-01"
DATA_FI = datetime.today().strftime("%Y-%m-%d")
ZONA_HORARIA = "Europe/Madrid"
RESOLUCIO_MIN = 15
PUNTS_DIA = 24 * 60 // RESOLUCIO_MIN   # 96
CARPETA = "dades_energycharts"
CARPETA_CACHE = os.path.join(CARPETA, "cache")   # respostes ja descarregades
PAUSA_ENTRE_PETICIONS = 5   # segons entre peticions (evita l'error 429)
MAX_REINTENTS = 6


# ---------------------------------------------------------------------------
# Descàrrega
# ---------------------------------------------------------------------------
def peticio(endpoint, params):
    """
    Fa una petició GET amb memòria cau i reintents.
    - Si la resposta ja està desada a disc, la reutilitza (no torna a demanar-la).
    - Si l'API respon 429 (massa peticions), espera cada cop més temps.
    """
    os.makedirs(CARPETA_CACHE, exist_ok=True)
    nom = endpoint + "_" + "_".join(f"{k}-{v}" for k, v in params.items()) + ".json"
    fitxer_cache = os.path.join(CARPETA_CACHE, nom)
    if os.path.exists(fitxer_cache):
        with open(fitxer_cache, encoding="utf-8") as f:
            return json.load(f)

    espera = 30
    for intent in range(1, MAX_REINTENTS + 1):
        try:
            r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=120)
            if r.status_code == 429:
                # Si l'API indica quant esperar (Retry-After), en fem cas
                espera_api = r.headers.get("Retry-After")
                t = int(espera_api) if espera_api and espera_api.isdigit() else espera
                print(f"    429 massa peticions: espero {t} s (reintent {intent}/{MAX_REINTENTS})")
                time.sleep(t)
                espera *= 2
                continue
            r.raise_for_status()
            js = r.json()
            with open(fitxer_cache, "w", encoding="utf-8") as f:
                json.dump(js, f)
            time.sleep(PAUSA_ENTRE_PETICIONS)
            return js
        except requests.RequestException as e:
            print(f"    error ({e}), espero {espera} s (reintent {intent}/{MAX_REINTENTS})")
            time.sleep(espera)
            espera *= 2
    raise RuntimeError(f"No s'ha pogut descarregar {endpoint} {params}")


def trams_mensuals(inici, fi):
    """Divideix el període en trams mensuals [a, b)."""
    limits = pd.date_range(inici, fi, freq="MS").union([pd.Timestamp(inici), pd.Timestamp(fi)])
    return list(zip(limits[:-1], limits[1:]))


def descarregar_eolica(inici, fi):
    """Generació eòlica (onshore + offshore si n'hi ha) en MW."""
    trossos = []
    for a, b in trams_mensuals(inici, fi):
        js = peticio("public_power", {"country": PAIS,
                                      "start": a.strftime("%Y-%m-%d"),
                                      "end": b.strftime("%Y-%m-%d")})
        tipus = [p["name"] for p in js["production_types"]]
        eol = [p for p in js["production_types"] if "wind" in p["name"].lower()]
        if not eol:
            raise ValueError(f"No hi ha cap sèrie eòlica. Tipus disponibles: {tipus}")
        df = pd.DataFrame({"unix": js["unix_seconds"]})
        for p in eol:
            df[p["name"]] = pd.to_numeric(pd.Series(p["data"]), errors="coerce")
        trossos.append(df)
        print(f"  eòlica {a.date()} -> {b.date()}: {len(df)} valors  {[p['name'] for p in eol]}")
    df = pd.concat(trossos).drop_duplicates("unix").set_index("unix").sort_index()
    cols_eol = [c for c in df.columns if "wind" in c.lower()]
    df["eolica_mw"] = df[cols_eol].sum(axis=1, min_count=1)
    return df


def descarregar_preu(inici, fi):
    """Preu del mercat diari en EUR/MWh."""
    trossos = []
    for a, b in trams_mensuals(inici, fi):
        js = peticio("price", {"bzn": ZONA_PREU,
                               "start": a.strftime("%Y-%m-%d"),
                               "end": b.strftime("%Y-%m-%d")})
        df = pd.DataFrame({"unix": js["unix_seconds"],
                           "preu_eur_mwh": pd.to_numeric(pd.Series(js["price"]), errors="coerce")})
        trossos.append(df)
        print(f"  preu   {a.date()} -> {b.date()}: {len(df)} valors  ({js.get('unit', '')})")
    return pd.concat(trossos).drop_duplicates("unix").set_index("unix").sort_index()


# ---------------------------------------------------------------------------
# Construcció del dataset
# ---------------------------------------------------------------------------
def resolucio_minuts(index_unix):
    """Resolució mediana de la sèrie en minuts."""
    return int(np.median(np.diff(np.asarray(index_unix))) / 60)


def a_corbes(serie):
    """Sèrie (índex datetime local) -> matriu dies x 96, només dies complets."""
    df = pd.DataFrame({"valor": serie.values,
                       "dia": serie.index.date,
                       "minut": serie.index.hour * 60 + serie.index.minute})
    m = df.pivot_table(index="dia", columns="minut", values="valor", aggfunc="mean")
    m = m.reindex(columns=np.arange(0, 24 * 60, RESOLUCIO_MIN))
    # Es compten les observacions reals de cada dia: el dia de 25 h en té 100
    # i el pivot les promitjaria, així que cal descartar-lo explícitament
    n_obs = df.dropna(subset=["valor"]).groupby("dia").size().reindex(m.index, fill_value=0)
    complets = (m.notna().sum(axis=1) == PUNTS_DIA) & (n_obs == PUNTS_DIA)
    print(f"  dies: {len(m)}  complets: {complets.sum()}  descartats: {(~complets).sum()}")
    return m[complets]


def main():
    os.makedirs(CARPETA, exist_ok=True)

    print("Descarregant eòlica...")
    eol = descarregar_eolica(DATA_INICI, DATA_FI)
    print("Descarregant preu...")
    pre = descarregar_preu(DATA_INICI, DATA_FI)

    print(f"Resolució eòlica: {resolucio_minuts(eol.index)} min | "
          f"resolució preu: {resolucio_minuts(pre.index)} min")

    # Sèrie conjunta en hora local
    serie = eol[["eolica_mw"]].join(pre, how="outer")
    serie.index = pd.to_datetime(serie.index, unit="s", utc=True)
    serie["hora_local"] = serie.index.tz_convert(ZONA_HORARIA)
    serie = serie.rename_axis("hora_utc").reset_index()
    serie.to_csv(f"{CARPETA}/serie_eolica_preu.csv", index=False)
    print(f"Sèrie conjunta: {len(serie)} files, "
          f"NaN eòlica: {serie.eolica_mw.isna().sum()}, NaN preu: {serie.preu_eur_mwh.isna().sum()}")

    # Corbes diàries (s'exclouen dies de canvi d'hora i dies incomplets)
    s = serie.set_index("hora_local")
    print("Corbes eòlica:")
    X_eol = a_corbes(s["eolica_mw"])
    print("Corbes preu:")
    X_pre = a_corbes(s["preu_eur_mwh"])

    dies = X_eol.index.intersection(X_pre.index)
    X_eol, X_pre = X_eol.loc[dies], X_pre.loc[dies]
    print(f"Dies comuns: {len(dies)}")

    X_eol.to_csv(f"{CARPETA}/corbes_eolica.csv")
    X_pre.to_csv(f"{CARPETA}/corbes_preu.csv")

    t = X_eol.columns.to_numpy() / (24 * 60)   # malla a [0, 1]
    np.savez(f"{CARPETA}/corbes_mfpca.npz",
             X_eolica=X_eol.to_numpy(), X_preu=X_pre.to_numpy(),
             t=t, dies=np.array([str(d) for d in dies]))
    print(f"Fitxers guardats a '{CARPETA}/'")


if __name__ == "__main__":
    main()