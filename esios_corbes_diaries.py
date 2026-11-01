"""
Descàrrega de dades d'ESIOS (REE) i construcció de corbes diàries per a MFPCA.

- Generació eòlica en temps real (resolució 5 min -> 288 punts/dia)
- Preu del mercat diari (resolució 15 min -> 96 punts/dia)

Ús:
    1) Crea un fitxer .env a la mateixa carpeta que el script amb la línia:
           ESIOS_TOKEN=el_teu_token
       (només cal fer-ho un cop; afegeix .env al .gitignore si fas servir git)
    2) Comprova els indicadors:   python esios_corbes_diaries.py --cerca
    3) Descarrega i construeix:   python esios_corbes_diaries.py
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# Carrega el token des del fitxer .env (si python-dotenv està instal·lat)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuració
# ---------------------------------------------------------------------------
TOKEN = os.environ.get("ESIOS_TOKEN", "")
BASE_URL = "https://api.esios.ree.es"

DATA_INICI = "2025-10-01"
DATA_FI = datetime.today().strftime("%Y-%m-%d")
ZONA = "Europe/Madrid"  # el dia es defineix en hora local

# IDs dels indicadors (verificar amb l'opció --cerca)
ID_EOLICA = 551   # "Generació T.Real eòlica" (comprovar)
ID_PREU = 600     # Preu mercat diari (spot)

# Zona geogràfica preferida per a cada indicador (s'agafa la primera que existeixi)
GEO_EOLICA = ["Península", "España"]
GEO_PREU = ["España"]

# Resolució esperada (minuts) i punts per dia
RES_EOLICA, PUNTS_EOLICA = 5, 288
RES_PREU, PUNTS_PREU = 15, 96

# Granularitat explícita a la crida (time_trunc). Valors vàlids: "five_minutes",
# "ten_minutes", "fifteen_minutes", "hour", "day", "month", "year".
# time_agg = "average" perquè són potències (MW) i preus: per defecte l'API suma.
TRUNC_EOLICA = "five_minutes"
TRUNC_PREU = "fifteen_minutes"
TIME_AGG = "average"

# Mida dels trams de descàrrega: els indicadors amb moltes zones geogràfiques
# donen error 504 si es demanen més de 21 dies de cop
DIES_PER_TRAM = 21

CARPETA_SORTIDA = "dades_esios"
CARPETA_CACHE = os.path.join(CARPETA_SORTIDA, "cache")
PAUSA_ENTRE_PETICIONS = 2   # segons
MAX_REINTENTS = 6

HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": TOKEN,
}


# ---------------------------------------------------------------------------
# Funcions d'accés a l'API
# ---------------------------------------------------------------------------
def peticio(url, params, fitxer_cache=None):
    """GET amb memòria cau a disc i reintents (429 / 5xx)."""
    if fitxer_cache and os.path.exists(fitxer_cache):
        with open(fitxer_cache, encoding="utf-8") as f:
            return json.load(f)

    espera = 20
    for intent in range(1, MAX_REINTENTS + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=180)
            if r.status_code in (401, 403):
                raise SystemExit(f"Error {r.status_code}: token no vàlid o sense permís.")
            if r.status_code == 429 or r.status_code >= 500:
                print(f"    error {r.status_code}: espero {espera} s (reintent {intent}/{MAX_REINTENTS})")
                time.sleep(espera)
                espera *= 2
                continue
            r.raise_for_status()
            js = r.json()
            if fitxer_cache:
                with open(fitxer_cache, "w", encoding="utf-8") as f:
                    json.dump(js, f)
            time.sleep(PAUSA_ENTRE_PETICIONS)
            return js
        except requests.RequestException as e:
            print(f"    error ({e}): espero {espera} s (reintent {intent}/{MAX_REINTENTS})")
            time.sleep(espera)
            espera *= 2
    raise RuntimeError(f"No s'ha pogut descarregar {url} {params}")


def buscar_indicadors(text):
    """Cerca indicadors pel nom i retorna un DataFrame amb id i nom."""
    js = peticio(f"{BASE_URL}/indicators", {"text": text})
    df = pd.DataFrame(js["indicators"])
    return df[["id", "name"]]


def descarregar_indicador(id_ind, inici, fi, time_trunc):
    """Descarrega un indicador en trams de DIES_PER_TRAM dies, amb la granularitat indicada."""
    os.makedirs(CARPETA_CACHE, exist_ok=True)
    limits = list(pd.date_range(inici, fi, freq=f"{DIES_PER_TRAM}D"))
    if limits[-1] < pd.Timestamp(fi):
        limits.append(pd.Timestamp(fi))
    trossos = []
    for a, b in zip(limits[:-1], limits[1:]):
        params = {
            "start_date": a.strftime("%Y-%m-%dT00:00:00"),
            "end_date": (b - pd.Timedelta(days=1)).strftime("%Y-%m-%dT23:59:59"),
            "time_trunc": time_trunc,
            "time_agg": TIME_AGG,
        }
        # El nom de la memòria cau inclou el time_trunc per no barrejar granularitats
        cache = os.path.join(CARPETA_CACHE, f"ind{id_ind}_{time_trunc}_{a:%Y%m%d}_{b:%Y%m%d}.json")
        js = peticio(f"{BASE_URL}/indicators/{id_ind}", params, cache)
        if not trossos:
            print(f"  indicador {id_ind}: «{js['indicator'].get('name', '?')}»")
        valors = js["indicator"]["values"]
        trossos.append(pd.DataFrame(valors))
        print(f"  {a.date()} -> {(b - pd.Timedelta(days=1)).date()}  ({len(valors)} valors)")
    df = pd.concat(trossos, ignore_index=True)
    return df.drop_duplicates(subset=["datetime_utc", "geo_id"])


def filtrar_geo(df, preferencies):
    """Es queda amb una sola zona geogràfica segons l'ordre de preferència."""
    disponibles = df["geo_name"].unique().tolist()
    for g in preferencies:
        if g in disponibles:
            print(f"  zona seleccionada: {g}  (disponibles: {disponibles})")
            return df[df["geo_name"] == g].copy()
    raise ValueError(f"Cap zona de {preferencies} disponible. Opcions: {disponibles}")


# ---------------------------------------------------------------------------
# Construcció de corbes diàries
# ---------------------------------------------------------------------------
def a_corbes_diaries(df, resolucio, punts_dia):
    """
    Converteix la sèrie temporal en una matriu (dies x punts del dia).
    Només es conserven els dies complets (s'exclouen els dies de canvi d'hora
    i els dies amb forats).
    """
    t = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_convert(ZONA)
    df = df.assign(dia=t.dt.date,
                   minut=t.dt.hour * 60 + t.dt.minute,
                   value=pd.to_numeric(df["value"], errors="coerce"))

    # Comprovem que la resolució és l'esperada
    res_obs = int(t.drop_duplicates().sort_values().diff().median().total_seconds() // 60)
    print(f"  resolució observada: {res_obs} min (esperada {resolucio} min)")
    if res_obs != resolucio:
        raise ValueError(f"Resolució observada {res_obs} min diferent de l'esperada {resolucio} min")

    matriu = df.pivot_table(index="dia", columns="minut", values="value", aggfunc="mean")
    matriu = matriu.reindex(columns=np.arange(0, 24 * 60, resolucio))

    # Es compten les observacions reals: el dia de 25 h en té més i el pivot les promitjaria
    n_obs = df.dropna(subset=["value"]).groupby("dia").size().reindex(matriu.index, fill_value=0)
    complets = (matriu.notna().sum(axis=1) == punts_dia) & (n_obs == punts_dia)
    print(f"  dies totals: {len(matriu)}  |  complets: {complets.sum()}  |  "
          f"descartats: {(~complets).sum()}")
    return matriu[complets]


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    if not TOKEN:
        raise SystemExit("Falta el token: crea un fitxer .env amb la línia ESIOS_TOKEN=el_teu_token "
                         "(i instal·la python-dotenv) o defineix la variable d'entorn ESIOS_TOKEN")

    if "--cerca" in sys.argv:
        for text in ["eólica", "precio mercado spot"]:
            print(f"\nIndicadors amb «{text}»:")
            print(buscar_indicadors(text).to_string(index=False))
        return

    os.makedirs(CARPETA_SORTIDA, exist_ok=True)

    print("Eòlica:")
    eol = filtrar_geo(descarregar_indicador(ID_EOLICA, DATA_INICI, DATA_FI, TRUNC_EOLICA), GEO_EOLICA)
    eol.to_csv(f"{CARPETA_SORTIDA}/eolica_raw.csv", index=False)
    X_eol = a_corbes_diaries(eol, RES_EOLICA, PUNTS_EOLICA)

    print("Preu:")
    pre = filtrar_geo(descarregar_indicador(ID_PREU, DATA_INICI, DATA_FI, TRUNC_PREU), GEO_PREU)
    pre.to_csv(f"{CARPETA_SORTIDA}/preu_raw.csv", index=False)
    X_pre = a_corbes_diaries(pre, RES_PREU, PUNTS_PREU)

    # Dies comuns a les dues variables (necessari per a MFPCA)
    dies = X_eol.index.intersection(X_pre.index)
    X_eol, X_pre = X_eol.loc[dies], X_pre.loc[dies]
    print(f"\nDies comuns: {len(dies)}")

    X_eol.to_csv(f"{CARPETA_SORTIDA}/corbes_eolica.csv")
    X_pre.to_csv(f"{CARPETA_SORTIDA}/corbes_preu.csv")

    # Malles en [0, 1] per a FDApy / MFPCA
    t_eol = X_eol.columns.to_numpy() / (24 * 60)
    t_pre = X_pre.columns.to_numpy() / (24 * 60)
    np.savez(f"{CARPETA_SORTIDA}/corbes_mfpca.npz",
             X_eolica=X_eol.to_numpy(), t_eolica=t_eol,
             X_preu=X_pre.to_numpy(), t_preu=t_pre,
             dies=np.array([str(d) for d in dies]))
    print(f"Fitxers guardats a '{CARPETA_SORTIDA}/'")


if __name__ == "__main__":
    main()