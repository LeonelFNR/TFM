"""
Descàrrega de dades d'ESIOS (REE) i construcció de corbes diàries per a MFPCA.

- Generació eòlica en temps real (resolució 5 min -> 288 punts/dia)
- Preu del mercat diari (15 min -> 96 punts/dia; si el període comença abans
  de l'1-10-2025, horari -> 24 punts/dia, perquè abans el mercat era horari)

Ús:
    1) Crea un fitxer .env a la mateixa carpeta que el script amb la línia:
           ESIOS_TOKEN=el_teu_token
       (afegeix .env al .gitignore si fas servir git)
    2) Comprova els indicadors:   python esios_corbes_diaries.py --cerca
    3) Descarrega i construeix:   python esios_corbes_diaries.py --inici 2023-01-01 --fi 2025-12-31
       (sense --inici/--fi s'usen els valors per defecte)

Des d'un notebook:
    from esios_corbes_diaries import crear_dataset
    crear_dataset("2023-01-01", "2025-12-31")
"""

import argparse
import json
import os
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
DATA_FI = datetime.today().strftime("%Y-%m-%d")   # exclusiva: no s'inclou aquest dia
ZONA = "Europe/Madrid"  # el dia es defineix en hora local

# IDs dels indicadors (verificar amb l'opció --cerca)
ID_EOLICA = 551   # "Generació T.Real eòlica" (comprovar)
ID_PREU = 600     # Preu mercat diari (spot)

# Zona geogràfica preferida per a cada indicador (s'agafa la primera que existeixi)
GEO_EOLICA = ["Península", "España"]
GEO_PREU = ["España"]

# Granularitat explícita a la crida (time_trunc) i els seus minuts
MINUTS_TRUNC = {"five_minutes": 5, "ten_minutes": 10, "fifteen_minutes": 15, "hour": 60}
TRUNC_EOLICA = "five_minutes"
TRUNC_PREU = "fifteen_minutes"
TRUNC_PREU_HISTORIC = "hour"            # abans de l'1-10-2025 el mercat diari era horari
INICI_PREU_15MIN = pd.Timestamp("2025-10-01")
TIME_AGG = "average"   # potències (MW) i preus: per defecte l'API suma

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


def descarregar_indicador(id_ind, inici, fi, time_trunc, carpeta_cache=CARPETA_CACHE):
    """Descarrega un indicador en trams de DIES_PER_TRAM dies, amb la granularitat indicada."""
    os.makedirs(carpeta_cache, exist_ok=True)
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
        cache = os.path.join(carpeta_cache, f"ind{id_ind}_{time_trunc}_{a:%Y%m%d}_{b:%Y%m%d}.json")
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
def a_corbes_diaries(df, resolucio):
    """
    Converteix la sèrie temporal en una matriu (dies x punts del dia).
    Només es conserven els dies complets (s'exclouen els dies de canvi d'hora
    i els dies amb forats).
    """
    punts_dia = 24 * 60 // resolucio
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
# Funció principal (cridable des d'un notebook)
# ---------------------------------------------------------------------------
def crear_dataset(data_inici=DATA_INICI, data_fi=DATA_FI, carpeta=CARPETA_SORTIDA):
    """
    Descarrega eòlica i preu entre data_inici (inclosa) i data_fi (exclosa)
    i desa a 'carpeta' els fitxers raw, les corbes i el .npz.
    Retorna (X_eol, X_pre) amb els dies complets comuns.
    """
    if not TOKEN:
        raise SystemExit("Falta el token: crea un fitxer .env amb la línia ESIOS_TOKEN=el_teu_token "
                         "(i instal·la python-dotenv) o defineix la variable d'entorn ESIOS_TOKEN")
    if pd.Timestamp(data_inici) >= pd.Timestamp(data_fi):
        raise ValueError(f"data_inici ({data_inici}) ha de ser anterior a data_fi ({data_fi})")

    cache = os.path.join(carpeta, "cache")
    os.makedirs(carpeta, exist_ok=True)

    # Resolució del preu: horària si el període inclou dies anteriors a l'1-10-2025
    trunc_preu = TRUNC_PREU if pd.Timestamp(data_inici) >= INICI_PREU_15MIN else TRUNC_PREU_HISTORIC
    if trunc_preu != TRUNC_PREU:
        print(f"Avís: el període comença abans de {INICI_PREU_15MIN.date()}; "
              f"el preu es descarrega en resolució horària per tenir una sèrie homogènia.")

    print(f"Període: {data_inici} -> {data_fi} (exclòs)")
    print("Eòlica:")
    eol = filtrar_geo(descarregar_indicador(ID_EOLICA, data_inici, data_fi, TRUNC_EOLICA, cache),
                      GEO_EOLICA)
    eol.to_csv(f"{carpeta}/eolica_raw.csv", index=False)
    X_eol = a_corbes_diaries(eol, MINUTS_TRUNC[TRUNC_EOLICA])

    print("Preu:")
    pre = filtrar_geo(descarregar_indicador(ID_PREU, data_inici, data_fi, trunc_preu, cache),
                      GEO_PREU)
    pre.to_csv(f"{carpeta}/preu_raw.csv", index=False)
    X_pre = a_corbes_diaries(pre, MINUTS_TRUNC[trunc_preu])

    # Dies comuns a les dues variables (necessari per a MFPCA)
    dies = X_eol.index.intersection(X_pre.index)
    X_eol, X_pre = X_eol.loc[dies], X_pre.loc[dies]
    print(f"\nDies comuns: {len(dies)}")

    X_eol.to_csv(f"{carpeta}/corbes_eolica.csv")
    X_pre.to_csv(f"{carpeta}/corbes_preu.csv")

    # Malles en [0, 1] per a FDApy / MFPCA
    t_eol = X_eol.columns.to_numpy() / (24 * 60)
    t_pre = X_pre.columns.to_numpy() / (24 * 60)
    np.savez(f"{carpeta}/corbes_mfpca.npz",
             X_eolica=X_eol.to_numpy(), t_eolica=t_eol,
             X_preu=X_pre.to_numpy(), t_preu=t_pre,
             dies=np.array([str(d) for d in dies]))
    print(f"Fitxers guardats a '{carpeta}/'")
    return X_eol, X_pre


def main():
    parser = argparse.ArgumentParser(description="Descàrrega de dades d'ESIOS i corbes diàries")
    parser.add_argument("--inici", default=DATA_INICI, help="data d'inici, inclosa (AAAA-MM-DD)")
    parser.add_argument("--fi", default=DATA_FI, help="data de fi, exclosa (AAAA-MM-DD)")
    parser.add_argument("--carpeta", default=CARPETA_SORTIDA, help="carpeta de sortida")
    parser.add_argument("--cerca", action="store_true", help="només cerca els IDs dels indicadors")
    args = parser.parse_args()

    if args.cerca:
        if not TOKEN:
            raise SystemExit("Falta el token ESIOS_TOKEN (fitxer .env)")
        for text in ["eólica", "precio mercado spot"]:
            print(f"\nIndicadors amb «{text}»:")
            print(buscar_indicadors(text).to_string(index=False))
        return

    crear_dataset(args.inici, args.fi, args.carpeta)


if __name__ == "__main__":
    main()