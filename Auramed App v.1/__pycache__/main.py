"""
╔══════════════════════════════════════════════════════════════════════════╗
║  AuraMed — main.py  (v2)                                                 ║
║  Single Page Application · NiceGUI · Assistente Medico AI Lombardia      ║
║                                                                          ║
║  Avvio:  python main.py                                                  ║
║  URL:    http://localhost:8080                                           ║
║                                                                          ║
║  Dipendenze:  pip install nicegui numpy requests openai httpx            ║
║  Variabile opzionale:  OPENAI_API_KEY=sk-proj-...                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


# ════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import requests


import asyncio
import json
import math
import os
import random
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
load_dotenv()

import geopandas as gpd
import numpy as np
import requests as _requests
from nicegui import ui, app as _nicegui_app



# ════════════════════════════════════════════════════════════════
# SEZIONE 1 — CONFIGURAZIONE & DATI
# ════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent

_GDF_LOMBARDIA = gpd.read_file(BASE_DIR / "lombardia.geojson").dissolve()
_LOMBARDIA_GEOJSON_STR = (BASE_DIR / "lombardia.geojson").read_text(encoding="utf-8")

def _random_point_lombardia() -> dict:
    pt = _GDF_LOMBARDIA.sample_points(1).iloc[0]
    return {"lat": pt.y, "lon": pt.x}

# ── Database ospedali ─────────────────────────────────────────
with open(BASE_DIR / "ospedali_docs.json", encoding="utf-8") as _f:
    OSPEDALI_DB: list[dict] = json.load(_f)

# ── Posizione utente di default ───────────────────────────────
POSIZIONE_UTENTE: dict = _random_point_lombardia()

# ── Logo SVG ──────────────────────────────────────────────────
# Logo_White.svg per sfondo chiaro (usato nell'UI).
# Logo_Dark.svg disponibile nella stessa cartella per varianti dark.
_LOGO_SVG: str = (BASE_DIR / "Logo_White.svg").read_text(encoding="utf-8")

# ════════════════════════════════════════════════════════════════
# SEZIONE 2 — stima_tempo.ipynb
# Calcolo tempo di percorrenza reale via TOMTOM
# ════════════════════════════════════════════════════════════════
#
# Replica e ottimizza la logica di stima_tempo.ipynb:
#   - Una sola chiamata API per tutti i 100 ospedali (batch)
#   - Fallback automatico a stima haversine se TOMTOM non risponde
#
# TOMTOM Table API: /table/v1/driving/{coordinates}
#   coordinates: lon,lat;lon,lat;...  (utente prima, poi ospedali)
#   sources=0           → solo l'utente come origine
#   destinations=1;2;.. → tutti gli ospedali come destinazioni
#   response["durations"][0][k] → secondi da utente a ospedale k
# ─────────────────────────────────────────────────────────────────

BASE_URL = "https://api.tomtom.com/routing/1/calculateRoute"

DEFAULT_TIMEOUT = 10  # secondi per richiesta HTTP


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distanza in km — usata come fallback se TOMTOM non risponde."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)


def _fallback_minuti(user_pos: dict, osp: dict) -> float:
    """Stima tempo in minuti via haversine (30 km/h urbana, min 5 min)."""
    c = osp.get("Coordinate", {})
    d = _haversine_km(user_pos["lat"], user_pos["lon"], c.get("lat", 0), c.get("long", 0))
    return float(max(5, round((d / 30.0) * 60, 1)))


def stima_tempi_batch(user_pos: dict, ospedali: list[dict]) -> list[float]:
    """
    ╔─────────────────────────────────────────────────────────────╗
    ║  WRAPPER — stima_tempo.ipynb                                ║
    ║                                                             ║
    ║  Una sola chiamata TomTom con tutti gli ospedali in batch.  ║
    ║  Ritorna lista di tempi in minuti, uno per ospedale,        ║
    ║  nello stesso ordine della lista in ingresso.               ║
    ║                                                             ║
    ║  Fallback automatico a stima haversine (30 km/h) se:        ║
    ║    - TomTom non è raggiungibile                             ║
    ║    - L'ospedale non è raggiungibile via strada (None)       ║
    ╚─────────────────────────────────────────────────────────────╝
    """
    # ── Costruzione stringa coordinate ──────────────────────────
    # Formato TomTom: "lat,lon;lat,lon;..."
    # Indice 0 = utente, indici 1..N = ospedali
    coords_parts: list[str] = [f"{user_pos['lon']},{user_pos['lat']}"]
    for osp in ospedali:
        c = osp.get("Coordinate", {})
        lat = c.get("lat", 0.0)
        lon = c.get("long", 0.0)
        coords_parts.append(f"{lat},{lon}")

    coord_string = ";".join(coords_parts)
    n = len(ospedali)
    destinations = ";".join(str(i) for i in range(1, n + 1))

    url = f"{BASE_URL}/{coord_string}"

    try:
        resp = _requests.get(
            url,
            params={"sources": "0", "destinations": destinations},
            timeout=_TOMTOM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "Ok":
            raise ValueError(f"TOMTOM code: {data.get('code')}")

        # durations[0] → array di secondi (None se non raggiungibile)
        durations_sec: list = data["durations"][0]
        return [
            round(d / 60.0, 1) if d is not None else _fallback_minuti(user_pos, osp)
            for d, osp in zip(durations_sec, ospedali)
        ]

    except Exception as exc:
        print(f"[AuraMed] TOMTOM non disponibile ({exc}) — fallback haversine")
        return [_fallback_minuti(user_pos, osp) for osp in ospedali]




# ════════════════════════════════════════════════════════════════
# SEZIONE 3 — ChatGPT.ipynb
# Analisi sintomi via OpenAI GPT
# ════════════════════════════════════════════════════════════════

_CSV_FILE_ID: str | None = None

def _get_openai_client():
    try:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        return OpenAI(api_key=key)
    except ImportError:
        return None


def _upload_csv_once(client) -> str:
    global _CSV_FILE_ID
    if _CSV_FILE_ID:
        return _CSV_FILE_ID
    f = client.files.create(
        file=open(BASE_DIR / "aree_specialistiche_lombardia_Chat.csv", "rb"),
        purpose="user_data",
    )
    _CSV_FILE_ID = f.id
    return _CSV_FILE_ID


def analizza_sintomi(testo: str) -> dict:
    """
    ╔─────────────────────────────────────────────────────────────╗
    ║  WRAPPER — ChatGPT.ipynb                                    ║
    ║  Input : testo libero con i sintomi                         ║
    ║  Output: {"isCritical": "Y"|"N", "reparto": "NOME"}        ║
    ║                                                             ║
    ║  Usa OpenAI GPT-4.1-mini se OPENAI_API_KEY è disponibile,  ║
    ║  altrimenti fallback keyword-based locale.                  ║
    ╚─────────────────────────────────────────────────────────────╝
    """
    client = _get_openai_client()
    if not client:
        return _mock_analisi(testo)

    try:
        file_id = _upload_csv_once(client)
        prompt = (
            f'Agisci come un operatore medico esperto di triage. Sulla base di questa affermazione "{testo}" definisci se il soggetto '
            f'è in una situazione critica di pericolo di morte e quale sia il reparto '
            f'ospedaliero più appropriato. '
            f'I nomi dei reparti devono essere presi dal file csv allegato, che contiene '
            f'una lista di reparti ospedalieri in Lombardia.\n\n'
            f'Restituisci unicamente un json valido senza markdown e senza testo '
            f'aggiuntivo, formattato in questo modo: '
            f'{{"isCritical": "Y", "reparto": "OCULISTICA"}}'
        )
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_file", "file_id": file_id},
                ],
            }],
        )
        return json.loads(response.output_text)
    except Exception as exc:
        print(f"[AuraMed] OpenAI error: {exc} — fallback mock")
        return _mock_analisi(testo)


def _mock_analisi(testo: str) -> dict:
    """Fallback senza API key. Stessa struttura output di ChatGPT.ipynb."""
    tl = testo.lower()
    _CRITICI = [
        "petto", "braccio sinistro", "manca il respiro", "infarto", "ictus",
        "convulsioni", "svenuto", "emorragia", "overdose", "forte dolore addominale",
        "addome rigido", "non respiro", "difficoltà respiratoria", "paralisi",
        "trauma cranico", "anafilassi", "gola che chiude", "shock",
    ]
    _REPARTI: list[tuple] = [
        (("petto", "cuore", "infarto", "cardiaco", "pressione", "aritmia", "angina"),
         "CARDIOLOGIA"),
        (("bambino", "figlio", "figlia", "neonato", "bimb", "lattante", "anni ha"),
         "PEDIATRIA"),
        (("testa", "emicrania", "ictus", "neurologico", "vista offuscata", "tremore"),
         "NEUROLOGIA"),
        (("osso", "frattura", "distorsione", "ginocchio", "caviglia", "polso"),
         "ORTOPEDIA E TRAUMATOLOGIA"),
        (("allergia", "allergico", "orticaria", "gonfiore", "prurito"),
         "ALLERGOLOGIA"),
        (("appendicite", "addome", "pancia", "intestin", "peritonite"),
         "CHIRURGIA GENERALE"),
        (("febbre", "infezione", "influenza", "polmonite", "covid"),
         "MEDICINA INTERNA - GENERALE"),
        (("respiro", "polmone", "asma", "bronchite"),
         "PNEUMOLOGIA"),
        (("occhio", "vista", "oculist"),
         "OCULISTICA"),
    ]
    is_critical = any(k in tl for k in _CRITICI)
    reparto, best = "PRONTO SOCCORSO", 0
    for kws, rep in _REPARTI:
        s = sum(1 for k in kws if k in tl)
        if s > best:
            best, reparto = s, rep
    return {"isCritical": "Y" if is_critical else "N", "reparto": reparto}





# ════════════════════════════════════════════════════════════════
# SEZIONE 4 — CodiciPS.ipynb
# Snapshot affollamento PS (5 codici triage)
# ════════════════════════════════════════════════════════════════

def snapshot_ps(n_pazienti_min,n_pazienti_max, score_max) -> dict:
    """
    ╔─────────────────────────────────────────────────────────────╗
    ║  DA CodiciPS.ipynb — cell 1 & 2                             ║
    ║  HUB   → n_pazienti_min=10, n_pazienti_max=80, score_max=180║
    ║  Altri → n_pazienti_min=5, n_pazienti_max=40, score_max=100 ║
    ╚─────────────────────────────────────────────────────────────╝
    """
    codici = ["Rosso", "Arancione", "Azzurro", "Verde", "Bianco"]
    prob    = [0.10, 0.15, 0.20, 0.40, 0.15]
    n = random.randint(n_pazienti_min, n_pazienti_max)
    cnts = np.random.multinomial(n, prob)
    snap = dict(zip(codici, [int(x) for x in cnts]))
    pesi = {"Rosso": 20, "Arancione": 10, "Azzurro": 5, "Verde": 2, "Bianco": 1}
    score = sum(snap[c] * pesi[c] for c in snap)
    return {
        "codici": snap,
        "n_pazienti": n,
        "affollamento": round(min(1.2, score / score_max), 2),
    }


def score_ps(snapshot: dict, tempo_min: float, reparto_match: bool) -> float:
    """
    ╔─────────────────────────────────────────────────────────────╗
    ║  DA CodiciPS.ipynb — cell 3                                 ║
    ║  ω1=0.65·T  ω2=0.25·A  ω3=0.1·R   (t_max=90 min)            ║
    ╚─────────────────────────────────────────────────────────────╝
    """
    T = max(-1, 1.0 - (tempo_min / 90.0))
    A = 1.0 - snapshot["affollamento"]
    R = 1.0 if reparto_match else 0.0
    return round(0.65 * T + 0.25 * A + 0.1 * R, 4)


# ════════════════════════════════════════════════════════════════
# SEZIONE 4b — StruttureReparti.ipynb
# Query grafo Neo4j: filtra ospedali e valorizza isReparto
# ════════════════════════════════════════════════════════════════

NEO4J_URI      = "neo4j+ssc://e50dcde9.databases.neo4j.io"
NEO4J_USER     = "e50dcde9"
NEO4J_PASSWORD = "5Wq7pNsDXym3-uMbo0Ug0BwK_dkTmko-B0NdiDJAg0k"
NEO4J_DATABASE = "e50dcde9"

try:
    from neo4j import GraphDatabase as _GraphDatabase
    _neo4j_driver = _GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    _neo4j_driver.verify_connectivity()
    with _neo4j_driver.session(database=NEO4J_DATABASE) as _s:
        _s.run("CREATE INDEX reparto_nome IF NOT EXISTS FOR (r:Reparto) ON (r.Nome)")
        _s.run("CREATE INDEX ospedale_nome IF NOT EXISTS FOR (o:Ospedale) ON (o.Nome)")
        _s.run("CREATE INDEX ospedale_codice IF NOT EXISTS FOR (o:Ospedale) ON (o.Codice)")
    print("[AuraMed] Neo4j connesso (indici verificati).")
except Exception as _neo4j_err:
    print(f"[AuraMed] Neo4j non disponibile: {_neo4j_err} — fallback locale")
    _neo4j_driver = None


def strutture_con_reparti(reparto: str) -> dict:
    """
    ╔─────────────────────────────────────────────────────────────╗
    ║  DA StruttureReparti.ipynb                                  ║
    ║  Critico  → tutti gli ospedali, isReparto = None            ║
    ║  Non crit → solo ospedali con PS;                           ║
    ║             isReparto = 1 se hanno anche il reparto, 0 no   ║
    ║  Returns: {codice: isReparto}   (vuoto se Neo4j offline)    ║
    ╚─────────────────────────────────────────────────────────────╝
    """
    if _neo4j_driver is None:
        return {}
    cypher = """
    MATCH (o:Ospedale)-[:HA_REPARTO]->(rep:Reparto)
    WHERE rep.Nome = "PRONTO SOCCORSO"
    RETURN o.Codice AS codice,
    CASE
        WHEN EXISTS {
            MATCH (o)-[:HA_REPARTO]->(rep2:Reparto)
            WHERE rep2.Nome = $reparto
        }
        THEN 1
        ELSE 0
    END AS isReparto"""
    params: dict = {"reparto": reparto}
    try:
        with _neo4j_driver.session(database=NEO4J_DATABASE) as s:
            return {r["codice"]: r["isReparto"] for r in s.run(cypher, **params)}
    except Exception as exc:
        print(f"[AuraMed] Errore query Neo4j: {exc} — fallback locale")
        return {}




# ════════════════════════════════════════════════════════════════
# SEZIONE 5 — PIPELINE PRINCIPALE
# ════════════════════════════════════════════════════════════════

def _is_hub(classif: str) -> bool:
    return "HUB" in classif.upper()


def google_maps_url(osp: dict) -> str:
    q = f"{osp['Nome']}, {osp['Indirizzo']}, {osp['CAP']} {osp['Città']}"
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(q)}"


def cerca_strutture(
    testo: str,
    posizione: dict = POSIZIONE_UTENTE,
    top_n: int = 6,
) -> list[dict]:
    """
    Pipeline completa:
      1. analizza_sintomi()     → isCritical (backend-only), reparto
      2. stima_tempi_batch()    → tempi reali TOMTOM per tutti gli ospedali
      3. snapshot_ps()          → affollamento per ciascuno
      4. score_ps()             → punteggio composito
      5. Ordinamento e top_n

    isCritical è usato SOLO per cambiare il criterio di ordinamento:
      - Critico     → ordine per tempo crescente (ogni secondo conta)
      - Non critico → score composito ω1·T + ω2·A + ω3·R

    Il risultato dell'analisi critico/non critico NON viene esposto all'UI.
    """
    # Step 1 — analisi LLM (backend only)
    analisi = analizza_sintomi(testo)
    reparto_cercato = analisi.get("reparto", "PRONTO SOCCORSO").upper().strip()
    is_critical     = analisi.get("isCritical", "N") == "Y"

    # Step 1b — filtraggio grafo Neo4j → {codice: isReparto}
    grafo = strutture_con_reparti(reparto_cercato)
    if grafo:
        ospedali_lista = [o for o in OSPEDALI_DB if o.get("Codice") in grafo]
    else:
        ospedali_lista = OSPEDALI_DB  # fallback se Neo4j non disponibile

    # Step 2 — tempi TOMTOM in batch (una sola chiamata API)
    tempi_minuti: list[float] = stima_tempi_batch(posizione, ospedali_lista)

    # Step 3 & 4 — arricchimento e scoring
    risultati: list[dict] = []
    for osp, tempo_min in zip(ospedali_lista, tempi_minuti):
        hub  = _is_hub(osp.get("Classificazione", ""))
        snap = snapshot_ps(10 if hub else 5, 80 if hub else 40, 200 if hub else 150)

        is_reparto    = grafo.get(osp.get("Codice"))   # 1, 0, o None (critico/fallback)
        reparto_match = is_reparto == 1

        # Display: grafo ha priorità, poi check locale su Aree_specialistiche
        if reparto_match:
            reparto_disp = reparto_cercato
        else:
            reparto_disp = "PRONTO SOCCORSO"

        if is_critical:
            punteggio = -tempo_min           # minor tempo → più alto in classifica
        else:
            punteggio = score_ps(snap, tempo_min, reparto_match)

        c = osp.get("Coordinate", {})
        dist_km = _haversine_km(
            posizione["lat"], posizione["lon"],
            c.get("lat", 0), c.get("long", 0),
        )

        risultati.append({
            **osp,
            "_distanza_km":     dist_km,
            "_tempo_min":       tempo_min,
            "_snapshot":        snap,
            "_reparto_trovato": reparto_disp or "PRONTO SOCCORSO",
            "_punteggio":       punteggio,
            "_maps_url":        google_maps_url(osp),
        })

    risultati.sort(key=lambda x: x["_punteggio"], reverse=True)
    return risultati[:top_n], is_critical





# ════════════════════════════════════════════════════════════════
# SEZIONE 6 — NICEGUI UI
# ════════════════════════════════════════════════════════════════

# Palette brand (da Logo_White.svg)
_C_NAVY  = "#0A1628"
_C_TEAL  = "#00C2B5"
_C_TEAL2 = "#009E94"


def _affollamento_info(val: float) -> tuple[str, str, str]:
    if val < 0.35:
        return "Basso",  "#dcfce7", "#15803d"
    elif val < 0.70:
        return "Medio",  "#fef9c3", "#854d0e"
    else:
        return "Alto",   "#fee2e2", "#b91c1c"


@ui.page("/")
def pagina_principale() -> None:

    # ── Head ──────────────────────────────────────────────────
    ui.add_head_html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
          rel="stylesheet">
<style>
      body, .q-page { font-family:'Inter',sans-serif !important;
                      background:#f8fafc !important; }
      .am-card { transition: box-shadow .22s, transform .22s; }
      .am-card:hover { box-shadow:0 10px 32px rgba(0,0,0,.1) !important;
                       transform:translateY(-2px); }
      .am-link { color:#009E94; text-decoration:none; font-weight:600; font-size:.82rem; }
      .am-link:hover { text-decoration:underline; }
      .am-badge { display:inline-flex; align-items:center; gap:5px;
                  border-radius:20px; padding:4px 11px;
                  font-size:.76rem; font-weight:700; white-space:nowrap; }
    </style>
    """)

    # ── Root ──────────────────────────────────────────────────
    with ui.column().classes("w-full items-center px-4 py-8"):
        with ui.column().classes("w-full gap-0").style("max-width:720px"):

            # ══════════════════════════════════════════════════
            # 1. DISCLAIMER — non chiudibile
            # ══════════════════════════════════════════════════
            ui.html("""
            <div style="background:#fef2f2;border:1.5px solid #fca5a5;
                        border-radius:14px;padding:14px 18px;margin-bottom:24px;
                        display:flex;align-items:flex-start;gap:12px">
              <span style="font-size:1.3rem;flex-shrink:0;line-height:1.2">⚠️</span>
              <div>
                <p style="margin:0 0 4px;font-weight:700;color:#b91c1c;font-size:.88rem">
                  Attenzione — Supporto informativo
                </p>
                <p style="margin:0;color:#7f1d1d;font-size:.82rem;line-height:1.55">
                  AuraMed è un supporto informativo,
                  <strong>non è un servizio di emergenza</strong>.
                  In caso critico <strong>chiama il 112</strong> immediatamente.
                </p>
              </div>
            </div>
            """)

            # ══════════════════════════════════════════════════
            # 2. LOGO + TITOLO  (Logo_White.svg dalla cartella)
            # ══════════════════════════════════════════════════
            ui.html(f"""
            <div style="text-align:left;margin-bottom:28px">
              <div style="display:inline-block;margin-bottom:10px;
                          filter:drop-shadow(0 4px 10px rgba(0,194,181,.2))">
                {_LOGO_SVG}
              </div>
              <p style="margin:4px 0 0;font-size:.9rem;color:#64748b;
                        line-height:1.65;max-width:520px;margin:0 auto">
                Descrivi i tuoi sintomi in linguaggio naturale e trova
                le strutture ospedaliere più idonee nella tua area.
              </p>
            </div>
            """)

            # ══════════════════════════════════════════════════
            # 3. BOX RICERCA
            # ══════════════════════════════════════════════════
            with ui.element("div").classes("w-full").style(
                "background:white;border-radius:20px;padding:24px;"
                "border:1px solid #e2e8f0;"
                "box-shadow:0 2px 12px rgba(0,0,0,.05);margin-bottom:32px"
            ):
                ui.html(f"""
                <div style="display:flex;align-items:center;gap:9px;margin-bottom:14px">
                  <div style="width:5px;height:20px;border-radius:3px;
                              background:linear-gradient(180deg,{_C_TEAL},{_C_TEAL2})">
                  </div>
                  <span style="font-weight:700;color:{_C_NAVY};font-size:.95rem">
                    Descrivi i tuoi sintomi
                  </span>
                </div>
                """)

                textarea = (
                    ui.textarea(
                        placeholder=(
                            'Es: «Mio figlio di 3 anni ha febbre alta da ieri sera e '
                            'fa fatica a deglutire» oppure «Ho un forte dolore al petto '
                            'che si irradia al braccio sinistro»...'
                        )
                    )
                    .classes("w-full")
                    .props("outlined rows=4")
                    .style("font-size:.92rem")
                )

                with ui.row().classes(
                    "w-full items-center justify-between mt-4 flex-wrap gap-3"
                ):
                    ui.html(
                        '<span style="font-size:.73rem;color:#94a3b8">'
                        '🔒 I dati non vengono memorizzati &nbsp;·&nbsp; AI powered'
                        '</span>'
                    )
                    with ui.row().classes("items-center gap-3"):
                        spinner = ui.spinner("dots", size="sm", color="teal")
                        spinner.set_visibility(False)

                        cerca_btn = (
                            ui.button("🔍  Cerca Strutture Idonee")
                            .props("no-caps")
                            .style(
                                f"background:linear-gradient(135deg,{_C_TEAL},{_C_TEAL2});"
                                "color:white;font-weight:700;font-size:.88rem;"
                                "border-radius:10px;padding:10px 22px;"
                                "box-shadow:0 4px 14px rgba(0,194,181,.3);"
                                "border:none;cursor:pointer;"
                                "font-family:'Inter',sans-serif"
                            )
                        )

            # ══════════════════════════════════════════════════
            # 4. CONTAINER RISULTATI
            # ══════════════════════════════════════════════════
            results_col = ui.column().classes("w-full gap-4")

        # end centered column

    # ── Handler principale ─────────────────────────────────────
    async def on_cerca() -> None:
        testo = textarea.value.strip()
        if not testo:
            ui.notify(
                "⚠️  Inserisci i tuoi sintomi prima di cercare.",
                type="warning",
                position="top",
            )
            return

        cerca_btn.disable()
        spinner.set_visibility(True)
        results_col.clear()
        await asyncio.sleep(0.05)

        POSIZIONE_UTENTE.update(_random_point_lombardia())

        try:
            loop = asyncio.get_event_loop()
            ospedali, is_critical = await loop.run_in_executor(None, cerca_strutture, testo)
        except Exception as exc:
            ui.notify(f"Errore: {exc}", type="negative")
            cerca_btn.enable()
            spinner.set_visibility(False)
            return

        spinner.set_visibility(False)
        cerca_btn.enable()

        if not ospedali:
            with results_col:
                ui.label("Nessuna struttura trovata.").classes("text-slate-500 text-sm")
            return

        # ── Dati mappa calcolati prima del with block ──────────────
        u_lat      = POSIZIONE_UTENTE["lat"]
        u_lon      = POSIZIONE_UTENTE["lon"]
        markers_js = ""
        for h in ospedali:
            c   = h.get("Coordinate", {})
            lat = c.get("lat", 0.0)
            lon = c.get("long", 0.0)
            if lat and lon:
                nome = h.get("Nome", "").replace("'", "\\'")
                markers_js += f"L.marker([{lat},{lon}]).bindPopup('{nome}').addTo(m);\n"

        with results_col:
            if is_critical:
                ui.html("""
                <div style="background:#fff1f2;border:2px solid #f87171;border-radius:14px;
                            padding:16px 20px;display:flex;align-items:center;
                            justify-content:space-between;gap:16px;margin-bottom:4px">
                  <div style="display:flex;align-items:center;gap:10px">
                    <span style="font-size:1.4rem">🚨</span>
                    <div>
                      <div style="font-weight:700;color:#b91c1c;font-size:.95rem">
                        Situazione potenzialmente critica
                      </div>
                      <div style="color:#ef4444;font-size:.78rem;margin-top:2px">
                        Se sei in pericolo di vita chiama immediatamente il 112
                      </div>
                    </div>
                  </div>
                  <a href="tel:112"
                     style="display:inline-flex;align-items:center;gap:7px;
                            background:#dc2626;color:white;font-weight:700;
                            font-size:.9rem;padding:10px 20px;border-radius:10px;
                            text-decoration:none;white-space:nowrap;
                            box-shadow:0 4px 14px rgba(220,38,38,.4)">
                    📞 Chiama il 112
                  </a>
                </div>
                """)

            ui.html(f"""
            <div style="display:flex;align-items:center;gap:10px;
                        padding-bottom:4px;margin-bottom:4px">
              <div style="width:4px;height:20px;border-radius:2px;
                          background:linear-gradient(180deg,{_C_TEAL},{_C_TEAL2})">
              </div>
              <span style="font-size:1rem;font-weight:700;color:{_C_NAVY}">
                Strutture idonee trovate
              </span>
              <span style="font-size:.8rem;color:#94a3b8;margin-left:4px">
                — ordinate per idoneità clinica
              </span>
            </div>
            """)

            # ── Mappa: file HTML su disco, servito via /assets/ ───
            _map_fname = f"_map_{random.randint(10**9, 10**10)}.html"
            (BASE_DIR / _map_fname).write_text(f"""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{margin:0;padding:0;width:100%;height:100%}}</style>
</head>
<body>
<div id="map"></div>
<script>
var m = L.map("map");
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",
  {{attribution: "\\u00a9 OpenStreetMap contributors", maxZoom: 18}}).addTo(m);
var lombardiaLayer = L.geoJSON({_LOMBARDIA_GEOJSON_STR}, {{
  style: {{color:"#009E94", weight:2, fillColor:"#e0f7f5", fillOpacity:0.15}}
}}).addTo(m);
m.fitBounds(lombardiaLayer.getBounds());
L.circleMarker([{u_lat}, {u_lon}],
  {{radius:10, color:"#009E94", fillColor:"#00C2B5", fillOpacity:0.85, weight:2}})
  .bindPopup("La tua posizione").addTo(m);
{markers_js}
</script>
</body>
</html>
""", encoding="utf-8")
            (ui.element("iframe")
                .props(f'src="/assets/{_map_fname}"')
                .style(
                    "width:100%;height:320px;border:none;border-radius:14px;"
                    "box-shadow:0 2px 12px rgba(0,0,0,.05);margin-bottom:8px;"
                    "display:block"
                ))

            ui.html("""
            <div style="display:flex;gap:18px;font-size:.75rem;color:#64748b;
                        margin-bottom:12px;padding-left:4px">
              <span style="display:flex;align-items:center;gap:6px">
                <span style="display:inline-block;width:12px;height:12px;
                             border-radius:50%;background:#00C2B5;
                             border:2px solid #009E94"></span>
                La tua posizione
              </span>
              <span style="display:flex;align-items:center;gap:6px">
                <img
                src="https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png"
                style="height:20px;vertical-align:middle;margin-right:4px"
                >
                Strutture trovate
              </span>
            </div>
            """)

            for idx, h in enumerate(ospedali):
                _render_card(idx, h)

    cerca_btn.on_click(on_cerca)


# ── Componente card ospedale ────────────────────────────────────

def _render_card(idx: int, h: dict) -> None:
    snap         = h["_snapshot"]
    codici       = snap["codici"]
    affoll_txt, affoll_bg, affoll_fg = _affollamento_info(snap["affollamento"])
    pct          = round(snap["affollamento"] * 100)
    reparto_disp = h["_reparto_trovato"].title()
    nome         = h["Nome"].title()
    indirizzo    = h["Indirizzo"].title()
    citta        = h["Città"].title()

    with ui.element("div").classes("w-full am-card").style(
        "background:white;border-radius:18px;padding:22px 24px;"
        "border:1px solid #e2e8f0;box-shadow:0 2px 10px rgba(0,0,0,.05)"
    ):
        # ── Header: rank · nome · link mappa ─────────────────
        ui.html(f"""
        <div style="display:flex;align-items:flex-start;
                    justify-content:space-between;gap:12px;margin-bottom:14px">
          <div style="display:flex;align-items:center;gap:11px;flex:1;min-width:0">
            <div style="width:28px;height:28px;border-radius:8px;flex-shrink:0;
                        background:linear-gradient(135deg,{_C_TEAL},{_C_TEAL2});
                        display:flex;align-items:center;justify-content:center;
                        font-size:.8rem;font-weight:800;color:white">
              {idx + 1}
            </div>
            <div style="min-width:0">
              <div style="font-size:1rem;font-weight:700;color:{_C_NAVY};
                          line-height:1.3;overflow:hidden;text-overflow:ellipsis">
                {nome}
              </div>
              <div style="font-size:.72rem;color:#94a3b8;font-weight:500;margin-top:1px">
                {h.get('Classificazione','')} &nbsp;·&nbsp; {h.get('Ente','')}
              </div>
            </div>
          </div>
          <a href="{h['_maps_url']}" target="_blank" rel="noopener" class="am-link"
             style="display:flex;align-items:center;gap:5px;flex-shrink:0;
                    padding:6px 13px;border-radius:8px;
                    border:1.5px solid #b2ece7;background:#f0fdf9">
            📍 Mappa
          </a>
        </div>
        """)

        # ── Reparto badge ─────────────────────────────────────
        ui.html(f"""
        <div style="margin-bottom:14px">
          <span style="display:inline-flex;align-items:center;gap:6px;
                       background:#e0f7f5;color:{_C_TEAL2};
                       border:1.5px solid #b2ece7;border-radius:20px;
                       padding:4px 13px;font-size:.78rem;font-weight:700">
            🏥 {reparto_disp}
          </span>
        </div>
        """)

        # ── Metriche ──────────────────────────────────────────
        ui.html(f"""
        <div style="display:flex;gap:28px;flex-wrap:wrap;margin-bottom:14px">
          <div>
            <div style="font-size:.68rem;color:#94a3b8;text-transform:uppercase;
                        letter-spacing:.7px;font-weight:600;margin-bottom:3px">
              📏 Distanza
            </div>
            <div style="font-size:.92rem;font-weight:700;color:{_C_NAVY}">
              {h['_distanza_km']} km
            </div>
          </div>
          <div>
            <div style="font-size:.68rem;color:#94a3b8;text-transform:uppercase;
                        letter-spacing:.7px;font-weight:600;margin-bottom:3px">
              ⏱ Percorrenza
            </div>
            <div style="font-size:.92rem;font-weight:700;color:{_C_NAVY}">
              ~{h['_tempo_min']} min
            </div>
          </div>
          <div>
            <div style="font-size:.68rem;color:#94a3b8;text-transform:uppercase;
                        letter-spacing:.7px;font-weight:600;margin-bottom:3px">
              📊 Affollamento
            </div>
            <div style="display:inline-flex;align-items:center;gap:6px">
              <span style="background:{affoll_bg};color:{affoll_fg};
                           border-radius:20px;padding:2px 10px;
                           font-size:.8rem;font-weight:700">
                {affoll_txt}
              </span>
              <span style="font-size:.78rem;color:#94a3b8">{pct}%</span>
            </div>
          </div>
        </div>
        """)

        # ── Separatore ────────────────────────────────────────
        ui.element("div").style("border-top:1px solid #f1f5f9;margin-bottom:13px")

        # ── 5 codici triage ───────────────────────────────────
        ui.html("""
        <div style="font-size:.68rem;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:.7px;font-weight:600;margin-bottom:9px">
          Pazienti in sala d'attesa per codice triage
        </div>
        """)

        with ui.row().classes("gap-2 flex-wrap items-center"):
            _badge("🔴", "Rosso",     codici["Rosso"],     "#ef4444", "white",   "#ef4444")
            _badge("🟠", "Arancione", codici["Arancione"], "#f97316", "white",   "#f97316")
            _badge("🔵", "Azzurro",   codici["Azzurro"],   "#38bdf8", "#0c4a6e", "#38bdf8")
            _badge("🟢", "Verde",     codici["Verde"],     "#22c55e", "white",   "#22c55e")
            _badge("⚪", "Bianco",    codici["Bianco"],    "#f8fafc", "#475569", "#cbd5e1")

        # ── Footer card ───────────────────────────────────────
        ui.html(f"""
        <div style="margin-top:12px;font-size:.77rem;color:#94a3b8;
                    display:flex;flex-wrap:wrap;gap:10px;align-items:center">
          <span>👥 Tot. in attesa: <strong style="color:#64748b">
            {snap['n_pazienti']}
          </strong></span>
          <span style="color:#e2e8f0">·</span>
          <span>📌 {indirizzo}, {h['CAP']} {citta}</span>
        </div>
        """)


def _badge(emoji: str, nome: str, count: int, bg: str, fg: str, border: str) -> None:
    ui.html(f"""
    <div class="am-badge"
         style="background:{bg};color:{fg};border:1.5px solid {border}"
         title="{nome}">
      {emoji} {nome}
      <span style="background:rgba(0,0,0,.15);border-radius:20px;padding:1px 7px">
        {count}
      </span>
    </div>
    """)


# ════════════════════════════════════════════════════════════════
# SEZIONE 7 — ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ in {"__main__", "__mp_main__"}:

    # ── Imposta qui la chiave se non usi variabili d'ambiente ──
    # os.environ.setdefault("OPENAI_API_KEY", "sk-proj-...")

    # ── Serve i file statici (loghi, assets) ──────────────────
    _nicegui_app.add_static_files("/assets", str(BASE_DIR))

    ui.run(
        title="AuraMed — Assistente Medico AI",
        host="0.0.0.0",
        port=8080,
        dark=False,
        favicon="⚕",
        reload=False,
        storage_secret="auramed-lombardia-2025",
    )
