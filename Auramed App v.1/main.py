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
import re
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

from concurrent.futures import ThreadPoolExecutor, as_completed

# La API key viene letta dall'ambiente (TOMTOM_API_KEY), MAI hardcoded.
TOMTOM_KEY = os.environ.get("TOMTOM_API_KEY")
TOMTOM_ROUTE_URL = "https://api.tomtom.com/routing/1/calculateRoute"

DEFAULT_TIMEOUT = 10  # secondi per richiesta HTTP

# Quanti ospedali interrogare via TomTom per ricerca: si prefiltra ai più
# vicini (haversine) per contenere numero di chiamate e latenza.
TOP_ROUTING = 10

# Le chiamate a calculateRoute vengono eseguite in lotti paralleli di questa
# dimensione (es. 10 ospedali → 2 lotti da 5 chiamate ciascuno), invece che
# tutte insieme, per contenere il picco di richieste simultanee verso TomTom.
ROUTING_BATCH_SIZE = 5


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distanza in km — usata per il prefiltro e come fallback."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)


def _coord_osp(osp: dict) -> tuple[float, float]:
    """(lat, lon) di un ospedale dai campi flat 'Latitudine'/'Longitudine'."""
    return float(osp.get("Latitudine", 0.0)), float(osp.get("Longitudine", 0.0))


def _fallback_tempo(user_pos: dict, osp: dict) -> dict:
    """Stima di ripiego (haversine, 30 km/h urbana, min 5 min) se TomTom KO."""
    lat, lon = _coord_osp(osp)
    d = _haversine_km(user_pos["lat"], user_pos["lon"], lat, lon)
    return {
        "tempo_min": float(max(5, round((d / 30.0) * 60, 1))),
        "ritardo_min": None,
        "distanza_km": d,
        "geometria": [],
        "fallback": True,
    }


def _tomtom_route(user_pos: dict, osp: dict) -> dict:
    """
    Un percorso utente→ospedale via TomTom Routing API (auto, nel traffico).

    ATTENZIONE all'ordine coordinate:
        TomTom usa (lat, lon); OSRM usava (lon, lat). Qui: lat,lon.

    Ritorna dict con:
        tempo_min   — tempo nel traffico (best estimate)
        ritardo_min — ritardo dovuto al traffico
        distanza_km — distanza stradale reale
        geometria   — lista di [lat, lon] del tracciato (per la mappa)
    """
    lat, lon = _coord_osp(osp)
    loc = f"{user_pos['lat']},{user_pos['lon']}:{lat},{lon}"
    resp = _requests.get(
        f"{TOMTOM_ROUTE_URL}/{loc}/json",
        params={
            "key": TOMTOM_KEY,
            "traffic": "true",               # traffico in tempo reale
            "travelMode": "car",
            "routeType": "fastest",
            "computeTravelTimeFor": "all",   # anche il tempo senza traffico
            "routeRepresentation": "polyline",  # serve la geometria per i tracciati
        },
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    route = resp.json()["routes"][0]
    s = route["summary"]

    geom = [
        [p["latitude"], p["longitude"]]
        for leg in route.get("legs", [])
        for p in leg.get("points", [])
    ]
    ritardo_s = s.get("trafficDelayInSeconds")

    return {
        "tempo_min": round(s["travelTimeInSeconds"] / 60.0, 1),
        "ritardo_min": round(ritardo_s / 60.0, 1) if ritardo_s is not None else None,
        "distanza_km": round(s["lengthInMeters"] / 1000.0, 2),
        "geometria": geom,
        "fallback": False,
    }


def stima_tempi_ospedali(user_pos: dict, ospedali: list[dict]) -> list[dict]:
    """
    ╔─────────────────────────────────────────────────────────────╗
    ║  Tempi di percorrenza nel traffico via TomTom               ║
    ║                                                             ║
    ║  Una chiamata calculateRoute per ospedale, eseguita a lotti ║
    ║  paralleli da ROUTING_BATCH_SIZE (es. 10 ospedali → 2 lotti ║
    ║  da 5 chiamate ciascuno, in sequenza tra loro).             ║
    ║  Ritorna una lista di dict (tempo/ritardo/distanza/geom)    ║
    ║  allineata all'ordine di `ospedali`.                        ║
    ║                                                             ║
    ║  Fallback haversine per singolo ospedale se:               ║
    ║    - manca TOMTOM_API_KEY                                   ║
    ║    - la chiamata a quel percorso fallisce                  ║
    ╚─────────────────────────────────────────────────────────────╝
    """
    if not TOMTOM_KEY:
        print("[AuraMed] TOMTOM_API_KEY assente — fallback haversine per tutti.")
        return [_fallback_tempo(user_pos, o) for o in ospedali]

    risultati: list[Optional[dict]] = [None] * len(ospedali)
    for inizio in range(0, len(ospedali), ROUTING_BATCH_SIZE):
        lotto = list(enumerate(ospedali))[inizio:inizio + ROUTING_BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=len(lotto)) as ex:
            futuri = {ex.submit(_tomtom_route, user_pos, o): i for i, o in lotto}
            for fut in as_completed(futuri):
                i = futuri[fut]
                try:
                    risultati[i] = fut.result()
                except Exception as exc:
                    print(f"[AuraMed] TomTom KO su '{ospedali[i].get('Nome')}' "
                          f"({exc}) — fallback haversine")
                    risultati[i] = _fallback_tempo(user_pos, ospedali[i])
    return risultati


def _colore_traffico(tempo_min: Optional[float], ritardo_min: Optional[float]) -> str:
    """Colore del tracciato in base alla quota di ritardo da traffico."""
    if not ritardo_min or not tempo_min:
        return "#2ecc71"                      # scorre / dato assente
    quota = ritardo_min / tempo_min
    if quota >= 0.30:
        return "#e74c3c"                      # congestione
    if quota >= 0.10:
        return "#f1c40f"                      # rallentato
    return "#2ecc71"                          # scorre




# ════════════════════════════════════════════════════════════════
# SEZIONE 3 — ChatGPT.ipynb
# Analisi sintomi via OpenAI GPT
# ════════════════════════════════════════════════════════════════

def _get_openai_client():
    try:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        return OpenAI(api_key=key)
    except ImportError:
        return None


# Elenco reparti caricato una sola volta all'avvio e inserito come testo nel
# prompt (non più come file allegato via Files API: il CSV è ~1.6 KB e
# l'allegato costava ~3 s extra ad ogni chiamata senza alcun beneficio).
with open(BASE_DIR / "aree_specialistiche_lombardia_Chat.csv", encoding="utf-8") as _f:
    _REPARTI_DISPONIBILI: str = "\n".join(
        riga.strip() for riga in _f.readlines()[1:] if riga.strip()
    )


# Pattern di dati identificativi da rimuovere prima dell'invio all'LLM.
# NB: è una misura di minimizzazione, NON rende il testo anonimo:
# la descrizione dei sintomi resta un dato sanitario.

_PII_PATTERNS: list[tuple] = [
    (re.compile(r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b', re.I), '[CF]'),
    (re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b', re.I),                '[EMAIL]'),
    (re.compile(r'\bIT\d{2}[A-Z]\d{10}[0-9A-Z]{12}\b', re.I),         '[IBAN]'),
    (re.compile(r'(?:\+39[\s.-]?)?\b3\d{2}[\s.-]?\d{6,7}\b', re.I),   '[TEL]'),
    (re.compile(r'\b0\d{1,3}[\s.-]?\d{5,8}\b', re.I),                 '[TEL]'),
    (re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.I),          '[DATA]'),
    (re.compile(r'\b(?:via|viale|piazza|corso|largo)\s+[A-Z][\w\'’]+'
                r'(?:\s+[A-Z][\w\'’]+)*\s*,?\s*\d*', re.I),           '[INDIRIZZO]'),
    # Nomi propri introdotti da formule tipiche ("mi chiamo Mario", "sig. Rossi")
    (re.compile(r'\b(?:mi chiamo|sono|si chiama|il|la)\s+'
                r'(?:sig\.?r?a?\.?|dott\.?|signor[ea]?)\s+[A-Z][\w\'’]+', re.I),
     '[NOME]'),
    (re.compile(r'\bmi chiamo\s+[A-Z][\w\'’]+(?:\s+[A-Z][\w\'’]+)?', re.I), '[NOME]'),
    (re.compile(r'\b\d{9,}\b', re.I),                                 '[ID]'),
]



def anonimizza(testo: str) -> str:
    """Rimuove identificativi diretti dal testo prima dell'invio all'LLM.

    Copre: codice fiscale, email, IBAN, telefoni, date, indirizzi,
    nomi introdotti da formule tipiche, numeri lunghi (tessere/documenti).

    ATTENZIONE: misura di minimizzazione, non anonimizzazione ai sensi
    del GDPR. Il contenuto clinico resta un dato sanitario.
    """
    for pattern, sostituto in _PII_PATTERNS:
        testo = pattern.sub(sostituto, testo)
    return testo


def _descrivi_paziente(profilo: Optional[dict]) -> str:
    """
    Traduce le risposte del questionario iniziale in un blocco testuale
    da inserire nel prompt. Ritorna "" se il profilo non è disponibile.

    profilo = {"per_chi": "me"|"altri", "eta": int, "sesso": "M"|"F"}
    """
    if not profilo:
        return ""

    eta     = profilo.get("eta")
    sesso   = profilo.get("sesso")
    per_chi = profilo.get("per_chi")

    sesso_lbl = {"M": "maschio", "F": "femmina"}.get(sesso, "non specificato")
    righe = [
        "\nPROFILO DEL PAZIENTE (usa queste informazioni per affinare la valutazione):",
        f'- Età: {eta} anni' if eta is not None else "- Età: non specificata",
        f'- Sesso: {sesso_lbl}',
    ]

    if per_chi == "me":
        righe.append(
            "- Chi scrive è il paziente stesso: è presumibilmente SOLO e non "
            "necessariamente in grado di guidare o di accompagnarsi."
        )
    elif per_chi == "altri":
        righe.append(
            "- Chi scrive è un accompagnatore che riferisce per conto del "
            "paziente: è presente una seconda persona che può assistere e guidare."
        )

    # Indicazioni cliniche derivate dal profilo
    note: list[str] = []
    if isinstance(eta, int):
        if eta < 1:
            note.append(
                "Il paziente è un lattante (<1 anno): fascia molto fragile, "
                "privilegia reparti pediatrici/neonatologici e alza la soglia "
                "di attenzione (raramente codice bianco)."
            )
        elif eta < 14:
            note.append(
                "Il paziente è in età pediatrica: privilegia reparti pediatrici "
                "quando esistono in lista (es. PEDIATRIA) rispetto agli "
                "equivalenti per adulti."
            )
        elif eta >= 75:
            note.append(
                "Il paziente è un anziano: comorbidità e presentazioni atipiche "
                "sono frequenti (es. infarto senza dolore tipico, infezione con "
                "sola confusione). Sii più prudente nell'assegnare codice bianco."
            )
    if sesso == "F" and isinstance(eta, int) and 12 <= eta <= 55:
        note.append(
            "Paziente donna in età fertile: per dolore addominale/pelvico "
            "considera cause ostetrico-ginecologiche (es. gravidanza ectopica) "
            "e valuta reparti come GINECOLOGIA E OSTETRICIA."
        )
    if sesso == "M" and isinstance(eta, int) and eta >= 45:
        note.append(
            "Uomo over 45: rischio cardiovascolare più elevato; a parità di "
            "sintomi toracici o epigastrici mantieni alta la soglia di allerta."
        )
    if per_chi == "me":
        note.append(
            "Poiché il paziente è solo, se i sintomi possono compromettere la "
            "capacità di spostarsi autonomamente (vertigini, sincope, dolore "
            "toracico, deficit neurologici) NON assegnare codice bianco."
        )

    if note:
        righe.append("Considerazioni cliniche da applicare:")
        righe.extend(f"  · {n}" for n in note)

    return "\n".join(righe) + "\n"


def analizza_sintomi(testo: str, profilo: Optional[dict] = None) -> dict:
    """
    ╔─────────────────────────────────────────────────────────────────────────────────────╗
    ║  WRAPPER — ChatGPT.ipynb                                                            ║
    ║  Input : testo libero con i sintomi + profilo paziente (questionario)               ║
    ║  Output: {"isCritical": "Y"|"N", "white": "Y"|"N", "reparto": "NOME"}             ║
    ║                                                                                     ║
    ║  Usa OpenAI GPT-4.1-mini se OPENAI_API_KEY è disponibile,                           ║
    ║  altrimenti fallback keyword-based locale.                                          ║
    ╚─────────────────────────────────────────────────────────────────────────────────────╝
    """
    client = _get_openai_client()
    if not client:
        return _mock_analisi(testo, profilo)

    try:
        # Minimizzazione: rimuovi identificativi prima dell'invio esterno.
        testo_pulito = anonimizza(testo)
        prompt = (
             f'Sei un operatore esperto di triage di pronto soccorso. '
             f'Analizza la seguente affermazione di un paziente e classificala.\n\n'
             f'AFFERMAZIONE: "{testo_pulito}"\n'
             f'{_descrivi_paziente(profilo)}\n'
             f'Devi restituire tre informazioni:\n'
             f'1. "isCritical": "Y" se c\'è pericolo di vita imminente o un\'emergenza '
             f'che richiede intervento immediato (es. dolore toracico, difficoltà '
             f'respiratoria grave, perdita di coscienza, emorragia importante, '
             f'sospetto ictus o infarto, reazione allergica sistemica). Altrimenti "N".\n'
             f'2. "white": "Y" SOLO se la condizione è chiaramente lieve e non urgente, '
             f'gestibile anche senza pronto soccorso (es. febbre lieve, raffreddore, '
             f'singola puntura di insetto senza reazione sistemica, piccola '
             f'escoriazione, mal di gola). Altrimenti "N".\n'
             f'3. "reparto": il nome del reparto ospedaliero più appropriato, preso '
             f'ESATTAMENTE dalla lista di reparti disponibili qui sotto (nome '
             f'identico a quello in lista).\n\n'
             f'REPARTI DISPONIBILI:\n{_REPARTI_DISPONIBILI}\n\n'
             f'REGOLE:\n'
             f'- "isCritical" e "white" non possono essere entrambi "Y".\n'
             f'- Nel dubbio privilegia la sicurezza: NON assegnare "white":"Y" e non '
             f'abbassare la criticità.\n'
             f'- Un sintomo che sembra lieve ma presenta segnali d\'allarme NON è white '
             f'(es. puntura d\'ape CON difficoltà respiratoria -> "isCritical":"Y"; '
             f'febbre alta con confusione o rigidità nucale -> "white":"N").\n'
             f'- "reparto" va sempre valorizzato, anche per i codici white.\n'
             f'- Tieni conto del PROFILO DEL PAZIENTE: età, sesso e presenza o '
             f'meno di un accompagnatore influenzano sia il reparto sia la '
             f'criticità. A parità di sintomi, un bambino, un anziano o una '
             f'persona sola meritano maggiore prudenza.\n'
             f'- Scegli comunque il reparto SOLO tra quelli elencati sopra: se '
             f'il reparto pediatrico specifico non esiste in lista, usa il '
             f'reparto generale più appropriato.\n\n'
             f'ESEMPI:\n'
             f'- "Da un\'ora ho un forte dolore al petto e fatico a respirare" -> '
             f'{{"isCritical": "Y", "white": "N", "reparto": "CARDIOLOGIA"}}\n'
             f'- "Ho la febbre a 37.8 e un po\' di mal di gola da ieri" -> '
             f'{{"isCritical": "N", "white": "Y", "reparto": "MEDICINA GENERALE"}}\n'
             f'- "Mi ha punto un\'ape sul braccio, è solo un po\' gonfio" -> '
             f'{{"isCritical": "N", "white": "Y", "reparto": "PRONTO SOCCORSO"}}\n'
             f'- "Sono caduto e credo di essermi rotto il polso" -> '
             f'{{"isCritical": "N", "white": "N", "reparto": "ORTOPEDIA"}}\n\n'
             f'Restituisci UNICAMENTE un JSON valido, senza markdown e senza testo '
             f'aggiuntivo, in questo formato:\n'
             f'{{"isCritical": "Y", "white": "N", "reparto": "OCULISTICA"}}'
        )
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }],
        )
        return json.loads(response.output_text)
    except Exception as exc:
        print(f"[AuraMed] OpenAI error: {exc} — fallback mock")
        return _mock_analisi(testo)


def _mock_analisi(testo: str, profilo: Optional[dict] = None) -> dict:
    """Fallback senza API key. Stessa struttura output di ChatGPT.ipynb.

    Tiene conto del profilo (età/sesso/solo) con regole semplici:
      - età < 14      → reparto pediatrico se il testo non è già specifico
      - età < 1 o ≥75 → non assegna codice bianco (fascia fragile)
      - paziente solo → non assegna codice bianco se compaiono sintomi
                        che compromettono la capacità di spostarsi
    """
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
    _WHITE = [
        "raffreddore", "mal di gola", "graffio", "escoriazione", "lieve",
        "puntura d'insetto", "puntura di zanzara", "naso chiuso", "starnut",
        "piccolo taglio", "livido",
    ]
    is_critical = any(k in tl for k in _CRITICI)
    is_white = (not is_critical) and any(k in tl for k in _WHITE)
    reparto, best = "PRONTO SOCCORSO", 0
    for kws, rep in _REPARTI:
        s = sum(1 for k in kws if k in tl)
        if s > best:
            best, reparto = s, rep

    # ── Aggiustamenti in base al profilo del questionario ──────────
    if profilo:
        eta     = profilo.get("eta")
        per_chi = profilo.get("per_chi")

        # Età pediatrica → reparto pediatrico se il match è generico
        # (un reparto specifico tipo ORTOPEDIA/OCULISTICA viene mantenuto)
        _GENERICI = {"PRONTO SOCCORSO", "MEDICINA INTERNA - GENERALE"}
        if isinstance(eta, int) and eta < 14 and reparto in _GENERICI:
            reparto = "PEDIATRIA"

        # Fasce fragili: mai codice bianco
        if isinstance(eta, int) and (eta < 1 or eta >= 75):
            is_white = False

        # Paziente solo con sintomi che impediscono di spostarsi
        _NON_GUIDA = ("vertigin", "svenim", "sincope", "capogir", "confusione",
                      "vista offuscata", "debolezza", "non riesco a stare in piedi")
        if per_chi == "me" and any(k in tl for k in _NON_GUIDA):
            is_white = False

    return {
        "isCritical": "Y" if is_critical else "N",
        "white": "Y" if is_white else "N",
        "reparto": reparto,
    }





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

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

try:
    if not all((NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)):
        raise RuntimeError(
            "Credenziali Neo4j mancanti nell'ambiente "
            "(NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD). "
            "Configura il .env con: python setup_env.py"
        )
    import certifi
    from neo4j import GraphDatabase as _GraphDatabase, TrustCustomCAs

    # Come in GraphDB.ipynb: usiamo lo schema "neo4j" (senza "+s") con
    # encrypted=True e trusted_certificates=TrustCustomCAs(certifi.where())
    # perché su alcune macchine (es. Windows con store dei certificati non
    # aggiornato) il driver non trova la CA root usata dal certificato di
    # Aura nello store di sistema. Forzare il bundle certifi risolve
    # l'errore "Unable to retrieve routing information" / "self-signed
    # certificate in certificate chain".
    _neo4j_uri_no_ssl_scheme = NEO4J_URI.replace("neo4j+s://", "neo4j://").replace(
        "bolt+s://", "bolt://"
    )
    _neo4j_driver = _GraphDatabase.driver(
        _neo4j_uri_no_ssl_scheme,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        encrypted=True,
        trusted_certificates=TrustCustomCAs(certifi.where()),
    )
    _neo4j_driver.verify_connectivity()
    with _neo4j_driver.session(database=NEO4J_DATABASE) as _s:
        _s.run("CREATE INDEX area_nome IF NOT EXISTS FOR (a:AreaSpecialistica) ON (a.Nome)")
        _s.run("CREATE INDEX ospedale_nome IF NOT EXISTS FOR (o:Ospedale) ON (o.Nome)")
    print("[AuraMed] Neo4j connesso (indici verificati).")
except Exception as _neo4j_err:
    print(f"[AuraMed] Neo4j non disponibile: {_neo4j_err} — fallback locale")
    _neo4j_driver = None


def _carica_classificazioni() -> dict:
    """
    ╔─────────────────────────────────────────────────────────────╗
    ║  Ogni nodo :Struttura ha una proprietà Classificazione con  ║
    ║  valore "HUB", "SPOKE" o "CASA DI COMUNITA".                ║
    ║  Returns: {nome_struttura: classificazione} (vuoto offline) ║
    ╚─────────────────────────────────────────────────────────────╝
    """
    if _neo4j_driver is None:
        return {}
    cypher = "MATCH (s:Struttura) RETURN s.Nome AS nome, s.Classificazione AS classificazione"
    try:
        with _neo4j_driver.session(database=NEO4J_DATABASE) as s:
            return {r["nome"]: (r["classificazione"] or "") for r in s.run(cypher)}
    except Exception as exc:
        print(f"[AuraMed] Errore lettura classificazioni Neo4j: {exc} — fallback nome")
        return {}


# Caricata una sola volta all'avvio: la classificazione è statica per struttura.
CLASSIFICAZIONI: dict[str, str] = _carica_classificazioni()


def strutture_con_reparti(reparto: str) -> dict:
    """
    ╔─────────────────────────────────────────────────────────────╗
    ║  DA StruttureReparti.ipynb                                  ║
    ║  Critico  → tutti gli ospedali, isReparto = None            ║
    ║  Non crit → solo ospedali con PS;                           ║
    ║             isReparto = 1 se hanno anche il reparto, 0 no   ║
    ║  Returns: {nome_ospedale: isReparto}  (vuoto se Neo4j offline) ║
    ║                                                             ║
    ║  Nota: il grafo attuale non ha una 'Codice' univoco per     ║
    ║  Ospedale, quindi il join con OSPEDALI_DB avviene su Nome.  ║
    ║  Le aree specialistiche (incl. "PRONTO SOCCORSO") sono nodi ║
    ║  :AreaSpecialistica collegati via :HA_AREA (non :Reparto/   ║
    ║  :HA_REPARTO come nel prototipo StruttureReparti.ipynb).    ║
    ╚─────────────────────────────────────────────────────────────╝
    """
    if _neo4j_driver is None:
        return {}
    cypher = """
    MATCH (o:Ospedale)-[:HA_AREA]->(rep:AreaSpecialistica)
    WHERE rep.Nome = "PRONTO SOCCORSO"
    RETURN o.Nome AS nome,
    CASE
        WHEN EXISTS {
            MATCH (o)-[:HA_AREA]->(rep2:AreaSpecialistica)
            WHERE rep2.Nome = $reparto
        }
        THEN 1
        ELSE 0
    END AS isReparto"""
    params: dict = {"reparto": reparto}
    try:
        with _neo4j_driver.session(database=NEO4J_DATABASE) as s:
            return {r["nome"]: r["isReparto"] for r in s.run(cypher, **params)}
    except Exception as exc:
        print(f"[AuraMed] Errore query Neo4j: {exc} — fallback locale")
        return {}




# ════════════════════════════════════════════════════════════════
# SEZIONE 5 — PIPELINE PRINCIPALE
# ════════════════════════════════════════════════════════════════

def _classificazione(osp: dict) -> str:
    """Classificazione ('HUB'/'SPOKE'/'CASA DI COMUNITA...') da Neo4j; se il
    grafo non è disponibile, euristica di ripiego sul nome."""
    cl = CLASSIFICAZIONI.get(osp.get("Nome"))
    if cl:
        return cl.upper()
    return osp.get("Nome", "").upper()   # fallback: cerca "HUB"/"CASA DI COMUNIT" nel nome

def _is_hub(osp: dict) -> bool:
    return "HUB" in _classificazione(osp)

def _is_ccom(osp: dict) -> bool:
    return "CASA DI COMUNIT" in _classificazione(osp)


def google_maps_url(osp: dict) -> str:
    q = f"{osp['Nome']}, {osp['Indirizzo']}, {osp['Città']}"
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(q)}"


def _rating_display(osp: dict, reparto_nome: str) -> Optional[float]:
    """Rating da mostrare a video (SOLO informativo, non entra in alcun
    calcolo/punteggio): quello del reparto specifico se presente in
    Aree_specialistiche, altrimenti il Rating_globale dell'ospedale."""
    for area in osp.get("Aree_specialistiche") or []:
        if area.get("Nome", "").upper() == (reparto_nome or "").upper():
            rating = area.get("Rating")
            if rating is not None:
                return rating
            break
    return osp.get("Rating_globale")


def cerca_strutture(
    testo: str,
    profilo: Optional[dict] = None,
    posizione: dict = POSIZIONE_UTENTE,
    top_n: int = 5,
) -> list[dict]:
    """
    Pipeline completa:
      1. analizza_sintomi()     → isCritical / white (backend-only), reparto
      2. stima_tempi_ospedali() → tempi/ritardi/geometrie TomTom (top vicini)
      3. snapshot_ps()          → affollamento per ciascuno
      4. scoring                → dipende dal codice clinico
      5. Ordinamento e top_n

    Il codice clinico determina SIA le strutture candidate SIA l'ordinamento:
      - Critico (isCritical=Y) → PS più vicini per tempo di percorrenza
                                 (ogni secondo conta)
      - Bianco  (white=Y)      → Case di Comunità, con peso maggiore
                                 all'affollamento (non devono essere affollate)
      - Altrimenti             → PS con score composito ω1·T + ω2·A + ω3·R

    Il risultato dell'analisi (critico/bianco) NON viene esposto all'UI.
    """
    # Step 1 — analisi LLM (backend only)
    analisi = analizza_sintomi(testo, profilo)
    reparto_cercato = analisi.get("reparto", "PRONTO SOCCORSO").upper().strip()
    is_critical     = analisi.get("isCritical", "N") == "Y"
    is_white        = analisi.get("white", "N") == "Y"
    # Il critico ha sempre priorità: i due flag non possono coesistere.
    if is_critical:
        is_white = False

    # Step 1b — selezione candidati in base al codice clinico
    if is_white:
        # Codice bianco → Case di Comunità (nessun PS richiesto).
        grafo = {}
        ospedali_lista = [o for o in OSPEDALI_DB if _is_ccom(o)]
        if not ospedali_lista:                       # fallback difensivo
            ospedali_lista = OSPEDALI_DB
    else:
        # Critico / non critico → ospedali con PS (filtraggio grafo Neo4j).
        grafo = strutture_con_reparti(reparto_cercato)
        if grafo:
            ospedali_lista = [o for o in OSPEDALI_DB if o.get("Nome") in grafo]
        else:
            # Fallback se Neo4j non disponibile o senza match: le Case di
            # Comunità non hanno Pronto Soccorso, vanno comunque escluse.
            ospedali_lista = [o for o in OSPEDALI_DB if not _is_ccom(o)]

    # Step 2a — prefiltro ai TOP_ROUTING ospedali più vicini (haversine):
    # interroghiamo TomTom solo su un numero contenuto di candidati.
    def _dist_haversine(o: dict) -> float:
        lat, lon = _coord_osp(o)
        return _haversine_km(posizione["lat"], posizione["lon"], lat, lon)

    ospedali_vicini = sorted(ospedali_lista, key=_dist_haversine)[:TOP_ROUTING]

    # Step 2b — tempi/ritardi/geometrie via TomTom (chiamate parallele)
    percorsi: list[dict] = stima_tempi_ospedali(posizione, ospedali_vicini)

    # Step 3 & 4 — arricchimento e scoring
    risultati: list[dict] = []
    for osp, perc in zip(ospedali_vicini, percorsi):
        tempo_min = perc["tempo_min"]
        hub  = _is_hub(osp)
        ccom = _is_ccom(osp)
        if ccom:
            # Le Case di Comunità non hanno un sistema di triage/coda: non
            # conosciamo il numero di persone in attesa, quindi non lo
            # inventiamo (nessuno snapshot affollamento/pazienti).
            snap = {"codici": None, "n_pazienti": None, "affollamento": None}
        elif hub:
            snap = snapshot_ps(10, 80, 180)
        else:
            snap = snapshot_ps(5, 40, 100)

        is_reparto    = grafo.get(osp.get("Nome"))   # 1, 0, o None (critico/fallback/bianco)
        reparto_match = is_reparto == 1

        # Display: grafo ha priorità, poi check locale su Aree_specialistiche
        if is_white:
            reparto_disp = reparto_cercato or "CASA DI COMUNITÀ"
        else:
            reparto_disp = reparto_cercato if reparto_match else "PRONTO SOCCORSO"

        if is_critical or is_white:
            # Critico: ogni secondo conta. Bianco: le Case di Comunità non
            # hanno un dato di affollamento reale, quindi l'unico criterio
            # disponibile è il tempo di percorrenza (minor tempo → meglio).
            punteggio = -tempo_min
        else:
            punteggio = score_ps(snap, tempo_min, reparto_match)

        risultati.append({
            **osp,
            "_distanza_km":     perc["distanza_km"],   # distanza stradale reale (TomTom)
            "_tempo_min":       tempo_min,
            "_ritardo_min":     perc["ritardo_min"],   # ritardo da traffico
            "_geometria":       perc["geometria"],     # tracciato per la mappa
            "_snapshot":        snap,
            "_reparto_trovato": reparto_disp or "PRONTO SOCCORSO",
            "_punteggio":       punteggio,
            "_maps_url":        google_maps_url(osp),
            # Solo display: rating del reparto se presente, altrimenti globale.
            # NON entra nel calcolo di "_punteggio" né in nessun ordinamento.
            "_rating":          _rating_display(osp, reparto_disp or "PRONTO SOCCORSO"),
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


def _stelle_html(rating: Optional[float]) -> str:
    """Indicatore a stelle (riempimento 0-5) — SOLO display, nessun calcolo.
    Usa un overlay di stelle piene sopra stelle vuote, larghezza proporzionale
    al rating, per un riempimento parziale fedele (es. 4.3 → ~86%)."""
    if rating is None:
        return (
            '<span style="font-size:.76rem;color:#94a3b8;font-weight:600">'
            "Rating non disponibile</span>"
        )
    pct = max(0, min(100, (float(rating) / 5.0) * 100))
    return f"""
    <div style="display:inline-flex;align-items:center;gap:7px" title="{rating:.1f} / 5">
      <div style="position:relative;display:inline-block;font-size:1rem;
                  line-height:1;letter-spacing:2px">
        <span style="color:#e2e8f0">★★★★★</span>
        <span style="position:absolute;top:0;left:0;overflow:hidden;
                     width:{pct:.0f}%;white-space:nowrap;color:#f59e0b">★★★★★</span>
      </div>
      <span style="font-size:.78rem;color:#64748b;font-weight:700">{rating:.1f}</span>
    </div>
    """


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
                Descrivi i sintomi in linguaggio naturale e trova
                le strutture ospedaliere più idonee nella tua area.
              </p>
            </div>
            """)

            # ══════════════════════════════════════════════════
            # 2-bis. QUESTIONARIO INIZIALE (3 domande in sequenza)
            # ══════════════════════════════════════════════════
            # Stato del profilo paziente, compilato dallo stepper.
            profilo: dict = {"per_chi": None, "eta": None, "sesso": None}

            with ui.element("div").classes("w-full").style(
                "background:white;border-radius:20px;padding:24px;"
                "border:1px solid #e2e8f0;"
                "box-shadow:0 2px 12px rgba(0,0,0,.05);margin-bottom:24px"
            ) as profilo_card:
                ui.html(f"""
                <div style="display:flex;align-items:center;gap:9px;margin-bottom:14px">
                  <div style="width:5px;height:20px;border-radius:3px;
                              background:linear-gradient(180deg,{_C_TEAL},{_C_TEAL2})">
                  </div>
                  <span style="font-weight:700;color:{_C_NAVY};font-size:.95rem">
                    Prima di iniziare
                  </span>
                </div>
                """)

                with ui.stepper().props("vertical flat").classes("w-full") as stepper:

                    # ── Domanda 1: per chi ─────────────────────────
                    with ui.step("Per chi stai cercando?"):
                        per_chi_radio = ui.radio(
                            {"me": "Per me", "altri": "Per un'altra persona"}
                        ).props("inline")
                        with ui.stepper_navigation():
                            def _next_1() -> None:
                                if not per_chi_radio.value:
                                    ui.notify("Seleziona un'opzione.",
                                              type="warning", position="top")
                                    return
                                profilo["per_chi"] = per_chi_radio.value
                                stepper.next()
                            ui.button("Avanti", on_click=_next_1).props("no-caps")

                    # ── Domanda 2: età ─────────────────────────────
                    with ui.step("Età del richiedente cura"):
                        eta_input = (
                            ui.number(label="Età (anni)", value=None,
                                      min=0, max=120, format="%.0f")
                            .props("outlined dense")
                            .classes("w-40")
                        )
                        with ui.stepper_navigation():
                            def _next_2() -> None:
                                val = eta_input.value
                                if val is None or not (0 <= int(val) <= 120):
                                    ui.notify("Inserisci un'età valida (0-120).",
                                              type="warning", position="top")
                                    return
                                profilo["eta"] = int(val)
                                stepper.next()
                            ui.button("Avanti", on_click=_next_2).props("no-caps")
                            ui.button("Indietro", on_click=stepper.previous) \
                                .props("flat no-caps")

                    # ── Domanda 3: sesso ───────────────────────────
                    with ui.step("Sesso del richiedente cura"):
                        sesso_radio = ui.radio(
                            {"M": "Maschio", "F": "Femmina"}
                        ).props("inline")
                        with ui.stepper_navigation():
                            def _fine() -> None:
                                if not sesso_radio.value:
                                    ui.notify("Seleziona un'opzione.",
                                              type="warning", position="top")
                                    return
                                profilo["sesso"] = sesso_radio.value
                                _completa_profilo()
                            ui.button("Conferma", on_click=_fine).props("no-caps")
                            ui.button("Indietro", on_click=stepper.previous) \
                                .props("flat no-caps")

                # Riepilogo mostrato a questionario completato
                riepilogo = ui.html("").classes("w-full")
                riepilogo.set_visibility(False)

                modifica_btn = (
                    ui.button("✏️  Modifica risposte")
                    .props("flat dense no-caps")
                    .style(
                        f"color:{_C_TEAL2};font-size:.78rem;font-weight:700;"
                        "padding:6px 0 0;min-height:0"
                    )
                )
                modifica_btn.set_visibility(False)

            # ══════════════════════════════════════════════════
            # 3. BOX RICERCA
            # ══════════════════════════════════════════════════
            with ui.element("div").classes("w-full").style(
                "background:white;border-radius:20px;padding:24px;"
                "border:1px solid #e2e8f0;"
                "box-shadow:0 2px 12px rgba(0,0,0,.05);margin-bottom:32px"
            ) as ricerca_card:
                ui.html(f"""
                <div style="display:flex;align-items:center;gap:9px;margin-bottom:14px">
                  <div style="width:5px;height:20px;border-radius:3px;
                              background:linear-gradient(180deg,{_C_TEAL},{_C_TEAL2})">
                  </div>
                  <span style="font-weight:700;color:{_C_NAVY};font-size:.95rem">
                    Descrivi i sintomi
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

    # ── Gestione questionario ──────────────────────────────────
    # Finché il profilo non è completo, la ricerca resta bloccata.
    ricerca_card.style("opacity:.45;pointer-events:none")
    textarea.disable()
    cerca_btn.disable()

    def _completa_profilo() -> None:
        """Chiude il questionario, mostra il riepilogo e sblocca la ricerca."""
        stepper.set_visibility(False)

        per_chi_lbl = "Per me" if profilo["per_chi"] == "me" else "Per un'altra persona"
        sesso_lbl   = "Maschio" if profilo["sesso"] == "M" else "Femmina"
        riepilogo.set_content(f"""
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
          <span style="font-size:.8rem;color:#64748b">Profilo:</span>
          <span style="background:#e6fffd;color:{_C_NAVY};border-radius:8px;
                       padding:3px 10px;font-size:.78rem;font-weight:600">
            {per_chi_lbl}</span>
          <span style="background:#e6fffd;color:{_C_NAVY};border-radius:8px;
                       padding:3px 10px;font-size:.78rem;font-weight:600">
            {profilo['eta']} anni</span>
          <span style="background:#e6fffd;color:{_C_NAVY};border-radius:8px;
                       padding:3px 10px;font-size:.78rem;font-weight:600">
            {sesso_lbl}</span>
        </div>
        """)
        riepilogo.set_visibility(True)
        modifica_btn.set_visibility(True)

        # Sblocco del box ricerca
        ricerca_card.style("opacity:1;pointer-events:auto")
        textarea.enable()
        cerca_btn.enable()
        ui.notify("Profilo salvato — ora puoi descrivere i sintomi.",
                  type="positive", position="top")

    def _modifica_profilo() -> None:
        """Riapre il questionario per permettere di cambiare le 3 risposte,
        precompilato con i valori attuali (usato per una nuova ricerca)."""
        per_chi_radio.value = profilo["per_chi"]
        eta_input.value = profilo["eta"]
        sesso_radio.value = profilo["sesso"]

        riepilogo.set_visibility(False)
        modifica_btn.set_visibility(False)
        stepper.set_visibility(True)
        stepper.value = "Per chi stai cercando?"

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

        if None in (profilo["per_chi"], profilo["eta"], profilo["sesso"]):
            ui.notify(
                "⚠️  Completa prima le domande iniziali.",
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
            ospedali, is_critical = await loop.run_in_executor(
                None, cerca_strutture, testo, dict(profilo)
            )
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
        routes_js  = ""
        for h in ospedali:
            lat = h.get("Latitudine", 0.0)
            lon = h.get("Longitudine", 0.0)
            if lat and lon:
                nome = h.get("Nome", "").replace("'", "\\'")
                markers_js += f"L.marker([{lat},{lon}]).bindPopup('{nome}').addTo(m);\n"

            # Tracciato utente→ospedale, colorato in base al ritardo da traffico
            geom = h.get("_geometria") or []
            if len(geom) >= 2:
                colore = _colore_traffico(h.get("_tempo_min"), h.get("_ritardo_min"))
                pts = ",".join(f"[{p[0]},{p[1]}]" for p in geom)
                routes_js += (
                    f"L.polyline([{pts}],"
                    f"{{color:'{colore}',weight:5,opacity:0.85}}).addTo(m);\n"
                )

        # Layer traffico TomTom (tile raster: strade colorate per congestione).
        # La key è nell'URL lato client: vincolala a un dominio sul portale TomTom.
        if TOMTOM_KEY:
            traffic_layer_js = (
                'L.tileLayer("https://api.tomtom.com/traffic/map/4/tile/flow/'
                'relative0/{z}/{x}/{y}.png?key=' + TOMTOM_KEY + '",'
                # pane "overlayPane" (invece del default tilePane) per stare
                # sopra il velo bianco, insieme a geojson/tracciati/marker.
                '{maxZoom:22,opacity:1,pane:"overlayPane"}).addTo(m);'
            )
        else:
            traffic_layer_js = ""

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

            # ── Mappa: un unico file HTML riusato (sovrascritto ad ogni
            # ricerca), servito via /assets/. I confini della Lombardia
            # sono caricati via fetch dall'asset statico già servito
            # (lombardia.geojson) invece di essere incorporati inline nel
            # documento: il browser lo scarica/cachea una sola volta anziché
            # ri-scaricare ~230 KB di GeoJSON ad ogni ricerca, e il file
            # mappa su disco resta uno solo invece di accumularsi.
            _map_fname = "_map_risultati.html"
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
// Pane dedicato tra tilePane (200) e overlayPane (400, dove finiscono
// geojson/tracciati/marker): ci sta solo il velo bianco sopra OSM.
m.createPane("maskPane");
m.getPane("maskPane").style.zIndex = 350;
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",
  {{attribution: "\\u00a9 OpenStreetMap contributors", maxZoom: 19}}).addTo(m);
L.rectangle([[-90,-180],[90,180]],
  {{pane:"maskPane", stroke:false, fillColor:"#ffffff", fillOpacity:0.4,
   interactive:false}}).addTo(m);
fetch("/assets/lombardia.geojson").then(r => r.json()).then(function(data) {{
  var lombardiaLayer = L.geoJSON(data, {{
    style: {{color:"black", weight:2, opacity:0.55, fillColor:"#dcfce7", fillOpacity:0.4}}
  }}).addTo(m);
  m.fitBounds(lombardiaLayer.getBounds());
}});
{traffic_layer_js}
L.circleMarker([{u_lat}, {u_lon}],
  {{radius:10, color:"#009E94", fillColor:"#00C2B5", fillOpacity:0.85, weight:2}})
  .bindPopup("La tua posizione").addTo(m);
{routes_js}
{markers_js}
</script>
</body>
</html>
""", encoding="utf-8")
            # Cache-buster sul src dell'iframe: il nome file resta fisso, ma
            # ogni ricerca deve forzare il browser a ricaricare il contenuto
            # aggiornato (markers/routes) invece di servire la vecchia mappa
            # dalla cache.
            _cache_bust = int(time.time() * 1000)
            (ui.element("iframe")
                .props(f'src="/assets/{_map_fname}?v={_cache_bust}"')
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

            # ── Disclaimer TomTom + disclaimer affluenze PS con logo E015 (sotto la mappa) ──────
            ui.html(f"""
            <div style="display:flex;flex-direction:column;gap:10px;margin-bottom:20px;
                        padding:10px 14px;background:#f8fafc;border:1px solid #e2e8f0;
                        border-radius:10px">
              <span style="font-size:.72rem;color:#64748b;line-height:1.4">
                🚦 Percorsi e traffico stimati tramite servizi
                <a href="https://www.tomtom.com" target="_blank" rel="noopener"
                   style="color:{_C_TEAL2};text-decoration:none;font-weight:600">TomTom</a>.
              </span>
              <div style="display:flex;align-items:center;justify-content:space-between;
                          flex-wrap:wrap;gap:12px">
                <span style="font-size:.72rem;color:#64748b;line-height:1.4;max-width:640px">
                  🏥 Dati sulle affluenze dei Pronto Soccorso forniti tramite i servizi
                  <a href="https://www.e015.regione.lombardia.it" target="_blank" rel="noopener"
                     style="color:{_C_TEAL2};text-decoration:none;font-weight:600">E015 Digital Ecosystem</a>.
                </span>
                <a href="https://www.e015.regione.lombardia.it" target="_blank" rel="noopener"
                   title="E015 Digital Ecosystem" style="display:inline-flex;flex-shrink:0">
                  <img src="/assets/Logo_E015.png" alt="E015 Digital Ecosystem"
                       style="height:28px;width:auto;display:block">
                </a>
              </div>
            </div>
            """)

            for idx, h in enumerate(ospedali):
                _render_card(idx, h)

    cerca_btn.on_click(on_cerca)
    modifica_btn.on_click(_modifica_profilo)


# ── Componente card ospedale ────────────────────────────────────

def _render_card(idx: int, h: dict) -> None:
    snap         = h["_snapshot"]
    ha_affoll    = snap["affollamento"] is not None
    reparto_disp = h["_reparto_trovato"].title()
    nome         = h["Nome"].title()
    indirizzo    = h["Indirizzo"].title()
    citta        = h["Città"].title()
    ritardo_min  = h.get("_ritardo_min")
    classif      = _classificazione(h).title()
    ritardo_html = (
        f'<div style="font-size:.72rem;color:#e11d48;font-weight:600;'
        f'margin-top:2px">+{ritardo_min} min traffico</div>'
        if ritardo_min else ""
    )

    if ha_affoll:
        affoll_txt, affoll_bg, affoll_fg = _affollamento_info(snap["affollamento"])
        pct = round(snap["affollamento"] * 100)
        affollamento_html = f"""
          <span style="background:{affoll_bg};color:{affoll_fg};
                       border-radius:20px;padding:2px 10px;
                       font-size:.8rem;font-weight:700">
            {affoll_txt}
          </span>
          <span style="font-size:.78rem;color:#94a3b8">{pct}%</span>
        """
    else:
        affollamento_html = """
          <span style="background:#f1f5f9;color:#64748b;
                       border-radius:20px;padding:2px 10px;
                       font-size:.8rem;font-weight:700">
            Dato non disponibile
          </span>
        """

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
                {classif} &nbsp;·&nbsp; {h.get('Ente','')}
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

        # ── Reparto badge + rating (solo display, non usato nei calcoli) ──
        ui.html(f"""
        <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;
                    margin-bottom:14px">
          <span style="display:inline-flex;align-items:center;gap:6px;
                       background:#e0f7f5;color:{_C_TEAL2};
                       border:1.5px solid #b2ece7;border-radius:20px;
                       padding:4px 13px;font-size:.78rem;font-weight:700">
            🏥 {reparto_disp}
          </span>
          {_stelle_html(h.get('_rating'))}
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
            {ritardo_html}
          </div>
          <div>
            <div style="font-size:.68rem;color:#94a3b8;text-transform:uppercase;
                        letter-spacing:.7px;font-weight:600;margin-bottom:3px">
              📊 Affollamento
            </div>
            <div style="display:inline-flex;align-items:center;gap:6px">
              {affollamento_html}
            </div>
          </div>
        </div>
        """)

        # ── Separatore ────────────────────────────────────────
        ui.element("div").style("border-top:1px solid #f1f5f9;margin-bottom:13px")

        # ── 5 codici triage — solo per strutture con PS reale ─
        if ha_affoll:
            codici = snap["codici"]
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

            ui.element("div").style("border-top:1px solid #f1f5f9;margin:13px 0")

        # ── Footer card ───────────────────────────────────────
        attesa_html = (
            f"""<span>👥 Tot. in attesa: <strong style="color:#64748b">
                  {snap['n_pazienti']}
                </strong></span>
                <span style="color:#e2e8f0">·</span>"""
            if ha_affoll else ""
        )
        ui.html(f"""
        <div style="margin-top:12px;font-size:.77rem;color:#94a3b8;
                    display:flex;flex-wrap:wrap;gap:10px;align-items:center">
          {attesa_html}
          <span>📌 {indirizzo}, {citta}</span>
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
