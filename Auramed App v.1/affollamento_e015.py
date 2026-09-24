#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
affollamento_e015.py — Affollamento reale dei Pronto Soccorso lombardi
via API E015 "Situazione Pronto Soccorso In Lombardia v2" (SISS_EUOL).

Sostituisce i dati di affollamento generati casualmente (snapshot_ps):
fornisce l'indice di affollamento reale e i pazienti in attesa per codice
triage, nella STESSA forma dello snapshot usato dall'app, così il resto di
main.py (scoring e card) non cambia.

Forma dello snapshot restituito (identica a snapshot_ps):
    {
      "codici": {"Rosso":n,"Arancione":n,"Azzurro":n,"Verde":n,"Bianco":n},
      "n_pazienti": int,          # totale pazienti in attesa
      "affollamento": float,      # indice IA / 100  (0.50=affollato, 1.00=sovraffollato)
      "indAff": int,              # indice IA grezzo (0-180+), per riferimento
      "ultimo_agg": str | None,   # data/ora ultimo aggiornamento (policy: va mostrata)
    }
Ritorna None se: struttura non è un PS / PS chiuso / API non disponibile /
nessun abbinamento trovato. Il chiamante decide il fallback.

──────────────────────────────────────────────────────────────────────────
AUTENTICAZIONE (OAuth2 client_credentials — dal descrittore §3.3)
    POST {TOKEN_URL}
      Authorization: Basic base64(ConsumerKey:ConsumerSecret)
      Content-Type: application/x-www-form-urlencoded
      body: grant_type=client_credentials&scope=<scope1%20scope2...>
    → { "access_token": "...", "token_type":"Bearer", "expires_in":1800 }
Il token dura 30 min: qui è messo in cache e rinnovato prima della scadenza
(e in modo reattivo su HTTP 401).

Le credenziali NON stanno nel codice: si leggono dall'ambiente (.env):
    E015_CONSUMER_KEY, E015_CONSUMER_SECRET, E015_PS_SCOPES
Endpoint sovrascrivibili via env (default = descrittore v2.0):
    E015_TOKEN_URL, E015_LISTA_URL, E015_DETTAGLIO_URL

──────────────────────────────────────────────────────────────────────────
LIMITI D'USO (policy Regione Lombardia)
  • max 1000 chiamate/ora  • dato aggiornato ogni 3 minuti
Per rispettarli, LISTA e DETTAGLIO sono in cache CACHE_TTL secondi (default
150s, sotto i 180s di refresh): le azioni ripetute sull'app NON generano
una chiamata ciascuna. La LISTA (una sola chiamata) copre l'indice di
affollamento di TUTTI i PS ed è usata anche per abbinare ogni ospedale al
suo identificativo EUOL.
"""

from __future__ import annotations

import base64
import math
import os
import threading
import time
from typing import Optional
from urllib.parse import quote

import requests

# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

TOKEN_URL = os.getenv("E015_TOKEN_URL", "https://api.servizirl.it/oauth2/token")
# Base URL dalla dotazione ARIA: deployment v2.0.0, entrambi i metodi sotto /v2/.
# Confermato dalla sonda diagnostica: LISTA e DETTAGLIO rispondono 200 su
# .../euol/v2.0.0/v2/...  (il descrittore riportava v3, non attivo per questa
# abilitazione). Gli URL restano sovrascrivibili via .env se necessario.
LISTA_URL = os.getenv(
    "E015_LISTA_URL",
    "https://api.servizirl.it/c/servizi.rl/siss/euol/v2.0.0/v2/lista-pronto-soccorso",
)
DETTAGLIO_URL = os.getenv(
    "E015_DETTAGLIO_URL",
    "https://api.servizirl.it/c/servizi.rl/siss/euol/v2.0.0/v2/dettaglio-pronto-soccorso",
)

CONSUMER_KEY    = os.getenv("E015_CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("E015_CONSUMER_SECRET")
PS_SCOPES       = os.getenv("E015_PS_SCOPES", "")   # scope separati da spazio

CACHE_TTL = 150          # secondi di validità della cache (refresh dato = 180s)
HTTP_TIMEOUT = int(os.getenv("E015_HTTP_TIMEOUT", "30"))  # secondi per richiesta
HTTP_RETRIES = 2         # tentativi aggiuntivi su timeout / errori di rete
MATCH_MAX_KM = 2.0       # distanza massima per abbinare ospedale ↔ PS dell'API

# Codice colore triage: sigla dell'API v2 → etichetta (app).
# Il DETTAGLIO v2 restituisce output.codici = [{colore, attesa, trat}, ...]
# con SIGLE a una lettera. La maggior parte dei PS lombardi usa ancora la
# scala storica a 4 colori (R/G/V/B); la riforma a 5 codici usa R/A/Z/V/B.
# Mappiamo entrambe, così ogni PS mostra esattamente i codici che espone.
_SIGLA_COLORE = {
    "R": "Rosso", "A": "Arancione", "G": "Giallo",
    "Z": "Azzurro", "AZ": "Azzurro", "V": "Verde", "B": "Bianco",
}
# Ordine di gravità (per presentare i badge dal più al meno urgente).
_ORDINE_COLORI = ["Rosso", "Arancione", "Giallo", "Azzurro", "Verde", "Bianco"]

# File opzionale per "fissare" gli abbinamento ospedale→id dopo revisione
# manuale (Nome ospedale → id EUOL). Se presente, ha priorità sul match
# per coordinate. Sta nella stessa cartella di questo modulo.
_ID_MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "e015_id_map.json")


# ---------------------------------------------------------------------------
# STATO INTERNO (cache thread-safe: l'app chiama in thread executor)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_token: dict = {"value": None, "scad": 0.0}          # access token + scadenza epoch
_lista_cache: dict = {"ts": 0.0, "data": None}        # risposta LISTA in cache
_dettaglio_cache: dict = {}                           # id -> {"ts":.., "data":..}
_id_map: Optional[dict] = None                         # Nome ospedale -> id EUOL
_id_map_caricata = False


def disponibile() -> bool:
    """True se le credenziali sono configurate (API utilizzabile)."""
    return bool(CONSUMER_KEY and CONSUMER_SECRET and PS_SCOPES)


# ---------------------------------------------------------------------------
# HTTP con ritentativi (l'endpoint ARIA può essere lento/intermittente)
# ---------------------------------------------------------------------------

def _post(url: str, **kwargs):
    """requests.post con timeout di default e ritentativi su timeout/errori
    di rete (non sui codici HTTP, che sono gestiti dai chiamanti)."""
    kwargs.setdefault("timeout", HTTP_TIMEOUT)
    ultimo = None
    for tentativo in range(HTTP_RETRIES + 1):
        try:
            return requests.post(url, **kwargs)
        except requests.exceptions.RequestException as exc:
            ultimo = exc
            if tentativo < HTTP_RETRIES:
                time.sleep(1.5 * (tentativo + 1))   # backoff: 1.5s, 3s
    raise ultimo


# ---------------------------------------------------------------------------
# AUTENTICAZIONE
# ---------------------------------------------------------------------------

def _get_token() -> Optional[str]:
    """Restituisce un AccessToken valido, rinnovandolo se scaduto."""
    with _lock:
        if _token["value"] and time.time() < _token["scad"]:
            return _token["value"]
        if not disponibile():
            return None

        basic = base64.b64encode(
            f"{CONSUMER_KEY}:{CONSUMER_SECRET}".encode()
        ).decode()
        scope_enc = quote(PS_SCOPES.strip())
        try:
            resp = _post(
                TOKEN_URL,
                headers={
                    "Authorization": f"Basic {basic}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data=f"grant_type=client_credentials&scope={scope_enc}",
            )
            resp.raise_for_status()
            j = resp.json()
        except Exception as exc:
            print(f"[E015] Errore richiesta token: {exc}")
            return None

        _token["value"] = j.get("access_token")
        # rinnovo 60s prima della scadenza dichiarata
        _token["scad"] = time.time() + int(j.get("expires_in", 1800)) - 60
        return _token["value"]


def _headers() -> Optional[dict]:
    tok = _get_token()
    if not tok:
        return None
    return {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}


def _estrai_lista(data) -> list:
    """Estrae in modo difensivo l'array di PS dalla risposta LISTA.
    Il descrittore rende 'output' in modo ambiguo: gestiamo output=lista,
    oppure la prima lista trovata nel dict, oppure data stesso già lista."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        out = data.get("output")
        if isinstance(out, list):
            return out
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def _estrai_output(data) -> Optional[dict]:
    """Estrae l'oggetto 'output' dalla risposta DETTAGLIO."""
    if isinstance(data, dict):
        out = data.get("output")
        if isinstance(out, dict):
            return out
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return out[0]
        # a volte i campi sono già al primo livello
        if "codici" in data:
            return data
    return None


# ---------------------------------------------------------------------------
# CHIAMATE API (con cache)
# ---------------------------------------------------------------------------

def _get_lista() -> list:
    """LISTA di tutti i PS (id, coord, indAff, ...). In cache CACHE_TTL s."""
    with _lock:
        if _lista_cache["data"] is not None and \
           time.time() - _lista_cache["ts"] < CACHE_TTL:
            return _lista_cache["data"]
    h = _headers()
    if not h:
        return []
    try:
        resp = _post(LISTA_URL, headers=h, json={})
        if resp.status_code == 401:                 # token scaduto: rinnova 1 volta
            _token["scad"] = 0.0
            h = _headers()
            resp = _post(LISTA_URL, headers=h, json={})
        resp.raise_for_status()
        lista = _estrai_lista(resp.json())
    except Exception as exc:
        print(f"[E015] Errore LISTA pronto soccorso: {exc}")
        return []
    with _lock:
        _lista_cache["data"] = lista
        _lista_cache["ts"] = time.time()
    return lista


def _get_dettaglio(id_ps: str) -> Optional[dict]:
    """DETTAGLIO di un PS (livelli triage + indAff). In cache CACHE_TTL s."""
    now = time.time()
    with _lock:
        c = _dettaglio_cache.get(id_ps)
        if c and now - c["ts"] < CACHE_TTL:
            return c["data"]
    h = _headers()
    if not h:
        return None
    try:
        resp = _post(DETTAGLIO_URL, headers=h, json={"id": id_ps})
        if resp.status_code == 401:
            _token["scad"] = 0.0
            h = _headers()
            resp = _post(DETTAGLIO_URL, headers=h, json={"id": id_ps})
        resp.raise_for_status()
        out = _estrai_output(resp.json())
    except Exception as exc:
        print(f"[E015] Errore DETTAGLIO PS id={id_ps}: {exc}")
        return None
    with _lock:
        _dettaglio_cache[id_ps] = {"ts": time.time(), "data": out}
    return out


# ---------------------------------------------------------------------------
# ABBINAMENTO ospedale (app) → id EUOL (API)
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _carica_id_map() -> dict:
    """Carica (una volta) l'eventuale mappa manuale Nome→id EUOL."""
    global _id_map, _id_map_caricata
    if _id_map_caricata:
        return _id_map or {}
    _id_map_caricata = True
    try:
        import json
        with open(_ID_MAP_FILE, encoding="utf-8") as f:
            _id_map = {_norm(k): v for k, v in json.load(f).items()}
        print(f"[E015] Mappa id manuale caricata: {len(_id_map)} voci")
    except FileNotFoundError:
        _id_map = {}
    except Exception as exc:
        print(f"[E015] Mappa id non valida ({exc}) — uso match per coordinate")
        _id_map = {}
    return _id_map


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").upper() if ch.isalnum() or ch == " ").strip()


# Parole "di riempimento" ignorate nel confronto dei nomi (non distinguono
# una struttura dall'altra): così restano solo i token significativi.
_FILLER = {"OSPEDALE", "OSPEDALIERO", "OSPEDALIERA", "PRESIDIO", "IRCCS",
           "FONDAZIONE", "ISTITUTO", "CLINICO", "CLINICA", "CENTRO",
           "POLICLINICO", "AZIENDA", "PRONTO", "SOCCORSO", "GENERALE",
           "PEDIATRICO", "PEDIATRICA", "GRUPPO", "PPI"}


def _tokens_nome(s: str) -> set:
    return {w for w in _norm(s).split() if len(w) > 2 and w not in _FILLER}


def _name_score(nome_osp: str, nome_api: str) -> float:
    """Quota di token significativi dell'ospedale presenti nel nome API (0..1)."""
    ta = _tokens_nome(nome_osp)
    if not ta:
        return 0.0
    return len(ta & _tokens_nome(nome_api)) / len(ta)


def _match_dettaglio(osp: dict) -> dict:
    """Abbina un ospedale dell'app a un PS della LISTA. Strategia:
      1) mappa manuale (Nome → id EUOL), se presente
      2) coordinate ravvicinate (nearest ≤ MATCH_MAX_KM)
      3) stesso comune + nome che combacia  → recupera i PS con coordinate
         errate nel dataset (es. Lodi/Sondrio) SENZA mappe manuali
      4) entro il doppio soglia con conferma sul nome
    Restituisce {id, id_vicino, distanza_km, nome_api, metodo}; id=None se
    nessun match valido (id_vicino = PS più vicino, per diagnostica/mappa)."""
    nome = osp.get("Nome", "")
    citta = _norm(osp.get("Città", ""))
    vuoto = {"id": None, "id_vicino": None, "distanza_km": None,
             "nome_api": None, "metodo": None}

    # 1) mappa manuale (Nome ospedale → id EUOL)
    manuale = _carica_id_map().get(_norm(nome))
    if manuale:
        return {"id": manuale, "id_vicino": manuale, "distanza_km": 0.0,
                "nome_api": None, "metodo": "mappa"}

    lista = _get_lista()
    if not lista:
        return vuoto

    lat = float(osp.get("Latitudine", 0.0) or 0.0)
    lon = float(osp.get("Longitudine", 0.0) or 0.0)

    piu_vicino = None            # (dist, id, nome_api)  — minor distanza
    best_comune = None           # (name_score, dist|None, id, nome_api) — stesso comune
    for ps in lista:
        pid, pnome = ps.get("id"), ps.get("nome", "")
        d = None
        if lat and lon:
            try:
                d = _haversine_km(lat, lon, float(ps.get("lat")), float(ps.get("lon")))
            except (TypeError, ValueError):
                d = None
        if d is not None and (piu_vicino is None or d < piu_vicino[0]):
            piu_vicino = (d, pid, pnome)
        if citta and _norm(ps.get("comune", "")) == citta:
            ns = _name_score(nome, pnome)
            if best_comune is None or ns > best_comune[0]:
                best_comune = (ns, d, pid, pnome)

    # 2) coordinate ravvicinate
    if piu_vicino and piu_vicino[0] <= MATCH_MAX_KM:
        d, pid, pnome = piu_vicino
        return {"id": pid, "id_vicino": pid, "distanza_km": round(d, 3),
                "nome_api": pnome, "metodo": "coordinate"}

    # 3) stesso comune + nome coerente (coordinate assenti o errate)
    if best_comune and best_comune[0] >= 0.5:
        ns, d, pid, pnome = best_comune
        return {"id": pid, "id_vicino": pid,
                "distanza_km": round(d, 3) if d is not None else None,
                "nome_api": pnome, "metodo": "comune+nome"}

    # 4) entro il doppio soglia, con conferma sul nome
    if piu_vicino and piu_vicino[0] <= MATCH_MAX_KM * 2 and \
       _name_score(nome, piu_vicino[2]) >= 0.5:
        d, pid, pnome = piu_vicino
        return {"id": pid, "id_vicino": pid, "distanza_km": round(d, 3),
                "nome_api": pnome, "metodo": "coordinate+nome"}

    # nessun match valido: riporto il più vicino (per diagnostica/mappa)
    if piu_vicino:
        d, pid, pnome = piu_vicino
        return {"id": None, "id_vicino": pid, "distanza_km": round(d, 3),
                "nome_api": pnome, "metodo": "oltre_soglia"}
    return vuoto


def _match_id(osp: dict) -> Optional[str]:
    """Id EUOL del PS corrispondente all'ospedale dell'app, o None."""
    return _match_dettaglio(osp).get("id")


def _num(v) -> Optional[float]:
    """Converte in float in modo tollerante ('' / None → None)."""
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _lista_entry(id_ps: str) -> Optional[dict]:
    """Voce della LISTA (indAff, aperto, ultimoAgg, ...) per un id EUOL."""
    for ps in _get_lista():
        if ps.get("id") == id_ps:
            return ps
    return None


# ---------------------------------------------------------------------------
# API PUBBLICA DEL MODULO
# ---------------------------------------------------------------------------

def get_affollamento(osp: dict) -> Optional[dict]:
    """Snapshot affollamento reale per un ospedale, nella forma dell'app.

    Ritorna None se non abbinabile / PS chiuso / API non disponibile:
    il chiamante applica il fallback che preferisce.
    """
    if not disponibile():
        return None

    id_ps = _match_id(osp)
    if not id_ps:
        return None

    # Dati dalla LISTA (una sola chiamata condivisa): indice di affollamento,
    # stato aperto/chiuso, ultimo aggiornamento. Sempre disponibili.
    entry = _lista_entry(id_ps)
    if entry and str(entry.get("aperto", "true")).lower() in ("false", "0", "no"):
        return None   # PS chiuso → nessun dato valido (policy)

    # DETTAGLIO: OPZIONALE. Dà la scomposizione per codice triage, ma se la
    # chiamata non è disponibile (es. endpoint 404) NON si blocca tutto:
    # l'indice di affollamento arriva comunque dalla LISTA.
    det = _get_dettaglio(id_ps)

    codici = None
    n_pazienti = None
    n_trattamento = None
    ultimo = entry.get("ultimoAgg") if entry else None

    # Indice di affollamento: fonte CANONICA = LISTA (campo indAff, valorizzato
    # per la maggior parte dei PS). Il DETTAGLIO v2 non lo calcola in modo
    # affidabile (spesso 0), quindi la LISTA ha la priorità.
    indAff = _num(entry.get("indAff")) if entry else None

    if det:
        if str(det.get("aperto", "true")).lower() in ("false", "0", "no"):
            return None
        # Scomposizione per codice triage: dal campo output.codici del v2,
        # una lista [{colore, attesa, trat}, ...] con sigle a una lettera.
        # Costruiamo il dizionario SOLO con i colori realmente esposti dal PS
        # (4 o 5 codici a seconda della scala usata), ordinati per gravità.
        # 'attesa' = pazienti in sala d'attesa (i badge in card);
        # 'trat'   = pazienti già in trattamento dentro il PS (indicatore
        #            chiave del carico reale: spiega perché l'indice può
        #            essere alto anche con la sala d'attesa quasi vuota).
        codici_raw = det.get("codici") or []
        if codici_raw:
            conteggi: dict = {}
            tot_trat = 0
            for c in codici_raw:
                if not isinstance(c, dict):
                    continue
                etichetta = _SIGLA_COLORE.get(str(c.get("colore", "")).strip().upper())
                if not etichetta:
                    continue
                try:
                    n = int(c.get("attesa") or 0)
                except (TypeError, ValueError):
                    n = 0
                try:
                    t = int(c.get("trat") or 0)
                except (TypeError, ValueError):
                    t = 0
                conteggi[etichetta] = conteggi.get(etichetta, 0) + max(0, n)
                tot_trat += max(0, t)
            if conteggi:
                codici = {col: conteggi[col]
                          for col in _ORDINE_COLORI if col in conteggi}
                n_pazienti = sum(codici.values())
                n_trattamento = tot_trat
        ultimo = det.get("ultimoAgg") or ultimo
        if indAff is None:                 # fallback estremo
            indAff = _num(det.get("indAff"))

    if indAff is None:
        return None   # senza indice non inventiamo un valore

    # Normalizzazione sulla scala dell'app: IA/100
    #   IA 50 → 0.50 (soglia "affollato"), IA 100 → 1.00 ("sovraffollato")
    return {
        "codici": codici,               # None se il DETTAGLIO non è disponibile
        "n_pazienti": n_pazienti,       # totale in sala d'attesa (None idem)
        "n_trattamento": n_trattamento, # totale già in trattamento (None idem)
        "affollamento": round(indAff / 100.0, 2),
        "indAff": int(round(indAff)),
        "ultimo_agg": ultimo,
    }


# ---------------------------------------------------------------------------
# DIAGNOSTICA — esegui:  python affollamento_e015.py
# Verifica credenziali, token, chiamata LISTA e abbinamento di ogni ospedale.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    try:
        from dotenv import load_dotenv
        load_dotenv()
        # rileggo le variabili dopo aver caricato il .env
        CONSUMER_KEY    = os.getenv("E015_CONSUMER_KEY")
        CONSUMER_SECRET = os.getenv("E015_CONSUMER_SECRET")
        PS_SCOPES       = os.getenv("E015_PS_SCOPES", "")
    except Exception:
        pass

    print("=" * 64)
    print("  DIAGNOSTICA API E015 — Situazione Pronto Soccorso")
    print("=" * 64)

    print("1) Credenziali nel .env :", "OK" if disponibile() else "MANCANTI")
    if not disponibile():
        print("   → servono E015_CONSUMER_KEY, E015_CONSUMER_SECRET, E015_PS_SCOPES")
        raise SystemExit(1)

    print("2) Token di accesso     :", "OK" if _get_token() else "ERRORE (vedi sopra)")
    if not _get_token():
        raise SystemExit(1)

    lista = _get_lista()
    print(f"3) LISTA PS dall'API    : {len(lista)} pronto soccorso")
    if not lista:
        print("   → LISTA vuota o chiamata fallita (vedi errori [E015] sopra)")
        raise SystemExit(1)
    # salvo la LISTA completa (id, nome, comune, coord, indAff): utile per
    # rivedere gli abbinamenti o costruire e015_id_map.json dai dati veri
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "e015_lista_dump.json"), "w", encoding="utf-8") as f:
            json.dump(lista, f, ensure_ascii=False, indent=2)
        print("   (LISTA salvata in e015_lista_dump.json)")
    except Exception:
        pass

    # Ospedali dell'app (stesso file usato da main.py)
    docs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ospedali_docs.json")
    try:
        with open(docs, encoding="utf-8") as f:
            ospedali = json.load(f)
    except Exception as exc:
        print(f"   Impossibile leggere ospedali_docs.json: {exc}")
        raise SystemExit(1)

    # Le Case di Comunità non hanno PS: non sono nella LISTA, le escludo
    osp_ps = [o for o in ospedali if not o.get("Nome", "").upper().startswith("CASA DI")]
    print(f"4) Ospedali da abbinare : {len(osp_ps)} (escluse Case di Comunità)\n")

    ok, verificare, non_abbinati = [], [], []
    for o in osp_ps:
        info = _match_dettaglio(o)
        if info["id"] and (info["distanza_km"] or 0) <= 1.0:
            ok.append((o, info))
        elif info["id"]:
            verificare.append((o, info))
        else:
            non_abbinati.append((o, info))

    print(f"   ✓ Abbinati sicuri (≤1 km)      : {len(ok)}")
    print(f"   ⚠ Abbinati DA VERIFICARE (>1km): {len(verificare)}")
    for o, info in verificare:
        print(f"       - {o.get('Nome')}  →  {info['nome_api']}  "
              f"({info['distanza_km']} km)  [id: {info.get('id')}]")
    print(f"   ✗ NON abbinati                 : {len(non_abbinati)}"
          f"  (possibili ospedali senza PS)")
    for o, info in non_abbinati:
        extra = (f"  [più vicino: {info['nome_api']} a {info['distanza_km']} km, "
                 f"id: {info.get('id_vicino')}]" if info.get("nome_api") else "")
        print(f"       - {o.get('Nome')}{extra}")

    # Sonda endpoint DETTAGLIO: prova gli URL candidati e mostra lo stato,
    # così individui quello giusto (da mettere in E015_DETTAGLIO_URL nel .env).
    if ok:
        id_prova = ok[0][1]["id"]
        print(f"\n5) Sonda endpoint DETTAGLIO (id di prova = {id_prova}):")
        base = "https://api.servizirl.it/c/servizi.rl/siss/euol"
        candidati = [
            f"{base}/v2.0.0/v3/dettaglio-pronto-soccorso",
            f"{base}/v3.0.0/v3/dettaglio-pronto-soccorso",
            f"{base}/v2.0.0/v2/dettaglio-pronto-soccorso",
            f"{base}/v3.0.0/v2/dettaglio-pronto-soccorso",
        ]
        h = _headers()
        url_ok = None
        for url in candidati:
            try:
                r = _post(url, headers=h, json={"id": id_prova})
                esito = "200 OK" if r.status_code == 200 else str(r.status_code)
                if r.status_code == 200 and url_ok is None:
                    url_ok = url
            except Exception as exc:
                esito = f"errore ({exc})"
            print(f"     [{esito:>10}]  {url}")
        if url_ok:
            print(f"     → funziona: metti nel .env  E015_DETTAGLIO_URL={url_ok}")
        else:
            print("     → nessun endpoint DETTAGLIO risponde 200: l'app userà")
            print("       comunque l'indice di affollamento dalla LISTA (senza")
            print("       la scomposizione per codice triage).")

        # Salvo un DETTAGLIO grezzo (per ispezionare lo schema v2 del triage)
        try:
            h = _headers()
            rr = _post(DETTAGLIO_URL, headers=h, json={"id": id_prova})
            raw = rr.json()
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "e015_dettaglio_sample.json"), "w",
                      encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False, indent=2)
            print("   (DETTAGLIO grezzo salvato in e015_dettaglio_sample.json)")
            if isinstance(raw, dict):
                print("     chiavi risposta:", list(raw.keys()))
                out = raw.get("output")
                if isinstance(out, dict):
                    print("     chiavi output :", list(out.keys()))
        except Exception as exc:
            print(f"   dump DETTAGLIO fallito: {exc}")

        o, _info = ok[0]
        print(f"\n6) Esempio affollamento reale — {o.get('Nome')}:")
        print(json.dumps(get_affollamento(o), indent=2, ensure_ascii=False))
    print("\nFine diagnostica.")
