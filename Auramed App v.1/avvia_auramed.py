#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
avvia_auramed.py — Avvia AuraMed e il tunnel pubblico in un colpo solo.

Cosa fa, ogni volta che lo lanci:
  1. avvia l'app  (python main.py)  in locale sulla porta 8080;
  2. avvia il tunnel Cloudflare (cloudflared) che espone la porta 8080
     dietro un URL pubblico https temporaneo;
  3. legge l'output di cloudflared, estrae il link e te lo mostra dentro
     un riquadro ben evidente (e lo copia negli appunti, su Windows);
  4. quando premi Ctrl+C, ferma sia l'app sia il tunnel.

Se cloudflared.exe non è nella cartella, lo scarica in automatico.

Uso:
    python avvia_auramed.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import signal
import threading
import subprocess
import urllib.request
from pathlib import Path

# ── Configurazione ─────────────────────────────────────────────────────────
PORT = 8080
QUI = Path(__file__).resolve().parent
MAIN = QUI / "main.py"

# Nome dell'eseguibile cloudflared per la piattaforma corrente.
_IS_WIN = os.name == "nt"
CF_EXE = QUI / ("cloudflared.exe" if _IS_WIN else "cloudflared")
CF_DOWNLOAD = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    + ("cloudflared-windows-amd64.exe" if _IS_WIN else "cloudflared-linux-amd64")
)

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


# ── Colori/ANSI (best effort) ──────────────────────────────────────────────
def _abilita_ansi() -> None:
    if _IS_WIN:
        os.system("")  # abilita le sequenze ANSI nel terminale di Windows


_R = "\033[0m"; _B = "\033[1m"; _G = "\033[92m"; _CY = "\033[96m"; _Y = "\033[93m"


def _banner_link(url: str) -> None:
    """Stampa il link dentro un riquadro ben visibile."""
    larghezza = max(len(url) + 6, 60)
    riga = "═" * larghezza
    print("\n")
    print(f"{_G}{_B}╔{riga}╗{_R}")
    print(f"{_G}{_B}║{' ' * larghezza}║{_R}")
    etichetta = "  🔗  LINK DA CONDIVIDERE (https, GPS attivo):"
    print(f"{_G}{_B}║{etichetta.ljust(larghezza)}║{_R}")
    print(f"{_G}{_B}║{' ' * larghezza}║{_R}")
    centro = f"  {_CY}{_B}{url}{_R}{_G}{_B}"
    # padding tenendo conto che i codici ANSI non occupano spazio a video
    pad = larghezza - (len(url) + 2)
    print(f"{_G}{_B}║{centro}{' ' * max(pad, 0)}║{_R}")
    print(f"{_G}{_B}║{' ' * larghezza}║{_R}")
    print(f"{_G}{_B}╚{riga}╝{_R}")
    print(f"{_Y}   (Ctrl+C per fermare app e tunnel){_R}\n")


def _copia_appunti(testo: str) -> bool:
    """Copia il link negli appunti (Windows: clip). Ritorna True se riuscito."""
    try:
        if _IS_WIN:
            subprocess.run("clip", input=testo, text=True, check=True)
            return True
    except Exception:
        pass
    return False


# ── Download di cloudflared se assente ─────────────────────────────────────
def _assicura_cloudflared() -> bool:
    if CF_EXE.exists():
        return True
    print(f"{_Y}cloudflared non trovato: lo scarico (~50 MB, una volta sola)…{_R}")
    try:
        urllib.request.urlretrieve(CF_DOWNLOAD, CF_EXE)
        if not _IS_WIN:
            os.chmod(CF_EXE, 0o755)
        print(f"{_G}cloudflared scaricato.{_R}")
        return True
    except Exception as exc:
        print(f"\033[91mImpossibile scaricare cloudflared: {exc}{_R}")
        print("Scaricalo a mano da:\n  " + CF_DOWNLOAD)
        return False


# ── Lettura output cloudflared (estrae il link, poi drena il buffer) ───────
def _sorveglia_cloudflared(proc: subprocess.Popen) -> None:
    trovato = False
    for linea in iter(proc.stdout.readline, ""):
        if not trovato:
            m = _URL_RE.search(linea)
            if m:
                trovato = True
                url = m.group(0)
                copiato = _copia_appunti(url)
                _banner_link(url)
                if copiato:
                    print(f"{_G}   ✔ Link copiato negli appunti — incollalo dove vuoi.{_R}\n")
        # dopo aver trovato l'URL continuo a leggere per non bloccare il
        # processo (buffer pieno), ma resto silenzioso salvo errori evidenti
        elif "ERR" in linea or "error" in linea.lower():
            print(f"\033[91m[cloudflared] {linea.rstrip()}{_R}")
    # se arrivo qui, cloudflared è terminato
    if not trovato:
        print("\033[91m[cloudflared] terminato senza fornire un link. "
              "Controlla la connessione.\033[0m")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    _abilita_ansi()

    if not MAIN.exists():
        print(f"\033[91mNon trovo main.py accanto a questo script ({MAIN}).{_R}")
        return 1
    if not _assicura_cloudflared():
        return 1

    print(f"{_B}Avvio AuraMed sulla porta {PORT}…{_R}")
    # L'app eredita stdout/stderr: vedi normalmente i suoi log (Neo4j, ecc.).
    app_proc = subprocess.Popen([sys.executable, str(MAIN)], cwd=str(QUI))

    # Piccola attesa perché l'app inizi ad ascoltare prima del tunnel.
    time.sleep(3)

    print(f"{_B}Avvio il tunnel Cloudflare…{_R}")
    cf_proc = subprocess.Popen(
        [str(CF_EXE), "tunnel", "--url", f"http://localhost:{PORT}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(QUI), bufsize=1,
    )

    # Il sorvegliante legge l'output del tunnel in un thread dedicato.
    t = threading.Thread(target=_sorveglia_cloudflared, args=(cf_proc,), daemon=True)
    t.start()

    def _spegni(*_a):
        print(f"\n{_Y}Arresto in corso…{_R}")
        for p in (cf_proc, app_proc):
            try:
                p.terminate()
            except Exception:
                pass
        # tempo per una chiusura pulita, poi forza
        for p in (cf_proc, app_proc):
            try:
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print(f"{_G}Fatto. App e tunnel fermati.{_R}")

    try:
        # resto vivo finché uno dei due processi non muore o arriva Ctrl+C
        while True:
            if app_proc.poll() is not None:
                print(f"{_Y}L'app si è chiusa: fermo anche il tunnel.{_R}")
                break
            if cf_proc.poll() is not None:
                print(f"{_Y}Il tunnel si è chiuso: fermo anche l'app.{_R}")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        _spegni()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
