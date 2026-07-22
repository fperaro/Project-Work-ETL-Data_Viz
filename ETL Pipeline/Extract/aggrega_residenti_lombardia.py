#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggrega i 12 file POSAS (residenti per età e sesso dei comuni lombardi)
in un unico CSV con conteggi maschi / femmine / totali:
  - per comune
  - per fascia d'età (0-17, 18-44, 45-69, 70+)
Esclude le righe con Età = 999 (totali di comune) e la riga di nota finale.

Uso:
    python aggrega_residenti_lombardia.py            # legge i CSV dalla cartella corrente
    python aggrega_residenti_lombardia.py <dir_input> <file_output.csv>
"""

import sys
import glob
import os
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

# Se True -> prima fascia "1-17" (esclude i neonati di età 0).
# Se False -> prima fascia "0-17" (include l'età 0; i totali riconciliano
#             con il totale ufficiale del comune, riga Età=999).
ESCLUDI_ETA_0 = False

# Definizione delle fasce: (etichetta, età_min inclusiva, età_max inclusiva)
# L'ultima fascia è aperta verso l'alto (max = None).
_min_prima_fascia = 1 if ESCLUDI_ETA_0 else 0
FASCE = [
    (f"{_min_prima_fascia}-17", _min_prima_fascia, 17),
    ("18-44", 18, 44),
    ("45-69", 45, 69),
    ("70+",   70, None),
]

# Nome del separatore usato in input e in output (i file POSAS usano ';')
SEP = ";"

# Colonna aggiunta da noi (dedotta dal nome file)
COL_PROV = "Provincia"

# Colonne attese nei file POSAS (dopo la riga di titolo)
COL_CODICE = "Codice comune"
COL_COMUNE = "Comune"
COL_ETA    = "Età"
COL_M      = "Totale maschi"
COL_F      = "Totale femmine"
COL_TOT    = "Totale"

ETA_TOTALE_COMUNE = 999  # righe da scartare (totali per comune)


# ---------------------------------------------------------------------------
# FUNZIONI
# ---------------------------------------------------------------------------

def leggi_file_posas(path: str) -> pd.DataFrame:
    """Legge un singolo file POSAS restituendo un DataFrame pulito.

    Gestisce: BOM UTF-8, riga di titolo iniziale (saltata),
    riga di nota finale e altre righe non numeriche (scartate),
    righe di totale comune (Età=999, scartate).
    """
    # skiprows=1 salta la riga di titolo; l'header vero è la riga successiva.
    df = pd.read_csv(
        path,
        sep=SEP,
        skiprows=1,
        encoding="utf-8-sig",   # gestisce il BOM
        dtype=str,              # leggo tutto come stringa e converto io
    )

    # Verifica minima delle colonne attese
    attese = {COL_CODICE, COL_COMUNE, COL_ETA, COL_M, COL_F, COL_TOT}
    mancanti = attese - set(df.columns)
    if mancanti:
        raise ValueError(f"{os.path.basename(path)}: colonne mancanti {mancanti}. "
                         f"Trovate: {list(df.columns)}")

    # Converto Età e i conteggi in numerico. Le righe non numeriche
    # (es. la nota finale "Nota: ...") diventano NaN e vengono scartate.
    for col in (COL_ETA, COL_M, COL_F, COL_TOT):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Scarto righe senza età valida (nota finale / righe sporche)
    df = df.dropna(subset=[COL_ETA]).copy()
    df[COL_ETA] = df[COL_ETA].astype(int)

    # Scarto i totali di comune (Età = 999)
    df = df[df[COL_ETA] != ETA_TOTALE_COMUNE].copy()

    # Conteggi come interi
    for col in (COL_M, COL_F, COL_TOT):
        df[col] = df[col].fillna(0).astype(int)

    # Provincia dedotta dal nome file: POSAS_2026_it_<cod>_<Provincia>.csv
    # (gli underscore nel nome provincia diventano spazi)
    nome = os.path.splitext(os.path.basename(path))[0]
    df[COL_PROV] = " ".join(nome.split("_")[4:])

    return df


def assegna_fascia(eta: int) -> str:
    """Restituisce l'etichetta della fascia per una data età (o None se fuori)."""
    for etichetta, emin, emax in FASCE:
        if eta < emin:
            continue
        if emax is None or eta <= emax:
            return etichetta
    return None  # es. età 0 quando ESCLUDI_ETA_0 = True


def main():
    dir_input = sys.argv[1] if len(sys.argv) > 1 else "."
    file_output = sys.argv[2] if len(sys.argv) > 2 else "residenti_lombardia_per_fascia.csv"

    pattern = os.path.join(dir_input, "POSAS_2026_it_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"Nessun file trovato con pattern: {pattern}")

    print(f"Trovati {len(files)} file da elaborare.")

    # Leggo e concateno tutti i file
    frames = []
    totali_ufficiali = {}  # (codice, comune) -> totale ufficiale (per controllo)
    for path in files:
        df = leggi_file_posas(path)
        frames.append(df)
        print(f"  - {os.path.basename(path):45s} righe valide: {len(df):>7}")

    dati = pd.concat(frames, ignore_index=True)

    # Totale ufficiale per comune = somma di tutte le età (0-100),
    # usato solo come controllo di riconciliazione.
    ctrl = (dati.groupby([COL_CODICE, COL_COMUNE])[COL_TOT]
                .sum().rename("Totale_atteso").reset_index())

    # Assegno la fascia d'età
    dati["Fascia"] = dati[COL_ETA].apply(assegna_fascia)

    # Righe fuori fascia (solo l'età 0 se ESCLUDI_ETA_0 = True) -> scartate
    escluse = dati["Fascia"].isna().sum()
    if escluse:
        print(f"\nNota: {escluse} righe (età 0) escluse perché ESCLUDI_ETA_0 = True.")
    dati = dati.dropna(subset=["Fascia"])

    # Aggregazione per comune + fascia (Provincia prima del Comune)
    out = (dati.groupby([COL_CODICE, COL_PROV, COL_COMUNE, "Fascia"])
                .agg(Maschi=(COL_M, "sum"),
                     Femmine=(COL_F, "sum"),
                     Totale=(COL_TOT, "sum"))
                .reset_index())

    # Ordino le fasce nell'ordine logico (non alfabetico)
    ordine_fasce = [f[0] for f in FASCE]
    out["Fascia"] = pd.Categorical(out["Fascia"], categories=ordine_fasce, ordered=True)
    out = out.sort_values([COL_CODICE, "Fascia"]).reset_index(drop=True)

    # --- Controllo di riconciliazione -------------------------------------
    somma_fasce = (out.groupby([COL_CODICE, COL_COMUNE])["Totale"]
                      .sum().rename("Totale_calcolato").reset_index())
    check = ctrl.merge(somma_fasce, on=[COL_CODICE, COL_COMUNE], how="outer")
    check["diff"] = check["Totale_calcolato"] - check["Totale_atteso"]
    n_discrepanze = (check["diff"].fillna(0) != 0).sum()

    print("\n--- Riepilogo ---")
    print(f"Comuni elaborati        : {out[COL_CODICE].nunique()}")
    print(f"Righe in output         : {len(out)} (comuni x fasce)")
    print(f"Totale residenti (output): {out['Totale'].sum():,}".replace(",", "."))
    if ESCLUDI_ETA_0:
        print("Riconciliazione col totale ufficiale non attesa (età 0 esclusa).")
    else:
        print(f"Comuni con discrepanza vs totale ufficiale: {n_discrepanze}")

    # Salvataggio
    out.to_csv(file_output, sep=SEP, index=False, encoding="utf-8-sig")
    print(f"\nFile scritto: {file_output}")


if __name__ == "__main__":
    main()
