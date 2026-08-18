#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
indice_adeguatezza_province.py — Indice di adeguatezza clinica provinciale
per le tre fasce di popolazione fragile: bambini, anziani, donne in età
fertile.

Metodologia (concettualmente imparentata con il location quotient usato
in economic complexity, cfr. Hidalgo, "Economic complexity theory and
applications", Nature Reviews Physics 2021, eq. 2: R_cp = X_cp·X/(X_c·X_p),
un rapporto tra presenza osservata e presenza attesa):

Per ogni provincia j e gruppo fragile g (bambini, anziani, donne_fertile):

    1. Presenza clinica  pc_{j,g} = reparti utili a g in j / reparti totali in j
    2. Fragilità         fr_{j,g} = popolazione di g in j / popolazione totale in j
    3. Indicizzazione a media 1: ciascun indicatore viene diviso per la
       propria media regionale (PC_{j,g} = pc_{j,g}/media_j(pc_{.,g}), idem
       per FR), così che 1 rappresenti letteralmente "la media lombarda" e
       PC/FR diventino comparabili pur vivendo su scale grezze diverse.
    4. Adeguatezza       A_{j,g}  = PC_{j,g} / FR_{j,g}
       (>1 = offerta clinica più che proporzionata, rispetto alla media
        regionale, al peso demografico del gruppo; <1 = sotto la media)
    5. Indice composito  A_j = media geometrica di A_{j,bambini}, A_{j,anziani},
       A_{j,donne_fertile}
       (la media geometrica, a differenza dell'aritmetica, penalizza gli
        squilibri forti tra i tre gruppi e preserva 1 come soglia neutra)

ATTENZIONE — approssimazioni dichiarate:
    - "Donne in età fertile" è approssimato con "femmine nella fascia 18-44"
      per limite del file residenti (fasce aggregate 0-17/18-44/45-69/70+),
      quindi perde le 15-17 e le 45-49 anni.
    - La mappatura specialità->gruppo fragile è una scelta di merito,
      esplicitata nella lista SPECIALITA_PER_GRUPPO sotto: include solo le
      specialità direttamente rivolte al gruppo (es. cardiologia "adulta"
      generica non è attribuita agli anziani, solo l'Unità Coronarica).
    - "Reparti totali" di una provincia è la somma delle specialità offerte
      da tutte le sue strutture (conteggio con ripetizione, non specialità
      uniche): una specialità offerta da 5 ospedali pesa 5, coerentemente
      con l'idea di "offerta clinica disponibile sul territorio".

Input attesi (stessa cartella dello script):
    - residenti_lombardia_per_fascia.csv  (sep=';', da aggrega_residenti_lombardia.py)
    - micuro_pulito_finale.csv            (dataset ospedali/specialità)

Output:
    - indice_adeguatezza_province.csv
"""

from __future__ import annotations

import ast
import pandas as pd

# ---------------------------------------------------------------------------
# MAPPATURA SPECIALITÀ -> GRUPPO FRAGILE (validata sui nomi reali del dataset)
# ---------------------------------------------------------------------------

SPECIALITA_PER_GRUPPO: dict[str, list[str]] = {
    "bambini": [
        "PEDIATRIA", "CHIRURGIA PEDIATRICA", "NEONATOLOGIA",
        "TERAPIA INTENSIVA NEONATALE", "TERAPIA INTENSIVA PEDIATRICA",
        "CARDIOLOGIA PEDIATRICA", "CARDIOCHIRURGIA PEDIATRICA",
        "NEFROLOGIA PEDIATRICA", "NEUROCHIRURGIA PEDIATRICA",
        "NEUROPSICHIATRIA INFANTILE", "NEUROPSICHIATRIA DELL'INFANZIA",
        "ONCOEMATOLOGIA PEDIATRICA", "RADIOLOGIA PEDIATRICA",
        "UROLOGIA PEDIATRICA", "DIALISI E TRAPIANTO PEDIATRICO",
        "PRONTO SOCCORSO PEDIATRICO",
        "CHIRURGIA DERMATOLOGICA E DERMATOLOGIA PEDIATRICA",
    ],
    "anziani": [
        "GERIATRIA", "LUNGODEGENTI", "CURE PALLIATIVE - HOSPICE",
        "RIABILITAZIONE NEUROLOGICA", "RIABILITAZIONE CARDIO-RESPIRATORIA",
        "RIABILITAZIONE ORTOPEDICA", "NEUROLOGIA D'URGENZA E STROKE UNIT",
        "UNITÀ CORONARICA - UNITÀ INTENSIVA CARDIOLOGICA",
    ],
    "donne_fertile": [
        "GINECOLOGIA", "OSTETRICIA", "FISIOPATOLOGIA DELLA RIPRODUZIONE - PMA",
        "PMA", "PRONTO SOCCORSO GINECOLOGICO",
        "PRONTO SOCCORSO OSTETRICO E GINECOLOGICO",
    ],
}

# Fasce d'età (dal file residenti) associate a ciascun gruppo.
# "donne_fertile" usa SOLO la colonna Femmine della fascia indicata.
FASCIA_PER_GRUPPO = {
    "bambini": "0-17",
    "anziani": "70+",
    "donne_fertile": "18-44",   # solo componente femminile
}

# Corrispondenza nome-provincia-esteso (file residenti) <-> sigla (file micuro)
PROVINCIA_NOME_TO_SIGLA = {
    "Bergamo": "BG", "Brescia": "BS", "Como": "CO", "Cremona": "CR",
    "Lecco": "LC", "Lodi": "LO", "Mantova": "MN", "Milano": "MI",
    "Monza e della Brianza": "MB", "Pavia": "PV", "Sondrio": "SO",
    "Varese": "VA",
}


# ---------------------------------------------------------------------------
# 1. CARICAMENTO E AGGREGAZIONE POPOLAZIONE PER PROVINCIA
# ---------------------------------------------------------------------------

def carica_popolazione_per_provincia(path: str) -> pd.DataFrame:
    """Ritorna un DataFrame indicizzato per sigla provincia con colonne:
    pop_totale, pop_bambini, pop_anziani, pop_donne_fertile.
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df["Sigla"] = df["Provincia"].map(PROVINCIA_NOME_TO_SIGLA)

    mancanti = df[df["Sigla"].isna()]["Provincia"].unique()
    if len(mancanti):
        raise ValueError(f"Province non mappate a una sigla: {mancanti}")

    pop_totale = df.groupby("Sigla")["Totale"].sum().rename("pop_totale")

    bambini = (df[df["Fascia"] == FASCIA_PER_GRUPPO["bambini"]]
               .groupby("Sigla")["Totale"].sum().rename("pop_bambini"))

    anziani = (df[df["Fascia"] == FASCIA_PER_GRUPPO["anziani"]]
               .groupby("Sigla")["Totale"].sum().rename("pop_anziani"))

    # Donne età fertile: SOLO colonna Femmine della fascia 18-44
    donne = (df[df["Fascia"] == FASCIA_PER_GRUPPO["donne_fertile"]]
             .groupby("Sigla")["Femmine"].sum().rename("pop_donne_fertile"))

    out = pd.concat([pop_totale, bambini, anziani, donne], axis=1)
    return out


# ---------------------------------------------------------------------------
# 2. CARICAMENTO E AGGREGAZIONE REPARTI PER PROVINCIA
# ---------------------------------------------------------------------------

def _parse_aree(s) -> list[str]:
    """Fa il parsing della colonna aree_specialistiche (lista con doppie
    virgolette annidate stile CSV-in-CSV)."""
    if pd.isna(s) or not str(s).strip():
        return []
    s2 = str(s).replace('""', '"')
    try:
        return ast.literal_eval(s2)
    except Exception:
        return []


def carica_reparti_per_provincia(path: str) -> pd.DataFrame:
    """Ritorna un DataFrame indicizzato per sigla provincia con colonne:
    reparti_totali, reparti_bambini, reparti_anziani, reparti_donne_fertile.

    Il conteggio è "con ripetizione": ogni specialità offerta da ogni
    struttura conta 1, quindi una specialità comune a più ospedali pesa
    più volte (coerente con "offerta clinica disponibile sul territorio").
    """
    df = pd.read_csv(path)
    df["_aree"] = df["aree_specialistiche"].apply(_parse_aree)

    righe = []
    for _, row in df.iterrows():
        prov = row["Provincia"]
        for area in row["_aree"]:
            righe.append((prov, area))

    long_df = pd.DataFrame(righe, columns=["Provincia", "Area"])

    reparti_totali = long_df.groupby("Provincia").size().rename("reparti_totali")

    out = {"reparti_totali": reparti_totali}
    for gruppo, specialita in SPECIALITA_PER_GRUPPO.items():
        conteggio = (long_df[long_df["Area"].isin(specialita)]
                     .groupby("Provincia").size()
                     .rename(f"reparti_{gruppo}"))
        out[f"reparti_{gruppo}"] = conteggio

    result = pd.concat(out.values(), axis=1).fillna(0)
    return result


# ---------------------------------------------------------------------------
# 3. CALCOLO DEGLI INDICI
# ---------------------------------------------------------------------------

def calcola_indici(pop: pd.DataFrame, reparti: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce gli indici indicizzando ciascun indicatore (PC e FR) alla
    propria media regionale, così che il valore 1 rappresenti letteralmente
    "la media lombarda" e non un rapporto tra quote di scala diversa.

    Perché indicizzare: le quote grezze PC (specialità dedicate/totali) e
    FR (popolazione del gruppo/totale) vivono su scale strutturalmente
    diverse — PC è tipicamente 5-19%, FR è tipicamente 14-19%, perché solo
    una minoranza delle specialità totali è dedicata a un singolo gruppo
    fragile. Confrontando le quote grezze (PC/FR) l'indice risulterebbe
    quasi sempre <1, non perché l'offerta sia carente ma per un mero
    disallineamento di scala tra numeratore e denominatore.
    Indicizzando PC e FR alla propria media prima di dividerli, il
    confronto diventa "quanto una provincia si discosta dalla media
    lombarda su presenza clinica" vs "quanto si discosta sulla fragilità
    demografica" — entrambi centrati su 1, e il loro rapporto è quindi
    interpretabile in modo pulito: >1 offerta più che proporzionata alla
    media regionale, <1 sotto la media regionale.
    """
    df = pop.join(reparti, how="outer")

    gruppi = list(SPECIALITA_PER_GRUPPO.keys())
    for gruppo in gruppi:
        pc_grezzo = df[f"reparti_{gruppo}"] / df["reparti_totali"]
        fr_grezzo = df[f"pop_{gruppo}"] / df["pop_totale"]

        # Indicizzazione a media 1 (ciascun indicatore sulla propria media
        # regionale, non sulla media dell'altro)
        df[f"PC_{gruppo}"] = pc_grezzo / pc_grezzo.mean()
        df[f"FR_{gruppo}"] = fr_grezzo / fr_grezzo.mean()

        # Adeguatezza: rapporto tra i due indicatori, ora comparabili
        df[f"A_{gruppo}"] = df[f"PC_{gruppo}"] / df[f"FR_{gruppo}"]

    # Indice composito: media geometrica dei tre A_{j,g}
    prodotto = df[[f"A_{g}" for g in gruppi]].prod(axis=1)
    df["A_composito"] = prodotto ** (1 / len(gruppi))

    return df.sort_values("A_composito", ascending=False)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pop = carica_popolazione_per_provincia("residenti_lombardia_per_fascia.csv")
    reparti = carica_reparti_per_provincia("micuro_pulito_finale.csv")
    risultato = calcola_indici(pop, reparti)

    colonne_output = (
        ["pop_totale", "reparti_totali"]
        + [f"A_{g}" for g in SPECIALITA_PER_GRUPPO]
        + ["A_composito"]
    )
    print(risultato[colonne_output].round(2).to_string())

    risultato.to_csv("indice_adeguatezza_province.csv")
    print("\nFile salvato: indice_adeguatezza_province.csv")
    print("\nNota: A > 1 = offerta clinica più che adeguata rispetto al peso")
    print("demografico del gruppo; A < 1 = sotto-offerta relativa.")
