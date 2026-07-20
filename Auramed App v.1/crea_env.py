#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crea il file .env di AuraMed. Uso: python crea_env.py"""

env = """\
OPENAI_API_KEY={openai}
TOMTOM_API_KEY={tomtom}
NEO4J_URI={uri}
NEO4J_USER={user}
NEO4J_PASSWORD={pwd}
NEO4J_DATABASE={db}
""".format(
    openai = input("OpenAI API key: ").strip(),
    tomtom = input("TomTom API key: ").strip(),
    uri    = input("Neo4j URI: ").strip(),
    user   = input("Neo4j user: ").strip(),
    pwd    = input("Neo4j password: ").strip(),
    db     = input("Neo4j database: ").strip(),
)

with open(".env", "w", encoding="utf-8") as f:
    f.write(env)

print("\n✓ Creato .env — ricordati di aggiungerlo a .gitignore!")
