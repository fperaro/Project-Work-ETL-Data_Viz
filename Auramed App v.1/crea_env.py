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
E015_CONSUMER_KEY={e015_key}
E015_CONSUMER_SECRET={e015_secret}
E015_PS_SCOPES={e015_scopes}
""".format(
    openai      = input("OpenAI API key: ").strip(),
    tomtom      = input("TomTom API key: ").strip(),
    uri         = input("Neo4j URI: ").strip(),
    user        = input("Neo4j user: ").strip(),
    pwd         = input("Neo4j password: ").strip(),
    db          = input("Neo4j database: ").strip(),
    e015_key    = input("E015 Consumer Key: ").strip(),
    e015_secret = input("E015 Consumer Secret: ").strip(),
    e015_scopes = input("E015 scope PS (separati da spazio): ").strip(),
)

with open(".env", "w", encoding="utf-8") as f:
    f.write(env)

print("\n✓ Creato .env — ricordati di aggiungerlo a .gitignore!")
