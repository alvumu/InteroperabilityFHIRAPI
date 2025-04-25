#!/usr/bin/env bash

# 1. Crear venv en .venv
python3 -m venv .venv                              # :contentReference[oaicite:0]{index=0}

# 2. Activar
source .venv/bin/activate                          # :contentReference[oaicite:1]{index=1}

# 3. Actualizar pip e instalar deps
pip install --upgrade pip                          # :contentReference[oaicite:2]{index=2}
pip install -r requirements.txt
