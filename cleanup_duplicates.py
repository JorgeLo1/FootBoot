#!/usr/bin/env python3
"""
cleanup_duplicates.py
Elimina los archivos sin guión bajo (01_, 02_, etc.) que eran
copias exactas de los archivos _01_, _02_, etc.
Ejecutar una sola vez después de actualizar el proyecto.
"""
import os
from pathlib import Path

SRC = Path(__file__).parent / "src"

# Archivos duplicados a eliminar (sin guión bajo = versión vieja)
DUPLICATES = [
    SRC / "01_data_collector.py",
    SRC / "02_feature_builder.py",
    SRC / "03_model_engine.py",
    SRC / "04_value_detector.py",
    SRC / "05_result_updater.py",
    SRC / "data_collector_utils.py",  # reemplazado por utils.py
]

for path in DUPLICATES:
    if path.exists():
        path.unlink()
        print(f"✓ Eliminado: {path.name}")
    else:
        print(f"  No existe (ya limpio): {path.name}")

print("\nLimpieza completada.")
print("Módulos activos en src/:")
for f in sorted(SRC.glob("*.py")):
    print(f"  {f.name}")