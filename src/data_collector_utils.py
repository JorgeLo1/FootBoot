"""
data_collector_utils.py
Utilidades compartidas entre módulos — evita imports circulares.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src._01_data_collector import get_weather_for_fixture

__all__ = ["get_weather_for_fixture"]
