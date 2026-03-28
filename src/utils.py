"""
utils.py
Utilidades compartidas: clima, rate-limiting de API, normalización.
Centraliza código que antes estaba duplicado en 3 módulos distintos.
"""

import os
import json
import logging
import requests
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

# ─── ESTADIOS ─────────────────────────────────────────────────────────────────
ESTADIOS: dict[str, tuple[float, float]] = {
    "Arsenal":           (51.5549, -0.1084),
    "Chelsea":           (51.4816, -0.1910),
    "Manchester City":   (53.4831, -2.2004),
    "Manchester United": (53.4631, -2.2913),
    "Liverpool":         (53.4308, -2.9608),
    "Tottenham":         (51.6043, -0.0665),
    "Aston Villa":       (52.5090, -1.8847),
    "Newcastle":         (54.9756, -1.6218),
    "West Ham":          (51.5386, -0.0164),
    "Brighton":          (50.8618, -0.0834),
    "Real Madrid":       (40.4530, -3.6883),
    "Barcelona":         (41.3809,  2.1228),
    "Atletico Madrid":   (40.4361, -3.5996),
    "Sevilla":           (37.3840, -5.9706),
    "Bayern Munich":     (48.2188, 11.6248),
    "Borussia Dortmund": (51.4926,  7.4519),
    "Bayer Leverkusen":  (51.0330,  7.0024),
    "RB Leipzig":        (51.3457, 12.3484),
    "Juventus":          (45.1096,  7.6412),
    "AC Milan":          (45.4781,  9.1240),
    "Inter":             (45.4781,  9.1240),
    "Napoli":            (40.8279, 14.1932),
    "Paris SG":          (48.8414,  2.2530),
    "Lyon":              (45.7654,  4.9825),
    "Ajax":              (52.3143,  4.9419),
    "Porto":             (41.1612, -8.5833),
    "Benfica":           (38.7526, -9.1846),
}
DEFAULT_COORDS = (51.5074, -0.1278)  # Londres como fallback


def get_weather_for_fixture(home_team: str, match_datetime: str) -> dict:
    """
    Obtiene pronóstico del clima para un partido usando Open-Meteo (sin API key).
    Retorna dict con temperatura, lluvia y viento.
    """
    from config.settings import OPENMETEO_URL

    coords = ESTADIOS.get(home_team, DEFAULT_COORDS)
    lat, lon = coords

    try:
        dt = datetime.fromisoformat(str(match_datetime).replace("Z", "+00:00"))
        ds = dt.strftime("%Y-%m-%d")

        r = requests.get(
            OPENMETEO_URL,
            params={
                "latitude":    lat,
                "longitude":   lon,
                "timezone":    "auto",
                "daily": "precipitation_sum,windspeed_10m_max,temperature_2m_max",
                "forecast_days": 3,
                "start_date":  ds,
                "end_date":    ds,
            },
            timeout=8,
        )
        r.raise_for_status()
        d = r.json().get("daily", {})
        prec = (d.get("precipitation_sum") or [0])[0] or 0
        wind = (d.get("windspeed_10m_max") or [10])[0] or 10
        temp = (d.get("temperature_2m_max") or [15])[0] or 15
        return {
            "temp_max":      float(temp),
            "precipitation": float(prec),
            "wind_max":      float(wind),
            "rain_flag":     int(prec > 2),
            "wind_flag":     int(wind > 30),
        }
    except Exception as e:
        log.warning(f"Error clima para {home_team}: {e}")
        return {"temp_max": 15.0, "precipitation": 0.0, "wind_max": 10.0,
                "rain_flag": 0, "wind_flag": 0}


# ─── RATE LIMITER PARA API-FOOTBALL ─────────────────────────────────────────

class ApiRateLimiter:
    """
    Trackea el uso diario de la API de Football (free tier: 100 req/día).
    Persiste el contador en JSON para sobrevivir entre ejecuciones del cron.
    """
    def __init__(self, usage_file: str, daily_limit: int = 100):
        self.usage_file  = usage_file
        self.daily_limit = daily_limit
        self._state      = self._load()

    def _load(self) -> dict:
        today = str(date.today())
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file) as f:
                    state = json.load(f)
                # Resetear si es un día nuevo
                if state.get("date") != today:
                    return {"date": today, "count": 0}
                return state
        except Exception:
            pass
        return {"date": today, "count": 0}

    def _save(self):
        os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
        with open(self.usage_file, "w") as f:
            json.dump(self._state, f)

    @property
    def remaining(self) -> int:
        return max(0, self.daily_limit - self._state["count"])

    def can_request(self, n: int = 1) -> bool:
        return self._state["count"] + n <= self.daily_limit

    def consume(self, n: int = 1):
        """Registra N requests consumidos."""
        self._state["count"] += n
        self._save()
        log.debug(f"API Football: {self._state['count']}/{self.daily_limit} requests usados hoy")

    def status(self) -> str:
        return (f"API-Football: {self._state['count']}/{self.daily_limit} "
                f"requests usados ({self.remaining} restantes)")