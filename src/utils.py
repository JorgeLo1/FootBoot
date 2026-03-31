"""
utils.py
Utilidades compartidas: clima, rate-limiting de API, normalización.

FIXES v3:
  - Añadidos estadios de ligas LATAM (col.1, arg.1, bra.1, mex.1, libertadores)
    para que Open-Meteo reciba coordenadas reales en vez del fallback Londres.
  - Open-Meteo: si la fecha es futura (> 16 días), usa climatología en vez
    de pronóstico (evita el 400 Bad Request).

FIXES v4:
  - Open-Meteo 400 Bad Request para hoy/mañana: usar forecast_days sin
    start_date/end_date cuando days_ahead <= 1; indexar el array por days_ahead.
"""

import os
import json
import logging
import requests
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

# ─── ESTADIOS ─────────────────────────────────────────────────────────────────
# Formato: "Nombre exacto del equipo en ESPN" → (lat, lon)
ESTADIOS: dict[str, tuple[float, float]] = {
    # ── Premier League ──────────────────────────────────────────────────────
    "Arsenal":                   (51.5549,  -0.1084),
    "Chelsea":                   (51.4816,  -0.1910),
    "Manchester City":           (53.4831,  -2.2004),
    "Manchester United":         (53.4631,  -2.2913),
    "Liverpool":                 (53.4308,  -2.9608),
    "Tottenham":                 (51.6043,  -0.0665),
    "Aston Villa":               (52.5090,  -1.8847),
    "Newcastle":                 (54.9756,  -1.6218),
    "West Ham":                  (51.5386,  -0.0164),
    "Brighton":                  (50.8618,  -0.0834),
    "Brentford":                 (51.4882,  -0.2887),
    "Fulham":                    (51.4749,  -0.2217),
    "Crystal Palace":            (51.3983,  -0.0855),
    "Everton":                   (53.4388,  -2.9661),
    "Wolves":                    (52.5900,  -2.1302),
    "Nottingham Forest":         (52.9399,  -1.1328),
    "Leicester":                 (52.6204,  -1.1420),
    "Southampton":               (50.9058,  -1.3914),
    "Ipswich":                   (52.0550,   1.1450),
    # ── La Liga ─────────────────────────────────────────────────────────────
    "Real Madrid":               (40.4530,  -3.6883),
    "Barcelona":                 (41.3809,   2.1228),
    "Atletico Madrid":           (40.4361,  -3.5996),
    "Sevilla":                   (37.3840,  -5.9706),
    "Real Betis":                (37.3561,  -5.9817),
    "Valencia":                  (39.4748,  -0.3584),
    "Athletic Club":             (43.2642,  -2.9495),
    "Real Sociedad":             (43.3015,  -2.0011),
    "Villarreal":                (39.9444,  -0.1032),
    "Osasuna":                   (42.7966,  -1.6367),
    "Celta Vigo":                (42.2117,  -8.7347),
    "Rayo Vallecano":            (40.3914,  -3.6605),
    "Getafe":                    (40.3261,  -3.7136),
    "Girona":                    (41.9831,   2.8171),
    "Las Palmas":                (28.1001, -15.4443),
    "Mallorca":                  (39.5901,   2.6614),
    "Espanyol":                  (41.3480,   2.0764),
    "Leganés":                   (40.3322,  -3.7641),
    "Valladolid":                (41.6521,  -4.7286),
    "Alavés":                    (42.8474,  -2.6724),
    # ── Bundesliga ──────────────────────────────────────────────────────────
    "Bayern Munich":             (48.2188,  11.6248),
    "Borussia Dortmund":         (51.4926,   7.4519),
    "Bayer Leverkusen":          (51.0330,   7.0024),
    "RB Leipzig":                (51.3457,  12.3484),
    "Eintracht Frankfurt":       (50.0686,   8.6455),
    "VfB Stuttgart":             (48.7923,   9.2323),
    "SC Freiburg":               (47.9973,   7.8975),
    "Hoffenheim":                (49.2384,   8.8893),
    "Mainz":                     (49.9843,   8.2242),
    "Borussia Mönchengladbach":  (51.1748,   6.3856),
    "Werder Bremen":             (53.0665,   8.8381),
    "Augsburg":                  (48.3236,  10.8863),
    "Union Berlin":              (52.4571,  13.5682),
    "Wolfsburg":                 (52.4318,  10.8032),
    "VfL Bochum":                (51.4900,   7.2363),
    "Heidenheim":                (48.6827,  10.1399),
    "Holstein Kiel":             (54.3380,  10.1310),
    # ── Serie A ─────────────────────────────────────────────────────────────
    "Juventus":                  (45.1096,   7.6412),
    "AC Milan":                  (45.4781,   9.1240),
    "Inter":                     (45.4781,   9.1240),
    "Napoli":                    (40.8279,  14.1932),
    "Roma":                      (41.8339,  12.4878),
    "Lazio":                     (41.9343,  12.4544),
    "Fiorentina":                (43.7808,  11.2823),
    "Atalanta":                  (45.7087,   9.6797),
    "Torino":                    (45.0408,   7.6502),
    "Bologna":                   (44.4898,  11.3133),
    "Udinese":                   (46.0826,  13.2019),
    "Sassuolo":                  (44.5571,  10.8898),
    "Sampdoria":                 (44.4168,   8.9572),
    "Monza":                     (45.5849,   9.2703),
    "Lecce":                     (40.3518,  18.1842),
    "Cagliari":                  (39.2095,   9.1277),
    "Venezia":                   (45.4608,  12.3155),
    "Como":                      (45.8073,   9.0862),
    "Parma":                     (44.7979,  10.3330),
    "Verona":                    (45.4386,  10.9916),
    # ── Ligue 1 ─────────────────────────────────────────────────────────────
    "Paris SG":                  (48.8414,   2.2530),
    "Lyon":                      (45.7654,   4.9825),
    "Marseille":                 (43.2699,   5.3958),
    "Monaco":                    (43.7276,   7.4151),
    "Lille":                     (50.6120,   3.1302),
    "Rennes":                    (48.1076,  -1.7127),
    "Nice":                      (43.7058,   7.2733),
    "Lens":                      (50.4326,   2.8224),
    "Nantes":                    (47.2568,  -1.5254),
    "Strasbourg":                (48.5601,   7.7564),
    "Brest":                     (48.4185,  -4.4629),
    "Montpellier":               (43.6227,   3.8159),
    "Toulouse":                  (43.5824,   1.4342),
    "Reims":                     (49.2437,   4.0312),
    "Auxerre":                   (47.7898,   3.5710),
    "Saint-Étienne":             (45.4609,   4.3900),
    "Angers":                    (47.4621,  -0.5479),
    "Le Havre":                  (49.4979,   0.1198),
    # ── Eredivisie ──────────────────────────────────────────────────────────
    "Ajax":                      (52.3143,   4.9419),
    "PSV":                       (51.4416,   5.4674),
    "Feyenoord":                 (51.8935,   4.5237),
    "AZ":                        (52.6124,   4.7510),
    "Twente":                    (52.2342,   6.8740),
    "Utrecht":                   (52.0786,   5.1352),
    "Groningen":                 (53.2139,   6.5757),
    "Vitesse":                   (51.9653,   5.8912),
    # ── Primeira Liga ───────────────────────────────────────────────────────
    "Porto":                     (41.1612,  -8.5833),
    "Benfica":                   (38.7526,  -9.1846),
    "Sporting CP":               (38.7612,  -9.1600),
    "Braga":                     (41.5753,  -8.4040),
    "Vitória SC":                (41.4448,  -8.3020),
    "Famalicão":                 (41.4080,  -8.5177),
    "Rio Ave":                   (41.3741,  -8.7389),
    # ── Liga BetPlay (col.1) ────────────────────────────────────────────────
    "Atlético Nacional":         ( 6.2549, -75.5944),
    "Millonarios":               ( 4.6473, -74.0962),
    "América de Cali":           ( 3.4291, -76.5407),
    "Deportivo Cali":            ( 3.4291, -76.5407),
    "Junior":                    (10.9685, -74.8023),
    "Santa Fe":                  ( 4.6473, -74.0962),
    "Deportes Tolima":           ( 4.4149, -75.2101),
    "Once Caldas":               ( 5.0645, -75.5040),
    "Deportivo Pasto":           ( 1.2136, -77.2811),
    "Jaguares de Córdoba":       ( 8.7479, -75.8814),
    "Boyacá Chicó":              ( 5.5353, -73.3671),
    "Envigado":                  ( 6.1673, -75.5902),
    "La Equidad":                ( 4.6473, -74.0962),
    "Bucaramanga":               ( 7.1264, -73.1198),
    "Cúcuta Deportivo":          ( 7.8939, -72.5078),
    "Patriotas":                 ( 5.5353, -73.3671),
    "Alianza FC":                (13.6929, -89.2182),
    "Rionegro Águilas":          ( 6.1553, -75.3733),
    "Deportivo Pereira":         ( 4.8087, -75.6906),
    "Independiente Medellín":    ( 6.2549, -75.5944),
    "Águilas Doradas":           ( 6.5273, -74.6568),
    # ── Liga Profesional Argentina (arg.1) ──────────────────────────────────
    "River Plate":               (-34.5451, -58.4496),
    "Boca Juniors":              (-34.6354, -58.3655),
    "Racing Club":               (-34.6645, -58.3673),
    "Independiente":             (-34.6592, -58.3619),
    "San Lorenzo":               (-34.6289, -58.4398),
    "Huracán":                   (-34.6454, -58.4042),
    "Vélez Sársfield":           (-34.6350, -58.5270),
    "Estudiantes":               (-34.9215, -57.9544),
    "Talleres":                  (-31.4201, -64.1888),
    "Belgrano":                  (-31.3994, -64.2021),
    "Defensa y Justicia":        (-34.7012, -58.3668),
    "Lanús":                     (-34.7038, -58.3891),
    "Banfield":                  (-34.7414, -58.3983),
    "Gimnasia La Plata":         (-34.9215, -57.9544),
    "Newell's Old Boys":         (-32.9515, -60.6603),
    "Rosario Central":           (-32.9488, -60.6411),
    "Unión Santa Fe":            (-31.6333, -60.7000),
    "Colón":                     (-31.6333, -60.7000),
    "Godoy Cruz":                (-32.8908, -68.8272),
    "Tigre":                     (-34.4322, -58.5792),
    "San Martín (T)":            (-26.8241, -65.2226),
    "Atlético Tucumán":          (-26.8241, -65.2226),
    "Platense":                  (-34.5451, -58.4496),
    # ── Brasileirão (bra.1) ─────────────────────────────────────────────────
    "Flamengo":                  (-22.9121, -43.2302),
    "Palmeiras":                 (-23.5270, -46.6795),
    "São Paulo":                 (-23.5993, -46.7228),
    "Corinthians":               (-23.5451, -46.5344),
    "Santos":                    (-23.9605, -46.3322),
    "Fluminense":                (-22.9121, -43.2302),
    "Botafogo":                  (-22.9121, -43.2302),
    "Vasco da Gama":             (-22.9121, -43.2302),
    "Grêmio":                    (-29.9835, -51.1959),
    "Internacional":             (-29.9835, -51.1959),
    "Athletico Paranaense":      (-25.4484, -49.2765),
    "Atlético Mineiro":          (-19.8658, -43.9703),
    "Cruzeiro":                  (-19.8658, -43.9703),
    "Bahia":                     (-12.9714, -38.5014),
    "Vitória":                   (-12.9714, -38.5014),
    "Fortaleza":                 ( -3.7327, -38.5270),
    "Ceará":                     ( -3.7327, -38.5270),
    "Sport Recife":              ( -8.0476, -34.8770),
    "Cuiabá":                    (-15.5961, -56.0963),
    "RB Bragantino":             (-22.9509, -46.5428),
    # ── Liga MX (mex.1) ─────────────────────────────────────────────────────
    "Club América":              (19.3033, -99.1504),
    "Guadalajara":               (20.6597, -103.3159),
    "Cruz Azul":                 (19.3033, -99.1504),
    "Pumas UNAM":                (19.3033, -99.1504),
    "Tigres UANL":               (25.6866,  -100.3161),
    "Monterrey":                 (25.6866, -100.3161),
    "Toluca":                    (19.2826,  -99.6629),
    "Santos Laguna":             (25.5428, -103.4068),
    "León":                      (21.1619, -101.7080),
    "Atlas":                     (20.6597, -103.3159),
    "Necaxa":                    (21.5174,  -99.8855),
    "Pachuca":                   (20.1011, -98.7591),
    "Puebla":                    (19.0413,  -98.2062),
    "Mazatlán":                  (23.2494, -106.4111),
    "Querétaro":                 (20.5888,  -100.3899),
    "FC Juárez":                 (31.7381, -106.4870),
    "San Luis":                  (22.1565,  -100.9855),
    "Tijuana":                   (32.5149, -117.0382),
    # ── Copa Libertadores / Sudamericana (equipos frecuentes) ───────────────
    "Olimpia":                   (-25.2637,  -57.6359),
    "Cerro Porteño":             (-25.2637,  -57.6359),
    "Peñarol":                   (-34.8941,  -56.1648),
    "Nacional":                  (-34.8941,  -56.1648),
    "Colo-Colo":                 (-33.4569,  -70.6483),
    "Universidad de Chile":      (-33.4569,  -70.6483),
    "LDU Quito":                 ( -0.2295,  -78.5243),
    "Barcelona SC":              ( -2.1894,  -79.8891),
    "Sporting Cristal":          (-12.0566,  -77.0878),
    "Alianza Lima":              (-12.0566,  -77.0878),
    "Bolívar":                   (-16.5000,  -68.1500),
    "The Strongest":             (-16.5000,  -68.1500),
    "Caracas FC":                (10.4806,  -66.9036),
    "Flamengo RJ":               (-22.9121,  -43.2302),
}

DEFAULT_COORDS = (51.5074, -0.1278)  # Londres como fallback

# Umbral de alerta de requests restantes
API_ALERT_THRESHOLD = 20

# Open-Meteo soporta pronóstico hasta 16 días adelante
_METEO_MAX_FORECAST_DAYS = 16


def get_weather_for_fixture(home_team: str, match_datetime) -> dict:
    """
    Obtiene pronóstico del clima para un partido usando Open-Meteo (sin API key).

    FIX v4:
    - Si la fecha excede el horizonte (>16 días) devuelve valores neutros (no llama API).
    - Para hoy/mañana (days_ahead <= 1) usa forecast_days sin start/end_date,
      evitando el 400 Bad Request que Open-Meteo lanza cuando start_date == hoy
      en timezones UTC-5 (Colombia, etc.). Se selecciona el índice correcto del array.
    - Para fechas 2-16 días adelante sigue usando start_date/end_date.
    """
    from config.settings import OPENMETEO_URL

    coords   = ESTADIOS.get(home_team, DEFAULT_COORDS)
    lat, lon = coords

    if coords == DEFAULT_COORDS:
        log.debug(
            f"Estadio no encontrado para '{home_team}' — usando coords de Londres. "
            "Añadir al diccionario ESTADIOS en utils.py para mayor precisión."
        )

    # Normalizar match_datetime a objeto date
    try:
        if isinstance(match_datetime, datetime):
            target_date = match_datetime.date()
        elif isinstance(match_datetime, date):
            target_date = match_datetime
        else:
            dt = datetime.fromisoformat(str(match_datetime).replace("Z", "+00:00"))
            target_date = dt.date()
    except Exception:
        target_date = date.today()

    ds = target_date.strftime("%Y-%m-%d")

    # Open-Meteo solo soporta pronóstico hasta 16 días — más allá devuelve 400
    days_ahead = (target_date - date.today()).days
    if days_ahead > _METEO_MAX_FORECAST_DAYS:
        log.debug(f"Fecha {ds} fuera del horizonte de pronóstico — usando valores neutros")
        return {"temp_max": 18.0, "precipitation": 0.0, "wind_max": 12.0,
                "rain_flag": 0, "wind_flag": 0}

    try:
        # Open-Meteo da 400 Bad Request cuando start_date == hoy en algunos
        # timezones. Para partidos de hoy (days_ahead == 0) o mañana (== 1)
        # usamos forecast_days sin start/end_date y seleccionamos el día correcto.
        # Para fechas futuras (2-16 días) sí usamos start_date/end_date.
        if days_ahead <= 1:
            params: dict = {
                "latitude":      lat,
                "longitude":     lon,
                "timezone":      "auto",
                "daily":         "precipitation_sum,windspeed_10m_max,temperature_2m_max",
                "forecast_days": max(2, days_ahead + 1),
            }
        else:
            params = {
                "latitude":      lat,
                "longitude":     lon,
                "timezone":      "auto",
                "daily":         "precipitation_sum,windspeed_10m_max,temperature_2m_max",
                "forecast_days": days_ahead + 1,
                "start_date":    ds,
                "end_date":      ds,
            }

        r = requests.get(OPENMETEO_URL, params=params, timeout=8)
        r.raise_for_status()
        d   = r.json().get("daily", {})
        # Cuando usamos forecast_days sin start_date, el array arranca desde hoy.
        # El día que nos interesa está en el índice days_ahead.
        idx  = days_ahead if days_ahead <= 1 else 0
        prec = (d.get("precipitation_sum") or [0] * (idx + 1))[idx] or 0
        wind = (d.get("windspeed_10m_max") or [10] * (idx + 1))[idx] or 10
        temp = (d.get("temperature_2m_max") or [15] * (idx + 1))[idx] or 15
        return {
            "temp_max":      float(temp),
            "precipitation": float(prec),
            "wind_max":      float(wind),
            "rain_flag":     int(prec > 2),
            "wind_flag":     int(wind > 30),
        }
    except Exception as e:
        log.warning(f"Error clima para {home_team} ({ds}): {e}")
        return {"temp_max": 15.0, "precipitation": 0.0, "wind_max": 10.0,
                "rain_flag": 0, "wind_flag": 0}


# ─── RATE LIMITER PARA API-FOOTBALL ─────────────────────────────────────────

class ApiRateLimiter:
    """
    Trackea el uso diario de la API de Football (free tier: 100 req/día).
    Persiste el contador en JSON para sobrevivir entre ejecuciones del cron.

    Dispara alerta Telegram si los requests restantes caen por debajo
    de API_ALERT_THRESHOLD (20 por defecto).
    """

    def __init__(self, usage_file: str, daily_limit: int = 100):
        self.usage_file  = usage_file
        self.daily_limit = daily_limit
        self._state      = self._load()
        self._alerted    = False

    def _load(self) -> dict:
        today = str(date.today())
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file) as f:
                    state = json.load(f)
                if state.get("date") != today:
                    return {"date": today, "count": 0}
                return state
        except Exception:
            pass
        return {"date": today, "count": 0}

    def _save(self):
        os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
        try:
            with open(self.usage_file, "w") as f:
                json.dump(self._state, f)
        except Exception as e:
            log.warning(f"No se pudo guardar estado de rate limiter: {e}")

    @property
    def remaining(self) -> int:
        return max(0, self.daily_limit - self._state["count"])

    def can_request(self, n: int = 1) -> bool:
        return self._state["count"] + n <= self.daily_limit

    def consume(self, n: int = 1):
        self._state["count"] += n
        self._save()
        log.debug(
            f"API Football: {self._state['count']}/{self.daily_limit} "
            f"requests usados hoy"
        )
        if self.remaining <= API_ALERT_THRESHOLD and not self._alerted:
            self._alerted = True
            msg = (
                f"⚠️ *FOOTBOT — Rate Limit*\n"
                f"API-Football: solo `{self.remaining}` requests restantes hoy.\n"
                f"_Usado: {self._state['count']}/{self.daily_limit}_"
            )
            log.warning(f"Rate limit bajo: {self.remaining} requests restantes")
            try:
                from src.telegram_sender import send_telegram
                send_telegram(msg)
            except Exception:
                pass

    def status(self) -> str:
        return (
            f"API-Football: {self._state['count']}/{self.daily_limit} "
            f"requests usados ({self.remaining} restantes)"
        )