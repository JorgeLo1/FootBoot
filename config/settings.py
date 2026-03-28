import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# ─── API KEYS ────────────────────────────────────────────────────────────────
FOOTBALL_DATA_ORG_KEY = os.getenv("FOOTBALL_DATA_ORG_KEY", "")   # football-data.org (free, permanente)
TELEGRAM_TOKEN        = os.getenv("TELEGRAM_TOKEN",   "TU_BOT_TOKEN_AQUI")
TELEGRAM_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID", "TU_CHAT_ID_AQUI")
SUPABASE_URL          = os.getenv("SUPABASE_URL",     "TU_SUPABASE_URL_AQUI")
SUPABASE_KEY          = os.getenv("SUPABASE_KEY",     "TU_SUPABASE_ANON_KEY_AQUI")

# Mantenemos API_FOOTBALL_KEY como fallback opcional para result_updater
API_FOOTBALL_KEY      = os.getenv("API_FOOTBALL_KEY", "")

# ─── TEMPORADA DINÁMICA ───────────────────────────────────────────────────────
def current_season() -> int:
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1

CURRENT_SEASON = current_season()

def _build_seasons(n: int = 5) -> list[str]:
    start   = current_season()
    seasons = []
    for i in range(n):
        y1 = start - i
        y2 = y1 + 1
        seasons.append(f"{str(y1)[2:]}{str(y2)[2:]}")
    return seasons

FOOTBALL_DATA_SEASONS = _build_seasons(5)

# ─── LIGAS ACTIVAS ────────────────────────────────────────────────────────────
# league_id (football-data.org) → (nombre, fd_code Football-Data.co.uk, fdorg_id)
LIGAS = {
    39:  ("Premier League", "E0", 2021),
    140: ("La Liga",        "SP1", 2014),
    78:  ("Bundesliga",     "D1",  2002),
    135: ("Serie A",        "I1",  2019),
    61:  ("Ligue 1",        "F1",  2015),
    88:  ("Eredivisie",     "N1",  2003),
    94:  ("Primeira Liga",  "P1",  2017),
}

# ─── UMBRALES DE CONFIANZA ────────────────────────────────────────────────────
UMBRAL_EDGE_ALTA   = 8.0
UMBRAL_EDGE_MEDIA  = 4.0
UMBRAL_PROB_ALTA   = 0.62
UMBRAL_PROB_MEDIA  = 0.55
MIN_PARTIDOS_ALTA  = 30
MIN_PARTIDOS_MEDIA = 15
KELLY_FRACCION     = 0.25

# ─── MODELO ───────────────────────────────────────────────────────────────────
VENTANA_FORMA     = 10
LAMBDA_DECAY      = 0.02
XGB_N_ESTIMATORS  = 300
XGB_MAX_DEPTH     = 4
XGB_LEARNING_RATE = 0.05
RANDOM_SEED       = 42

# ─── RATE LIMITS ─────────────────────────────────────────────────────────────
API_FOOTBALL_DAILY_LIMIT = 100
API_USAGE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "api_usage.json"
)

# ─── RUTAS ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW       = os.path.join(BASE_DIR, "data", "raw")
DATA_STATSBOMB = os.path.join(BASE_DIR, "data", "statsbomb")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
LOGS_DIR       = os.path.join(BASE_DIR, "logs")

# ─── URLs EXTERNAS ────────────────────────────────────────────────────────────
FOOTBALL_DATA_URL    = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_DATA_ORG_URL= "https://api.football-data.org/v4"
OPENMETEO_URL        = "https://api.open-meteo.com/v1/forecast"
CLUBELO_URL          = "http://api.clubelo.com"