import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# ─── API KEYS ────────────────────────────────────────────────────────────────
FOOTBALL_DATA_ORG_KEY = os.getenv("FOOTBALL_DATA_ORG_KEY", "")
TELEGRAM_TOKEN        = os.getenv("TELEGRAM_TOKEN",   "TU_BOT_TOKEN_AQUI")
TELEGRAM_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID", "TU_CHAT_ID_AQUI")
SUPABASE_URL          = os.getenv("SUPABASE_URL",     "TU_SUPABASE_URL_AQUI")
SUPABASE_KEY          = os.getenv("SUPABASE_KEY",     "TU_SUPABASE_ANON_KEY_AQUI")
API_FOOTBALL_KEY      = os.getenv("API_FOOTBALL_KEY", "")
ESPN_ONLY = os.getenv("ESPN_ONLY", "false").lower() == "true"

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

# ─── LIGAS EUROPEAS (football-data.org + Football-Data.co.uk) ────────────────
# league_id → (nombre, fd_code Football-Data.co.uk, fdorg_id)
LIGAS = {
    39:  ("Premier League", "E0", 2021),
    140: ("La Liga",        "SP1", 2014),
    78:  ("Bundesliga",     "D1",  2002),
    135: ("Serie A",        "I1",  2019),
    61:  ("Ligue 1",        "F1",  2015),
    88:  ("Eredivisie",     "N1",  2003),
    94:  ("Primeira Liga",  "P1",  2017),
}

# ─── LIGAS ESPN (sin API key) ─────────────────────────────────────────────────
# slug_espn → (league_id_interno, nombre_display)
# league_id_interno: rangos 500+ para no colisionar con LIGAS ni COMPETICIONES_NACIONALES
LIGAS_ESPN: dict[str, tuple[int, str]] = {
    # Sudamérica — clubes
    "col.1":                 (501, "Liga BetPlay"),
    "arg.1":                 (502, "Liga Profesional Argentina"),
    "bra.1":                 (503, "Brasileirão Serie A"),
    "chi.1":                 (504, "Primera División Chile"),
    "per.1":                 (505, "Liga 1 Perú"),
    "ecu.1":                 (506, "LigaPro Ecuador"),
    "uru.1":                 (507, "Liga AUF Uruguay"),
    "par.1":                 (508, "Primera División Paraguay"),
    "bol.1":                 (509, "Liga Profesional Bolivia"),
    "ven.1":                 (510, "Primera División Venezuela"),
    # Sudamérica — copas
    "conmebol.libertadores": (511, "Copa Libertadores"),
    "conmebol.sudamericana": (512, "Copa Sudamericana"),
    "conmebol.recopa":       (513, "Recopa Sudamericana"),
    # Europa — backup / ampliación
    "uefa.champions":        (514, "Champions League"),
    "uefa.europa":           (515, "Europa League"),
    "uefa.europa.conf":      (516, "Conference League"),
    # CONCACAF
    "concacaf.champions":    (517, "Concacaf Champions Cup"),
    "mex.1":                 (518, "Liga MX"),
    "usa.1":                 (519, "MLS"),
}

# Slugs activos para predicciones diarias
# (los que tienen volumen suficiente de partidos para entrenar el modelo)
LIGAS_ESPN_ACTIVAS: set[str] = {
    "col.1",
    "arg.1",
    "bra.1",
    "conmebol.libertadores",
    "conmebol.sudamericana",
    "uefa.champions",
    "mex.1",
}

# ─── COMPETICIONES DE SELECCIONES NACIONALES ─────────────────────────────────
# slug_espn → (league_id_interno, nombre_display)
COMPETICIONES_NACIONALES_ESPN: dict[str, tuple[int, str]] = {
    "fifa.worldq.conmebol": (361, "Eliminatorias CONMEBOL"),
    "conmebol.america":     (271, "Copa América"),
    "fifa.world":           (1,   "Copa del Mundo"),
    "uefa.nations":         (5,   "UEFA Nations League"),
    "concacaf.gold":        (30,  "Gold Cup"),
    "fifa.worldq.uefa":     (600, "Eliminatorias UEFA"),
    "fifa.worldq.concacaf": (601, "Eliminatorias CONCACAF"),
    "caf.nations":          (602, "Copa África de Naciones"),
}

# IDs tier-1 de selecciones (competiciones oficiales, no amistosos)
TIER_1_NATIONAL_LEAGUES: set[int] = {361, 271, 1, 4, 5, 30, 600, 601}

# ─── UMBRALES DE CONFIANZA ────────────────────────────────────────────────────
UMBRAL_EDGE_ALTA   = 8.0
UMBRAL_EDGE_MEDIA  = 4.0
UMBRAL_PROB_ALTA   = 0.62
UMBRAL_PROB_MEDIA  = 0.55
MIN_PARTIDOS_ALTA  = 30
MIN_PARTIDOS_MEDIA = 15
KELLY_FRACCION     = 0.25

# Umbrales más bajos para selecciones (menos partidos disponibles)
MIN_PARTIDOS_ALTA_NACIONAL  = 15
MIN_PARTIDOS_MEDIA_NACIONAL = 8

# Umbrales para ligas ESPN sin cuotas reales (más conservador)
UMBRAL_EDGE_ALTA_ESPN  = 10.0
UMBRAL_EDGE_MEDIA_ESPN =  6.0

# ─── MODELO ───────────────────────────────────────────────────────────────────
VENTANA_FORMA          = 10
VENTANA_FORMA_NACIONAL = 8
LAMBDA_DECAY           = 0.02
XGB_N_ESTIMATORS       = 300
XGB_MAX_DEPTH          = 4
XGB_LEARNING_RATE      = 0.05
RANDOM_SEED            = 42

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
FOOTBALL_DATA_URL     = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_DATA_ORG_URL = "https://api.football-data.org/v4"
OPENMETEO_URL         = "https://api.open-meteo.com/v1/forecast"
CLUBELO_URL           = "http://api.clubelo.com"

# ESPN API base URLs
ESPN_SITE_V2  = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_SITE_V2B = "https://site.api.espn.com/apis/v2/sports/soccer"       # standings (fix)
ESPN_CORE_V2  = "https://sports.core.api.espn.com/v2/sports/soccer/leagues"