import os

# ─── API KEYS (rellenar con tus credenciales) ────────────────────────────────
API_FOOTBALL_KEY   = os.getenv("API_FOOTBALL_KEY", "TU_API_KEY_AQUI")
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN",   "TU_BOT_TOKEN_AQUI")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "TU_CHAT_ID_AQUI")
SUPABASE_URL       = os.getenv("SUPABASE_URL",      "TU_SUPABASE_URL_AQUI")
SUPABASE_KEY       = os.getenv("SUPABASE_KEY",      "TU_SUPABASE_ANON_KEY_AQUI")

# ─── LIGAS ACTIVAS ────────────────────────────────────────────────────────────
# Formato: {league_id_api_football: (nombre, football_data_codigo)}
LIGAS = {
    39:  ("Premier League",  "E0"),
    140: ("La Liga",         "SP1"),
    78:  ("Bundesliga",      "D1"),
    135: ("Serie A",         "I1"),
    61:  ("Ligue 1",         "F1"),
    88:  ("Eredivisie",      "N1"),
    94:  ("Primeira Liga",   "P1"),
}

# ─── UMBRALES DE CONFIANZA ────────────────────────────────────────────────────
UMBRAL_EDGE_ALTA   = 8.0    # edge% mínimo para confianza ALTA
UMBRAL_EDGE_MEDIA  = 4.0    # edge% mínimo para confianza MEDIA
UMBRAL_PROB_ALTA   = 0.62   # prob mínima para confianza ALTA
UMBRAL_PROB_MEDIA  = 0.55   # prob mínima para confianza MEDIA
MIN_PARTIDOS_ALTA  = 30     # partidos históricos mínimos para ALTA
MIN_PARTIDOS_MEDIA = 15     # partidos históricos mínimos para MEDIA
KELLY_FRACCION     = 0.25   # usar 1/4 del Kelly completo

# ─── MODELO ───────────────────────────────────────────────────────────────────
VENTANA_FORMA      = 10     # últimos N partidos para calcular forma
LAMBDA_DECAY       = 0.02   # factor de decaimiento exponencial por día
XGB_N_ESTIMATORS   = 300
XGB_MAX_DEPTH      = 4
XGB_LEARNING_RATE  = 0.05
RANDOM_SEED        = 42

# ─── RUTAS ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW        = os.path.join(BASE_DIR, "data", "raw")
DATA_STATSBOMB  = os.path.join(BASE_DIR, "data", "statsbomb")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

# ─── FOOTBALL-DATA BASE URL ───────────────────────────────────────────────────
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_DATA_SEASONS = ["2324", "2223", "2122", "2021", "1920"]

# ─── OPEN-METEO ───────────────────────────────────────────────────────────────
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

# ─── CLUBELO ──────────────────────────────────────────────────────────────────
CLUBELO_URL = "http://api.clubelo.com"
