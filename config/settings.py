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
#
# RANGOS:
#   500–519  → LATAM núcleo (ya existentes)
#   520–529  → LATAM extra (nuevo)
#   530–539  → Europa ESPN (ligas domésticas via ESPN, backup de fd.co.uk)
#   540–549  → UEFA copas extra (nuevo)
#   550–559  → CONCACAF extra (nuevo)
LIGAS_ESPN: dict[str, tuple[int, str]] = {

    # ── Sudamérica clubes ─────────────────────────────────────────────────
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

    # ── Sudamérica copas ──────────────────────────────────────────────────
    "conmebol.libertadores": (511, "Copa Libertadores"),
    "conmebol.sudamericana": (512, "Copa Sudamericana"),
    "conmebol.recopa":       (513, "Recopa Sudamericana"),

    # ── Europa (backup / ligas con buena cobertura de cuotas ESPN) ────────
    "uefa.champions":        (514, "Champions League"),
    "uefa.europa":           (515, "Europa League"),
    "uefa.europa.conf":      (516, "Conference League"),

    # ── Europa ligas domésticas vía ESPN ─────────────────────────────────
    # Solo se usan para histórico y cuotas si ESPN_ONLY=true.
    # En modo normal (fd.co.uk activo) se omiten en download_espn_historical.
    "eng.1":                 (530, "Premier League"),
    "esp.1":                 (531, "La Liga"),
    "ger.1":                 (532, "Bundesliga"),
    "ita.1":                 (533, "Serie A"),
    "fra.1":                 (534, "Ligue 1"),
    "ned.1":                 (535, "Eredivisie"),
    "por.1":                 (536, "Primeira Liga"),
    "tur.1":                 (537, "Süper Lig"),
    "sco.1":                 (538, "Scottish Premiership"),
    "bel.1":                 (539, "Pro League Bélgica"),

    # ── CONCACAF ─────────────────────────────────────────────────────────
    "concacaf.champions":    (517, "Concacaf Champions Cup"),
    "mex.1":                 (518, "Liga MX"),
    "usa.1":                 (519, "MLS"),

    # ── Clasificatorias mundialistas y copas continentales ────────────────
    # (datos históricos útiles para modelo de selecciones)
    "bra.2":                 (520, "Brasileirão Serie B"),
    "arg.copa":              (521, "Copa Argentina"),
    "conmebol.america":      (522, "Copa América"),
}

# ── Conjuntos de slugs por región ─────────────────────────────────────────────
# Útil para filtrar en download_espn_historical y en el scheduler.

# Ligas de clubes LATAM con volumen de partidos suficiente para DC propio
SLUGS_LATAM_CLUBES: set[str] = {
    "col.1", "arg.1", "bra.1", "chi.1", "per.1",
    "ecu.1", "uru.1", "par.1", "mex.1",
}

# Copas CONMEBOL — volumen bajo, sin DC propio (requieren backfill completo)
SLUGS_LATAM_COPAS: set[str] = {
    "conmebol.libertadores", "conmebol.sudamericana", "conmebol.recopa",
}

# Ligas europeas que ESPN cubre y que se usan SOLO cuando ESPN_ONLY=true
SLUGS_EU_ESPN: set[str] = {
    "eng.1", "esp.1", "ger.1", "ita.1", "fra.1", "ned.1",
    "por.1", "tur.1", "sco.1", "bel.1",
}

# Ligas UEFA de clubes con cuotas ESPN disponibles
SLUGS_UEFA_COPAS: set[str] = {
    "uefa.champions", "uefa.europa", "uefa.europa.conf",
}

# Slugs activos para predicciones diarias
# (los que tienen volumen suficiente de partidos para entrenar el modelo)
# AMPLIADO: se añaden chi.1, per.1, ecu.1, uru.1, par.1 y las copas UEFA.
LIGAS_ESPN_ACTIVAS: set[str] = {
    # LATAM núcleo — ya funcionaban
    "col.1",
    "arg.1",
    "bra.1",
    "mex.1",
    # LATAM ampliación — suficiente histórico post-backfill
    "chi.1",
    "per.1",
    "ecu.1",
    "uru.1",
    "par.1",
    # Copas sudamericanas
    "conmebol.libertadores",
    "conmebol.sudamericana",
    # UEFA
    "uefa.champions",
    "uefa.europa",
    "uefa.europa.conf",
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
    "concacaf.nations.league": (603, "Concacaf Nations League"),
}

# IDs tier-1 de selecciones (competiciones oficiales, no amistosos)
TIER_1_NATIONAL_LEAGUES: set[int] = {361, 271, 1, 4, 5, 30, 600, 601, 603}

# ─── FEATURE FLAGS ────────────────────────────────────────────────────────────
# Controlan qué fuentes de datos adicionales se activan.
# Se pueden sobreescribir desde .env.

# Activar descarga de lesiones desde ESPN (teams/{id}/injuries)
# Añade feature `home_injured_starters` / `away_injured_starters` al modelo.
ESPN_INJURIES_ENABLED: bool = os.getenv("ESPN_INJURIES_ENABLED", "true").lower() == "true"

# Activar ESPN BPI (win probability pre-partido desde /summary?event={id})
# Añade feature `espn_bpi_home` / `espn_bpi_away` al modelo.
# NOTA: ESPN BPI no devuelve datos para ligas LATAM (confirmado 0/12, 2026-04-02).
# Se desactiva automáticamente para slugs en SLUGS_SIN_BPI para evitar
# llamadas a la API que siempre devuelven 0.
ESPN_BPI_ENABLED: bool = os.getenv("ESPN_BPI_ENABLED", "true").lower() == "true"

# Slugs para los que ESPN BPI no tiene datos — se omite el enriquecimiento BPI
# aunque ESPN_BPI_ENABLED=true. Evita llamadas innecesarias y features con valor 0.
SLUGS_SIN_BPI: set[str] = SLUGS_LATAM_CLUBES | SLUGS_LATAM_COPAS | {
    "conmebol.libertadores", "conmebol.sudamericana", "conmebol.recopa",
    "concacaf.champions", "mex.1", "usa.1",
}

# Activar descarga de standings para feature de posición en tabla
ESPN_STANDINGS_FEATURES: bool = os.getenv("ESPN_STANDINGS_FEATURES", "true").lower() == "true"

# ─── UMBRALES DE CONFIANZA ────────────────────────────────────────────────────
UMBRAL_EDGE_ALTA   = 8.0
UMBRAL_EDGE_MEDIA  = 4.0
UMBRAL_PROB_ALTA   = 0.62
UMBRAL_PROB_MEDIA  = 0.55
MIN_PARTIDOS_ALTA  = 30
MIN_PARTIDOS_MEDIA = 15
KELLY_FRACCION     = 0.25

# ── Nivel BAJA ────────────────────────────────────────────────────────────────
UMBRAL_EDGE_BAJA   = 2.0
UMBRAL_PROB_BAJA   = 0.52
MIN_PARTIDOS_BAJA  = 6

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

# ─── PONDERACIÓN TEMPORAL (time decay) ────────────────────────────────────────
# Controla cuánto "pesan" los partidos más antiguos vs los recientes.
#
# DC_TIME_DECAY_XI (ξ): factor de decaimiento para Dixon-Coles.
#   Cada partido recibe weight = exp(−ξ × días_atrás).
#   ξ = 0.003 → semivida ~231 días (un partido de hace 1 año vale ~33% de hoy).
#   ξ = 0.005 → semivida ~139 días (más agresivo, favorece últimas 2 temporadas).
#   ξ = 0.0   → todos los partidos pesan igual (comportamiento anterior).
#   Rango recomendado para sweep: 0.001 a 0.006 (paso 0.001).
DC_TIME_DECAY_XI: float = float(os.getenv("DC_TIME_DECAY_XI", "0.003"))

# XGB_TIME_DECAY_LAMBDA (λ): factor de decaimiento para sample_weight en XGBoost.
#   weight = exp(−λ × días_atrás).
#   λ = 0.002 → semivida ~347 días. Partidos de 2022 pesan ~15% de los de 2026.
#   λ = 0.003 → semivida ~231 días. Más agresivo.
#   λ = 0.0   → todos los partidos pesan igual (comportamiento anterior).
#   Rango recomendado para sweep: 0.001 a 0.005 (paso 0.001).
XGB_TIME_DECAY_LAMBDA: float = float(os.getenv("XGB_TIME_DECAY_LAMBDA", "0.002"))

# Fecha de referencia para calcular días_atrás en entrenamiento.
# None = usa date.today() automáticamente (recomendado para producción).
# Se puede fijar a una fecha específica para reproducibilidad en tests/sweeps.
TIME_DECAY_REFERENCE_DATE: str | None = os.getenv("TIME_DECAY_REFERENCE_DATE", None)

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

# ─── TEMPORADAS HISTÓRICAS ESPN ───────────────────────────────────────────────
# Ampliado a 2022. Se extiende automáticamente cada año sin tocar este archivo.
ESPN_HISTORICAL_SEASONS: list[int] = list(range(2022, date.today().year + 1))
# Resultado actual (2026-03-31): [2022, 2023, 2024, 2025, 2026]