"""
espn_collector.py — v3
Fuente unificada ESPN para FOOTBOT. Sin API key. Sin límite oficial.

FIXES v3:
  A1 — _parse_score usa displayValue como fuente primaria (value tiene coma decimal)
  A2 — get_match_odds itera providers hasta encontrar datos válidos (Bet365 null)
  A3 — to_decimal normaliza coma decimal antes de float() (DraftKings col.1)
  B1/B2 — _safe_int en get_standings para valores con coma decimal
  D1/D2 — get_team_schedule acepta parámetro seasons (10x más datos históricos)

Base URLs:
  Site API v2 : https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/
  Site API v2b: https://site.api.espn.com/apis/v2/sports/soccer/{slug}/   ← standings
  Core API v2 : https://sports.core.api.espn.com/v2/sports/soccer/leagues/{slug}/

Exporta los mismos símbolos que scheduler_nacional.py importa:
    ApifootballClient (alias de ESPNClient), get_fixtures_hoy, get_standings,
    load_historical_nacional, run_live_polling, run_daily, FINISHED_STATUSES

Exporta adicionalmente:
    get_fixtures_today      → fixtures del día (clubes + selecciones)
    get_results_espn        → resultados finalizados del día
    build_historical_espn   → histórico por equipo desde /schedule
    enrich_fixtures_with_odds → cuotas en tiempo real desde Core API /odds
    get_win_probability     → probabilidades en vivo desde Core API /probabilities
    get_plays               → play-by-play para xG aproximado
    get_team_ids            → mapa displayName → team_id
"""

import os
import json
import time
import logging
import requests
import pandas as pd
from datetime import date, datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_RAW, DATA_PROCESSED, LOGS_DIR,
    ESPN_SITE_V2, ESPN_SITE_V2B, ESPN_CORE_V2,
    LIGAS_ESPN, LIGAS_ESPN_ACTIVAS,
    COMPETICIONES_NACIONALES_ESPN,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── SLUGS (referencia local para compatibilidad) ────────────────────────────

SLUGS_SELECCIONES: dict[str, tuple[int, str]] = COMPETICIONES_NACIONALES_ESPN

SLUGS_CLUBES: dict[str, tuple[int, str]] = {
    k: v for k, v in LIGAS_ESPN.items()
    if k not in COMPETICIONES_NACIONALES_ESPN
}

TODOS_LOS_SLUGS: dict[str, tuple[int, str]] = {
    **LIGAS_ESPN,
    **COMPETICIONES_NACIONALES_ESPN,
}

# Inverso: league_id → slug
_ID_A_SLUG: dict[int, str] = {v[0]: k for k, v in TODOS_LOS_SLUGS.items()}

# ─── STATUSES ────────────────────────────────────────────────────────────────

FINISHED_STATUSES  = {"FT", "AET", "PEN"}
LIVE_STATUSES      = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"}
SCHEDULED_STATUSES = {"NS", "TBD"}

_STATUS_MAP = {
    "STATUS_SCHEDULED":   "NS",
    "STATUS_IN_PROGRESS": "1H",
    "STATUS_HALFTIME":    "HT",
    "STATUS_FULL_TIME":   "FT",
    "STATUS_FINAL":       "FT",
    "STATUS_FINAL_AET":   "AET",
    "STATUS_FINAL_PEN":   "PEN",
    "STATUS_POSTPONED":   "PST",
    "STATUS_CANCELED":    "CANC",
    "STATUS_SUSPENDED":   "SUSP",
    "STATUS_ABANDONED":   "ABD",
}

POLL_INTERVAL_LIVE = 5 * 60   # 5 min cuando hay partidos en juego
POLL_INTERVAL_IDLE = 10 * 60  # 10 min esperando inicio

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; FOOTBOT/1.0)",
    "Accept":     "application/json",
})


# ─── CLIENTE ─────────────────────────────────────────────────────────────────

class ESPNClient:
    """
    Cliente unificado para todas las ESPN APIs.
    Sin API key. Sin límite oficial documentado.
    Delay configurable para no abusar.
    """

    def __init__(self, delay: float = 0.5):
        self._calls = 0
        self._delay = delay
        log.info("ESPNClient listo — sin API key")

    def get(self, url: str, params: dict = None) -> dict | None:
        try:
            resp = _SESSION.get(url, params=params, timeout=12)
            resp.raise_for_status()
            self._calls += 1
            time.sleep(self._delay)
            return resp.json()
        except requests.exceptions.HTTPError as e:
            # 400/404 son esperados para slugs fuera de temporada
            level = logging.DEBUG if e.response.status_code in (400, 404) else logging.WARNING
            log.log(level, f"HTTP {e.response.status_code} → {url}")
            return None
        except Exception as e:
            log.warning(f"Error ESPN GET {url}: {e}")
            return None

    @property
    def remaining(self):
        """Compatibilidad con código que chequea client.remaining."""
        return None

    @property
    def calls(self) -> int:
        return self._calls

    def status(self) -> str:
        return f"ESPN API: {self._calls} llamadas esta sesión"


# Alias para compatibilidad con scheduler_nacional.py existente
ApifootballClient = ESPNClient


# ─── HELPERS DE PARSEO ───────────────────────────────────────────────────────

def _norm_status(raw: str) -> str:
    return _STATUS_MAP.get(raw, raw)


def _parse_score(score) -> int | None:
    """
    FIX A1: usa displayValue como fuente primaria.

    ESPN devuelve score de dos formas según el endpoint:
      - /scoreboard   → string plano "3"
      - /schedule     → dict {"value": 3,0, "displayValue": "3", ...}

    El campo value tiene coma decimal en entornos ES/COL (3,0 en vez de 3.0),
    lo que hace que float() falle silenciosamente. displayValue siempre es
    un string limpio sin coma ("3", "0") — es la fuente correcta.
    """
    if score is None:
        return None
    # Caso dict: endpoint /schedule
    if isinstance(score, dict):
        # displayValue primero — siempre string limpio
        dv = score.get("displayValue")
        if dv is not None:
            try:
                return int(dv)
            except (ValueError, TypeError):
                pass
        # Fallback a value normalizando coma decimal
        val = score.get("value")
        if val is None:
            return None
        try:
            return int(float(str(val).replace(",", ".")))
        except (ValueError, TypeError):
            return None
    # Caso string plano: endpoint /scoreboard
    try:
        return int(float(str(score).replace(",", ".")))
    except (ValueError, TypeError):
        return None


def _safe_int(val, default: int = 0) -> int:
    """
    FIX B1: convierte valores con coma decimal a int sin explotar.
    ESPN standings devuelve stats como 27,0 (Decimal con coma en ES/COL).
    int("27,0") explota — este helper lo resuelve.
    """
    try:
        return int(float(str(val).replace(",", ".")))
    except (TypeError, ValueError):
        return default


def _parse_fixture(event: dict, league_id: int, league_name: str) -> dict | None:
    """
    Convierte un evento del scoreboard ESPN al schema interno de FOOTBOT.
    Schema idéntico al usado por _01_data_collector y scheduler.
    """
    try:
        comp        = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            return None

        raw_status = comp.get("status", {}).get("type", {}).get("name", "STATUS_SCHEDULED")
        status     = _norm_status(raw_status)
        elapsed    = comp.get("status", {}).get("displayClock", "0'").replace("'", "")
        venue_data = comp.get("venue", {})
        home_score = home.get("score")
        away_score = away.get("score")

        return {
            "fixture_id":    str(event.get("id", "")),
            "date":          event.get("date", ""),
            "timestamp":     event.get("date", ""),
            "status":        status,
            "status_long":   raw_status,
            "elapsed":       elapsed,
            "league_id":     league_id,
            "league_name":   league_name,
            "season":        event.get("season", {}).get("year"),
            "round":         (comp.get("notes") or [{}])[0].get("headline", ""),
            "home_team":     home.get("team", {}).get("displayName", ""),
            "home_team_id":  str(home.get("team", {}).get("id", "")),
            "away_team":     away.get("team", {}).get("displayName", ""),
            "away_team_id":  str(away.get("team", {}).get("id", "")),
            "home_goals":    int(home_score) if home_score is not None else None,
            "away_goals":    int(away_score) if away_score is not None else None,
            "home_ht":       None,
            "away_ht":       None,
            "home_ft":       int(home_score) if status in FINISHED_STATUSES and home_score is not None else None,
            "away_ft":       int(away_score) if status in FINISHED_STATUSES and away_score is not None else None,
            "venue":         venue_data.get("fullName", ""),
            "venue_city":    (venue_data.get("address") or {}).get("city", ""),
        }
    except (KeyError, TypeError, ValueError) as e:
        log.debug(f"Error parseando fixture ESPN: {e}")
        return None


# ─── SCOREBOARD (fixtures del día) ───────────────────────────────────────────

def get_scoreboard(client: ESPNClient, slug: str,
                   league_id: int, league_name: str,
                   target_date: str = None) -> list[dict]:
    """
    Fixtures del día para un slug.
    target_date: YYYY-MM-DD (ESPN requiere YYYYMMDD sin guiones).
    HTTP 400/404 son normales para slugs fuera de temporada — se loguean como DEBUG.
    """
    date_param = (target_date or date.today().strftime("%Y-%m-%d")).replace("-", "")
    url  = f"{ESPN_SITE_V2}/{slug}/scoreboard"
    data = client.get(url, params={"dates": date_param})

    if not data:
        log.debug(f"Slug inactivo o sin datos: {league_name} ({slug})")
        return []

    fixtures = []
    for event in data.get("events", []):
        parsed = _parse_fixture(event, league_id, league_name)
        if parsed:
            fixtures.append(parsed)

    if fixtures:
        log.info(f"{league_name}: {len(fixtures)} partidos")
    else:
        log.debug(f"{league_name}: 0 partidos hoy")

    return fixtures


def get_fixtures_today(client: ESPNClient = None,
                       slugs: dict = None,
                       target_date: str = None) -> pd.DataFrame:
    """
    Fixtures del día para todos los slugs indicados.
    Reemplaza get_fixtures_today() de _01_data_collector.py para ligas ESPN.

    slugs: por defecto TODOS_LOS_SLUGS.
           Pasar SLUGS_CLUBES o SLUGS_SELECCIONES para filtrar.
    """
    if client is None:
        client = ESPNClient()
    if slugs is None:
        slugs = TODOS_LOS_SLUGS
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    all_fixtures = []
    for slug, (league_id, league_name) in slugs.items():
        all_fixtures.extend(
            get_scoreboard(client, slug, league_id, league_name, target_date)
        )

    df = pd.DataFrame(all_fixtures)
    if not df.empty:
        path = os.path.join(DATA_RAW, f"fixtures_espn_{target_date}.csv")
        df.to_csv(path, index=False)
        log.info(f"Fixtures ESPN guardados: {len(df)} partidos → {path}")
    else:
        log.info("Sin partidos ESPN hoy.")
    return df


# Alias para compatibilidad con scheduler_nacional.py
def get_fixtures_hoy(client=None, target_date: str = None) -> list[dict]:
    """Fixtures del día para SELECCIONES. Mismo contrato que la versión anterior."""
    if client is None:
        client = ESPNClient()
    return list(
        get_fixtures_today(client, SLUGS_SELECCIONES, target_date)
        .to_dict("records")
    )


# ─── CUOTAS EN TIEMPO REAL (Core API /odds) ───────────────────────────────────

def _to_decimal(val) -> float | None:
    """
    Convierte moneyline americano a decimal.
    También acepta valores ya en decimal (>1 y <100).
    Maneja coma decimal (270,0 → 270.0).
    """
    if val is None:
        return None
    try:
        v = float(str(val).replace(",", "."))
        if v >= 100:     return round(v / 100 + 1, 3)   # +200 → 3.00
        elif v <= -100:  return round(-100 / v + 1, 3)  # -150 → 1.67
        elif v > 1:      return round(v, 3)              # ya decimal
        return None
    except (TypeError, ValueError):
        return None


def get_match_odds(client, slug: str, event_id: str) -> dict | None:
    """
    Extrae cuotas del Core API /odds.
 
    Schema confirmado (response_schemas.md — Betting Odds section):
    {
      "items": [{
        "provider": {"id": "41", "name": "DraftKings", "priority": 1},
        "overUnder": 2.5,          ← línea de goles totales
        "spread": -0.5,            ← handicap (negativo = favorito local)
        "overOdds": -110,          ← moneyline americano para Over
        "underOdds": -110,         ← moneyline americano para Under
        "homeTeamOdds": {
          "moneyLine": -165,       ← 1X2 local
          "spreadOdds": -110       ← cuota del spread para local
        },
        "awayTeamOdds": {
          "moneyLine": 140         ← 1X2 visitante
        },
        "drawOdds": {              ← SOLO en fútbol
          "moneyLine": 280
        },
        "open": {                  ← líneas de apertura
          "over":  {"value": 2.5},
          "under": {"value": 2.5},
          "spread": {"home": {"line": -0.5}}
        }
      }]
    }
 
    Itera providers por prioridad ascendente, salta los que tienen
    moneyLine null en home o away.
    """
    import logging
    log = logging.getLogger(__name__)
 
    url  = f"https://sports.core.api.espn.com/v2/sports/soccer/leagues/{slug}/events/{event_id}/competitions/{event_id}/odds"
    data = client.get(url, params={"limit": 5})
    if not data:
        return None
 
    items = data.get("items", [])
    if not items:
        return None
 
    items_sorted = sorted(
        items,
        key=lambda x: x.get("provider", {}).get("priority", 999)
    )
 
    for item in items_sorted:
        try:
            home_odds_raw = item.get("homeTeamOdds") or {}
            away_odds_raw = item.get("awayTeamOdds") or {}
            draw_odds_raw = item.get("drawOdds")     or {}
 
            h_ml = home_odds_raw.get("moneyLine")
            a_ml = away_odds_raw.get("moneyLine")
            # Para fútbol, drawOdds es un objeto separado con su propio moneyLine
            d_ml = draw_odds_raw.get("moneyLine")
 
            if h_ml is None or a_ml is None:
                provider_name = item.get("provider", {}).get("name", "?")
                log.debug(f"Provider {provider_name} sin moneyLine — siguiente")
                continue
 
            h = _to_decimal(h_ml)
            a = _to_decimal(a_ml)
            d = _to_decimal(d_ml) if d_ml is not None else None
 
            if not h or not a:
                continue
 
            # ── Over/Under ────────────────────────────────────────────────
            # overUnder es la LÍNEA (ej: 2.5), no las cuotas
            total_line  = item.get("overUnder")    # float: 2.5
            over_odds   = item.get("overOdds")     # int americano: -110
            under_odds  = item.get("underOdds")    # int americano: -110
 
            over_dec  = _to_decimal(over_odds)  if over_odds  is not None else None
            under_dec = _to_decimal(under_odds) if under_odds is not None else None
 
            # ── Spread / Asian Handicap ───────────────────────────────────
            spread_line      = item.get("spread")                            # float: -0.5
            spread_home_odds = home_odds_raw.get("spreadOdds")               # int: -110
            spread_away_odds = away_odds_raw.get("spreadOdds")               # int: -110
            spread_home_dec  = _to_decimal(spread_home_odds) if spread_home_odds is not None else None
            spread_away_dec  = _to_decimal(spread_away_odds) if spread_away_odds is not None else None
 
            # ── Apertura (movimiento de línea) ────────────────────────────
            open_data        = item.get("open") or {}
            open_total_line  = (open_data.get("over")   or {}).get("value")
            open_spread_home = ((open_data.get("spread") or {}).get("home") or {}).get("line")
 
            provider_name = item.get("provider", {}).get("name", "ESPN")
 
            return {
                # 1X2
                "home":               h,
                "draw":               d or 3.40,   # fallback si no hay empate
                "away":               a,
                # Over/Under
                "total_line":         float(total_line) if total_line is not None else 2.5,
                "over_odds":          over_dec,
                "under_odds":         under_dec,
                # Spread / AH
                "spread_line":        float(spread_line) if spread_line is not None else None,
                "spread_home_odds":   spread_home_dec,
                "spread_away_odds":   spread_away_dec,
                # Apertura
                "open_total_line":    float(open_total_line) if open_total_line is not None else None,
                "open_spread_home":   float(open_spread_home) if open_spread_home is not None else None,
                # Meta
                "provider":           provider_name,
                "source":             "espn_odds",
            }
 
        except (TypeError, ValueError, KeyError) as e:
            log.debug(f"Error parseando odds item {event_id}: {e}")
            continue
 
    log.debug(f"Sin odds válidas para event {event_id}")
    return None


def enrich_fixtures_with_odds(client, fixtures_df) -> "pd.DataFrame":
    """
    Enriquece el DataFrame de fixtures con todas las cuotas ESPN disponibles.
 
    Columnas añadidas (nuevas en v5):
        espn_odds_home          float   cuota decimal local (1X2)
        espn_odds_draw          float   cuota decimal empate (1X2)
        espn_odds_away          float   cuota decimal visitante (1X2)
        espn_odds_provider      str     nombre del provider
        espn_odds_available     bool    True si hay cuotas válidas
        espn_total_line         float   línea over/under (ej: 2.5)
        espn_over_odds          float   cuota decimal Over
        espn_under_odds         float   cuota decimal Under
        espn_spread_line        float   handicap de línea (ej: -0.5 para local)
        espn_spread_home_odds   float   cuota decimal spread local
        espn_spread_away_odds   float   cuota decimal spread visitante
        espn_open_total_line    float   línea de apertura (movimiento)
        espn_open_spread_home   float   spread de apertura
    """
    import pandas as pd
    import logging
    log = logging.getLogger(__name__)
 
    if fixtures_df.empty:
        return fixtures_df
 
    # Importar mapping id→slug del módulo principal
    try:
        from src.espn_collector import _ID_A_SLUG
    except ImportError:
        # Si se usa standalone, construir desde settings
        from config.settings import LIGAS_ESPN, COMPETICIONES_NACIONALES_ESPN
        _todos = {**LIGAS_ESPN, **COMPETICIONES_NACIONALES_ESPN}
        _ID_A_SLUG = {v[0]: k for k, v in _todos.items()}
 
    df = fixtures_df.copy()
 
    # Inicializar columnas nuevas
    new_cols = {
        "espn_odds_home":        None,
        "espn_odds_draw":        None,
        "espn_odds_away":        None,
        "espn_odds_provider":    None,
        "espn_odds_available":   False,
        "espn_total_line":       None,
        "espn_over_odds":        None,
        "espn_under_odds":       None,
        "espn_spread_line":      None,
        "espn_spread_home_odds": None,
        "espn_spread_away_odds": None,
        "espn_open_total_line":  None,
        "espn_open_spread_home": None,
    }
    for col, default in new_cols.items():
        df[col] = default
 
    n_ok = 0
    for idx, row in df.iterrows():
        slug     = _ID_A_SLUG.get(int(row.get("league_id", 0)))
        event_id = str(row.get("fixture_id", ""))
        if not slug or not event_id:
            continue
 
        odds = get_match_odds(client, slug, event_id)
        if not odds:
            log.debug(f"Sin odds: {row.get('home_team')} vs {row.get('away_team')}")
            continue
 
        df.at[idx, "espn_odds_home"]        = odds["home"]
        df.at[idx, "espn_odds_draw"]        = odds["draw"]
        df.at[idx, "espn_odds_away"]        = odds["away"]
        df.at[idx, "espn_odds_provider"]    = odds["provider"]
        df.at[idx, "espn_odds_available"]   = True
        df.at[idx, "espn_total_line"]       = odds.get("total_line")
        df.at[idx, "espn_over_odds"]        = odds.get("over_odds")
        df.at[idx, "espn_under_odds"]       = odds.get("under_odds")
        df.at[idx, "espn_spread_line"]      = odds.get("spread_line")
        df.at[idx, "espn_spread_home_odds"] = odds.get("spread_home_odds")
        df.at[idx, "espn_spread_away_odds"] = odds.get("spread_away_odds")
        df.at[idx, "espn_open_total_line"]  = odds.get("open_total_line")
        df.at[idx, "espn_open_spread_home"] = odds.get("open_spread_home")
        n_ok += 1
 
        log.info(
            f"Odds {row['home_team']} vs {row['away_team']}: "
            f"1X2={odds['home']}/{odds['draw']}/{odds['away']} | "
            f"O/U {odds.get('total_line', '?')} "
            f"({odds.get('over_odds','?')}/{odds.get('under_odds','?')}) | "
            f"AH {odds.get('spread_line','?')} "
            f"[{odds['provider']}]"
        )
 
    log.info(f"Odds ESPN disponibles: {n_ok}/{len(df)} partidos")
    return df


# ─── WIN PROBABILITY EN VIVO (Core API /probabilities) ───────────────────────

def get_win_probability(client: ESPNClient, slug: str,
                        event_id: str) -> dict | None:
    """
    Probabilidades de victoria en tiempo real.
    Disponible solo durante el partido.

    Retorna {home_prob, draw_prob, away_prob, source: "espn_live"}
    """
    url  = f"{ESPN_CORE_V2}/{slug}/events/{event_id}/competitions/{event_id}/probabilities"
    data = client.get(url)
    if not data:
        return None

    items = data.get("items", [])
    if not items:
        return None

    latest = items[-1]
    try:
        h = float(str(latest.get("homeWinPercentage", 0)).replace(",", "."))
        a = float(str(latest.get("awayWinPercentage", 0)).replace(",", "."))
        d = round(max(0.0, 1.0 - h - a), 4)
        return {
            "home_prob": round(h, 4),
            "draw_prob": d,
            "away_prob": round(a, 4),
            "source":    "espn_live",
        }
    except (TypeError, ValueError):
        return None


# ─── PLAY-BY-PLAY para xG aproximado (Core API /plays) ───────────────────────

def get_plays(client: ESPNClient, slug: str,
              event_id: str, limit: int = 300) -> list[dict]:
    """
    Play-by-play: goles, remates, tarjetas, sustituciones.
    Útil para construir xG aproximado donde StatsBomb no tiene datos.
    """
    url  = f"{ESPN_CORE_V2}/{slug}/events/{event_id}/competitions/{event_id}/plays"
    data = client.get(url, params={"limit": limit})
    if not data:
        return []

    plays = []
    for item in data.get("items", []):
        play_type = item.get("type", {})
        plays.append({
            "event_id":   event_id,
            "play_id":    str(item.get("id", "")),
            "clock":      item.get("clock", {}).get("displayValue", ""),
            "period":     item.get("period", {}).get("number", 0),
            "type_id":    str(play_type.get("id", "")),
            "type_text":  play_type.get("text", ""),
            "team_id":    str(item.get("team", {}).get("id", "")),
            "text":       item.get("text", ""),
            "score_home": item.get("homeScore"),
            "score_away": item.get("awayScore"),
        })
    return plays


def estimate_xg_from_plays(plays: list[dict],
                            home_team_id: str,
                            away_team_id: str) -> dict:
    """
    Estima xG aproximado desde play-by-play.

    Ponderación (sin datos de StatsBomb):
        Gol              → 1.00 xG (ocurrió)
        Remate al arco   → 0.15 xG (promedio histórico)
        Remate fuera     → 0.05 xG
    """
    home_xg = 0.0
    away_xg = 0.0
    home_shots_on = 0
    away_shots_on = 0
    home_shots    = 0
    away_shots    = 0

    GOAL_KW     = {"goal", "score"}
    SHOT_ON_KW  = {"shot on goal", "on goal", "save"}
    SHOT_OFF_KW = {"shot", "miss", "blocked", "wide", "over"}

    for play in plays:
        t    = play.get("type_text", "").lower()
        tid  = str(play.get("team_id", ""))
        is_h = (tid == str(home_team_id))

        if any(k in t for k in GOAL_KW):
            if is_h: home_xg += 1.0
            else:    away_xg += 1.0
        elif any(k in t for k in SHOT_ON_KW):
            if is_h: home_shots_on += 1; home_xg += 0.15
            else:    away_shots_on += 1; away_xg += 0.15
        elif any(k in t for k in SHOT_OFF_KW):
            if is_h: home_shots += 1; home_xg += 0.05
            else:    away_shots += 1; away_xg += 0.05

    return {
        "home_xg_approx":   round(home_xg, 3),
        "away_xg_approx":   round(away_xg, 3),
        "home_shots_on":    home_shots_on,
        "away_shots_on":    away_shots_on,
        "home_shots_total": home_shots + home_shots_on,
        "away_shots_total": away_shots + away_shots_on,
    }


# ─── EQUIPOS (team IDs) ───────────────────────────────────────────────────────

def get_team_ids(client: ESPNClient, slug: str) -> dict[str, str]:
    """
    {displayName: team_id} para todos los equipos de una liga.
    Endpoint: /teams (Site API v2)
    Ejemplo: {"Atlético Nacional": "5264", "Millonarios": "5484"}
    """
    url  = f"{ESPN_SITE_V2}/{slug}/teams"
    data = client.get(url)
    if not data:
        return {}

    teams = {}
    try:
        for team in data["sports"][0]["leagues"][0]["teams"]:
            t = team["team"]
            teams[t["displayName"]] = str(t["id"])
    except (KeyError, IndexError) as e:
        log.warning(f"Error parseando equipos {slug}: {e}")

    log.info(f"{slug}: {len(teams)} equipos mapeados")
    return teams


def get_team_schedule(client: ESPNClient, slug: str,
                      team_id: str, team_name: str,
                      seasons: list[int] = None) -> list[dict]:
    """
    FIX D1/D2: acepta parámetro seasons para obtener histórico multi-temporada.

    Sin seasons: solo obtiene la temporada actual (~13 partidos).
    Con seasons=[2023, 2024, 2025]: obtiene ~137 partidos — 10x más datos.

    Por defecto usa las últimas 3 temporadas para balance calidad/velocidad.

    NOTA: este endpoint devuelve score como dict
    {"$ref": ..., "value": 3,0, "displayValue": "3"}.
    _parse_score() con FIX A1 maneja esto correctamente.
    """
    if seasons is None:
        yr = date.today().year
        # Últimas 3 temporadas por defecto
        seasons = [yr - 2, yr - 1, yr]

    all_events: list[dict] = []
    seen_ids:   set[str]   = set()

    for season in seasons:
        url  = f"{ESPN_SITE_V2}/{slug}/teams/{team_id}/schedule"
        data = client.get(url, params={"season": season})
        if not data:
            log.debug(f"{team_name} season {season}: sin datos")
            continue

        season_events = 0
        for event in data.get("events", []):
            try:
                eid = str(event.get("id", ""))
                if eid in seen_ids:
                    continue
                seen_ids.add(eid)

                comp        = event.get("competitions", [{}])[0]
                competitors = comp.get("competitors", [])
                home = next((c for c in competitors if c["homeAway"] == "home"), None)
                away = next((c for c in competitors if c["homeAway"] == "away"), None)
                if not home or not away:
                    continue

                raw_status = comp.get("status", {}).get("type", {}).get("name", "")
                status     = _norm_status(raw_status)

                all_events.append({
                    "match_id":     eid,
                    "date":         event.get("date", ""),
                    "slug":         slug,
                    "season":       season,
                    "home_team":    home["team"]["displayName"],
                    "home_team_id": str(home["team"]["id"]),
                    "away_team":    away["team"]["displayName"],
                    "away_team_id": str(away["team"]["id"]),
                    # FIX A1: _parse_score usa displayValue como primario
                    "home_score":   _parse_score(home.get("score")),
                    "away_score":   _parse_score(away.get("score")),
                    "status":       status,
                })
                season_events += 1
            except (KeyError, TypeError):
                continue

        log.debug(f"{team_name} season {season}: {season_events} partidos")
        time.sleep(0.3)

    log.debug(f"{team_name}: {len(all_events)} partidos totales ({len(seasons)} temporadas)")
    return all_events


# ─── HISTÓRICO POR LIGA ───────────────────────────────────────────────────────

def build_historical_espn(client: ESPNClient,
                           slug: str,
                           league_id: int,
                           league_name: str,
                           fetch_plays: bool = False,
                           max_per_team: int = None,
                           seasons: list[int] = None) -> pd.DataFrame:
    """
    FIX D1/D2: usa get_team_schedule con seasons para 10x más datos históricos.

    max_per_team: ya no es necesario con seasons — se obtiene la temporada
    completa por definición. Se mantiene para compatibilidad pero se ignora
    si se pasan seasons.

    seasons: por defecto las últimas 3 temporadas (configurado en get_team_schedule).
    """
    log.info(f"Construyendo histórico ESPN: {league_name} ({slug})")

    team_ids = get_team_ids(client, slug)
    if not team_ids:
        log.warning(f"Sin equipos para {league_name}")
        return pd.DataFrame()

    # Construir seasons si no se pasan
    if seasons is None:
        yr = date.today().year
        seasons = [yr - 2, yr - 1, yr]

    log.info(f"{league_name}: descargando temporadas {seasons}...")

    # Recopilar eventos únicos via schedules de cada equipo
    seen:   set[str]   = set()
    events: list[dict] = []

    for team_name, team_id in team_ids.items():
        team_events = get_team_schedule(
            client, slug, team_id, team_name, seasons=seasons
        )
        for event in team_events:
            if event["match_id"] not in seen:
                seen.add(event["match_id"])
                events.append(event)
        time.sleep(0.3)

    log.info(f"{league_name}: {len(events)} partidos únicos encontrados")

    # Solo finalizados — _parse_score con FIX A1 garantiza scores correctos
    finished = [
        e for e in events
        if e["status"] in FINISHED_STATUSES
        and e["home_score"] is not None
        and e["away_score"] is not None
    ]

    log.info(f"{league_name}: {len(finished)} partidos finalizados de {len(events)} totales")

    rows = []
    for i, event in enumerate(finished):
        row = {
            "match_id":     event["match_id"],
            "match_date":   pd.to_datetime(event["date"], utc=True, errors="coerce"),
            "league_id":    league_id,
            "league_name":  league_name,
            "season":       event.get("season"),
            "home_team":    event["home_team"],
            "home_team_id": event["home_team_id"],
            "away_team":    event["away_team"],
            "away_team_id": event["away_team_id"],
            "home_goals":   int(event["home_score"]),
            "away_goals":   int(event["away_score"]),
            "status":       "FT",
            "source":       "espn",
            # Cuotas históricas no disponibles en ESPN
            "B365H": None, "B365D": None, "B365A": None,
            "PSH":   None, "PSD":   None, "PSA":   None,
        }

        # xG aproximado desde play-by-play (opcional)
        if fetch_plays:
            plays = get_plays(client, slug, event["match_id"])
            if plays:
                xg = estimate_xg_from_plays(
                    plays, event["home_team_id"], event["away_team_id"]
                )
                row.update(xg)
            time.sleep(0.5)

        rows.append(row)

        if (i + 1) % 20 == 0:
            log.info(f"  {league_name}: {i+1}/{len(finished)} procesados")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["match_date"] = pd.to_datetime(df["match_date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("match_date").reset_index(drop=True)

    # Guardar en data/raw con prefijo espn_
    path = os.path.join(DATA_RAW, f"espn_{slug.replace('.', '_')}.csv")
    df.to_csv(path, index=False)
    log.info(f"{league_name}: {len(df)} partidos históricos ({seasons}) → {path}")
    return df


# ─── STANDINGS (fix: usa /apis/v2/ no /site/v2/) ─────────────────────────────

def get_standings(client: ESPNClient = None,
                  slugs: dict = None) -> dict[int, list]:
    """
    Posiciones por competición.
    CORRECCIÓN: usa ESPN_SITE_V2B (/apis/v2/) en lugar de /site/v2/
    que devuelve {} vacío para soccer standings.

    FIX B1/B2: _safe_int en todos los campos numéricos para manejar
    valores con coma decimal (27,0 → 27) sin explotar.
    """
    if client is None:
        client = ESPNClient()
    if slugs is None:
        slugs = SLUGS_SELECCIONES

    standings = {}

    for slug, (league_id, league_name) in slugs.items():
        url  = f"{ESPN_SITE_V2B}/{slug}/standings"
        data = client.get(url)
        if not data:
            log.debug(f"Sin standings: {league_name}")
            continue

        rows        = []
        entry_lists = []

        # La estructura varía: standings directos o con children (grupos Copa América, etc.)
        if data.get("standings", {}).get("entries"):
            entry_lists.append(data["standings"]["entries"])
        for child in data.get("children", []):
            entries = child.get("standings", {}).get("entries", [])
            if entries:
                entry_lists.append(entries)

        for entries in entry_lists:
            for entry in entries:
                team  = entry.get("team", {})
                # FIX B1: usar dict para acceso robusto con _safe_int
                stats = {s["name"]: s.get("value", 0) for s in entry.get("stats", [])}

                # Bonus B2: parsear "overall" W-D-L si disponible (más robusto)
                overall_str = next(
                    (s.get("displayValue", "") for s in entry.get("stats", [])
                     if s.get("name") == "overall"),
                    ""
                )
                wins_parsed = draws_parsed = losses_parsed = None
                if "-" in overall_str:
                    parts = overall_str.split("-")
                    if len(parts) == 3:
                        wins_parsed   = _safe_int(parts[0])
                        draws_parsed  = _safe_int(parts[1])
                        losses_parsed = _safe_int(parts[2])

                rows.append({
                    "league_id":     league_id,
                    "league_name":   league_name,
                    # FIX B1: _safe_int en todos los campos numéricos
                    "rank":          _safe_int(stats.get("rank",            0)),
                    "team":          team.get("displayName", ""),
                    "team_id":       str(team.get("id", "")),
                    "points":        _safe_int(stats.get("points",          0)),
                    "played":        _safe_int(stats.get("gamesPlayed",     0)),
                    "won":           wins_parsed   if wins_parsed   is not None else _safe_int(stats.get("wins",   0)),
                    "drawn":         draws_parsed  if draws_parsed  is not None else _safe_int(stats.get("ties",   0)),
                    "lost":          losses_parsed if losses_parsed is not None else _safe_int(stats.get("losses", 0)),
                    "goals_for":     _safe_int(stats.get("pointsFor",       0)),
                    "goals_against": _safe_int(stats.get("pointsAgainst",   0)),
                    "goal_diff":     _safe_int(stats.get("pointDifferential",0)),
                    "form":          str(stats.get("streak", "")),
                    "group":         data.get("name", ""),
                    "updated_at":    str(date.today()),
                })

        if rows:
            standings[league_id] = rows
            log.info(f"Standings {league_name}: {len(rows)} equipos")

    all_rows = [r for rows in standings.values() for r in rows]
    if all_rows:
        path = os.path.join(DATA_RAW, "standings.csv")
        pd.DataFrame(all_rows).to_csv(path, index=False)
        log.info(f"Standings guardados → {path}")

    return standings


# ─── RESULTADOS DEL DÍA ───────────────────────────────────────────────────────

def get_results_espn(target_date: str = None,
                     slugs: dict = None) -> dict:
    """
    Resultados finalizados del día.
    Reemplaza get_results_fdorg() de _01_data_collector para ligas ESPN.
    Retorna {(home_team, away_team): {home_goals, away_goals, fixture_id, status}}
    """
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")
    if slugs is None:
        slugs = TODOS_LOS_SLUGS

    client  = ESPNClient()
    results = {}

    for slug, (league_id, league_name) in slugs.items():
        for f in get_scoreboard(client, slug, league_id, league_name, target_date):
            if f["status"] in FINISHED_STATUSES:
                results[(f["home_team"], f["away_team"])] = {
                    "home_goals": f["home_goals"] or 0,
                    "away_goals": f["away_goals"] or 0,
                    "fixture_id": f["fixture_id"],
                    "status":     f["status"],
                }

    log.info(f"Resultados ESPN ({target_date}): {len(results)} finalizados")
    return results


# ─── LIVE POLLING ────────────────────────────────────────────────────────────

def _refresh_fixture(client: ESPNClient, fixture: dict) -> dict | None:
    """
    Refresca un partido consultando el scoreboard del día.
    Más eficiente que una llamada por partido: descarga el scoreboard
    de la liga completo y filtra por fixture_id.
    """
    slug = _ID_A_SLUG.get(int(fixture.get("league_id", 0)))
    if not slug:
        return None

    today_str = date.today().strftime("%Y-%m-%d")
    fresh     = get_scoreboard(
        client, slug,
        fixture["league_id"], fixture["league_name"],
        today_str,
    )
    return next((f for f in fresh if f["fixture_id"] == fixture["fixture_id"]), None)


def run_live_polling(client: ESPNClient = None,
                     fixtures: list[dict] = None,
                     on_update=None) -> list[dict]:
    """
    Polling en vivo enriquecido con win probability del Core API.
    Mismo contrato que la versión anterior.
    """
    if client is None:
        client = ESPNClient()
    if not fixtures:
        return []

    pending  = {f["fixture_id"]: f for f in fixtures
                if f["status"] in LIVE_STATUSES | SCHEDULED_STATUSES}
    finished = {f["fixture_id"]: f for f in fixtures
                if f["status"] in FINISHED_STATUSES}

    if not pending:
        log.info("Sin partidos activos para polling.")
        return list(finished.values())

    log.info(f"Live polling ESPN: {len(pending)} partidos pendientes")
    last_scores = {fid: (f.get("home_goals"), f.get("away_goals"))
                   for fid, f in pending.items()}

    while pending:
        n_live = sum(1 for f in pending.values() if f["status"] in LIVE_STATUSES)
        time.sleep(POLL_INTERVAL_LIVE if n_live > 0 else POLL_INTERVAL_IDLE)

        to_remove = []
        for fid, fixture in list(pending.items()):
            updated = _refresh_fixture(client, fixture)
            if not updated:
                continue

            # Enriquecer con win probability si está en juego
            if updated["status"] in LIVE_STATUSES:
                slug = _ID_A_SLUG.get(int(fixture.get("league_id", 0)))
                if slug:
                    wp = get_win_probability(client, slug, fid)
                    if wp:
                        updated.update({
                            "live_home_prob": wp["home_prob"],
                            "live_draw_prob": wp["draw_prob"],
                            "live_away_prob": wp["away_prob"],
                        })

            # Detectar gol
            new_score = (updated.get("home_goals"), updated.get("away_goals"))
            if new_score != last_scores.get(fid) and None not in new_score:
                last_scores[fid] = new_score
                log.info(
                    f"GOL: {updated['home_team']} {new_score[0]}-"
                    f"{new_score[1]} {updated['away_team']} "
                    f"(min {updated.get('elapsed', '?')})"
                )
                if on_update:
                    try:
                        on_update(updated)
                    except Exception as e:
                        log.warning(f"Error en on_update: {e}")

            pending[fid] = updated

            if updated["status"] in FINISHED_STATUSES:
                log.info(
                    f"Finalizado: {updated['home_team']} "
                    f"{updated['home_goals']}-{updated['away_goals']} "
                    f"{updated['away_team']}"
                )
                finished[fid] = updated
                to_remove.append(fid)

        for fid in to_remove:
            del pending[fid]

        log.info(f"Polling: {len(pending)} activos | {len(finished)} finalizados")

    all_results = list(finished.values()) + list(pending.values())
    _save_live_results(all_results)
    return all_results


def _save_live_results(fixtures: list[dict]):
    if not fixtures:
        return
    today = date.today().strftime("%Y-%m-%d")
    path  = os.path.join(DATA_RAW, f"live_results_{today}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"Resultados live guardados → {path}")


# ─── HISTÓRICO SELECCIONES ────────────────────────────────────────────────────

def load_historical_nacional() -> pd.DataFrame:
    """
    Carga histórico de selecciones desde disco.
    Lee JSONs de live_results + CSVs ESPN de competiciones nacionales.
    """
    frames  = []
    raw_dir = Path(DATA_RAW)

    # JSONs del polling en vivo (formato antiguo + nuevo)
    for pattern in ["nacional_results_*.json", "live_results_*.json"]:
        for json_file in sorted(raw_dir.glob(pattern)):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                frames.append(pd.DataFrame(data))
            except Exception as e:
                log.warning(f"Error cargando {json_file}: {e}")

    # CSVs ESPN de competiciones nacionales
    for slug in SLUGS_SELECCIONES:
        csv_path = raw_dir / f"espn_{slug.replace('.', '_')}.csv"
        if csv_path.exists():
            try:
                frames.append(pd.read_csv(csv_path))
            except Exception as e:
                log.warning(f"Error cargando {csv_path}: {e}")

    # CSV histórico manual si existe
    manual_path = os.path.join(DATA_RAW, "nacional_historical.csv")
    if os.path.exists(manual_path):
        frames.append(pd.read_csv(manual_path))

    if not frames:
        log.warning("Sin datos históricos de selecciones nacionales.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Normalizar fecha
    date_col = "date" if "date" in df.columns else "match_date"
    df["match_date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Solo finalizados
    if "status" in df.columns:
        df = df[df["status"].isin(FINISHED_STATUSES)].copy()

    df = df.dropna(subset=["match_date", "home_goals", "away_goals"])
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])

    # Deduplicar
    dedup_col = "fixture_id" if "fixture_id" in df.columns else "match_id"
    if dedup_col in df.columns:
        df = df.drop_duplicates(subset=[dedup_col])

    log.info(f"Histórico selecciones: {len(df)} partidos")
    return df.sort_values("match_date").reset_index(drop=True)


# ─── run_daily ────────────────────────────────────────────────────────────────

def run_daily(target_date: str = None) -> dict:
    """Punto de entrada para scheduler_nacional.py --mode calendar."""
    client = ESPNClient()
    log.info(client.status())
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    fixtures  = get_fixtures_hoy(client, target_date)
    standings = get_standings(client, SLUGS_SELECCIONES)
    log.info(f"run_daily: {len(fixtures)} fixtures · {client.status()}")
    return {"fixtures": fixtures, "standings": standings, "date": target_date}


# ─── PRUEBA STANDALONE ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    client = ESPNClient(delay=0.5)
    today  = date.today().strftime("%Y-%m-%d")

    print(f"\n=== FIXTURES HOY ({today}) ===")
    df_today = get_fixtures_today(client, target_date=today)
    if df_today.empty:
        print("Sin partidos hoy.")
    else:
        for _, r in df_today.iterrows():
            score = (f"{r['home_goals']}-{r['away_goals']}"
                     if r.get("home_goals") is not None else "vs")
            print(f"  [{r['status']:3s}] {r['league_name']:35s} "
                  f"{r['home_team']} {score} {r['away_team']}")

    print("\n=== STANDINGS Col.1 ===")
    st = get_standings(client, {"col.1": (501, "Liga BetPlay")})
    for row in (st.get(501) or [])[:5]:
        print(f"  {row['rank']:2d}. {row['team']:25s} {row['points']} pts "
              f"({row['played']} PJ)")

    print("\n=== HISTÓRICO Col.1 (últimas 3 temporadas) ===")
    hist = build_historical_espn(
        client, "col.1", 501, "Liga BetPlay",
        fetch_plays=False,
        seasons=[date.today().year - 2, date.today().year - 1, date.today().year],
    )
    print(f"  {len(hist)} partidos descargados")
    if not hist.empty:
        print(hist[["match_date", "season", "home_team", "home_goals",
                     "away_goals", "away_team"]].tail(5).to_string(index=False))