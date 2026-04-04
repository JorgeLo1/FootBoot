"""
espn_collector.py — v6
Fuente unificada ESPN para FOOTBOT. Sin API key. Sin límite oficial.

FIXES v3:
  A1 — _parse_score usa displayValue como fuente primaria (value tiene coma decimal)
  A2 — get_match_odds itera providers hasta encontrar datos válidos (Bet365 null)
  A3 — to_decimal normaliza coma decimal antes de float() (DraftKings col.1)
  B1/B2 — _safe_int en get_standings para valores con coma decimal
  D1/D2 — get_team_schedule acepta parámetro seasons (10x más datos históricos)

FIXES v4:
  Punto 3 — _parse_fixture usa _parse_score() en lugar de int() directo

NUEVO v5:
  E1 — get_team_injuries: descarga lesiones por equipo (teams/{id}/injuries)
       Añade features home_injured_count / away_injured_count al pipeline.
  E2 — get_match_summary_bpi: extrae ESPN BPI (win probability pre-partido)
       desde /summary?event={id} → predictor.gameProjection
  E3 — enrich_fixtures_with_injuries: enriquece fixtures_df con lesiones
  E4 — get_league_top_scorers: descarga goleadores de una liga (/leaders)
  E5 — Soporte para nuevas ligas en LIGAS_ESPN (chi.1, per.1, ecu.1, uru.1,
       par.1, uefa.europa, uefa.europa.conf, eng.1, esp.1, ger.1, ita.1, etc.)

NUEVO v6:
  F1 — get_standings_context: convierte standings en features de motivación
       por equipo. Calcula relegation_threat, title_race, european_race,
       points_to_safety, points_to_cut, season_progress y forma_reciente
       desde el record W-D-L del endpoint /standings.
       Diseñado para ser llamado una vez por día y cacheado en memoria.
  F2 — enrich_fixtures_with_standings: enriquece fixtures_df con las features
       de motivación para local y visitante. Añade 8 columnas por equipo
       (16 total + 4 diferenciales).

Exporta adicionalmente:
    get_standings_context          → dict {team_name → StandingsFeatures} (NUEVO v6)
    enrich_fixtures_with_standings → enriquece fixtures con motivación (NUEVO v6)
  Site API v2 : https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/
  Site API v2b: https://site.api.espn.com/apis/v2/sports/soccer/{slug}/   ← standings
  Core API v2 : https://sports.core.api.espn.com/v2/sports/soccer/leagues/{slug}/

Exporta los mismos símbolos que scheduler_nacional.py importa:
    ApifootballClient (alias de ESPNClient), get_fixtures_hoy, get_standings,
    load_historical_nacional, run_live_polling, run_daily, FINISHED_STATUSES

Exporta adicionalmente:
    get_fixtures_today         → fixtures del día (clubes + selecciones)
    get_results_espn           → resultados finalizados del día
    build_historical_espn      → histórico por equipo desde /schedule
    enrich_fixtures_with_odds  → cuotas en tiempo real desde Core API /odds
    enrich_fixtures_with_injuries → lesiones de titulares por equipo (NUEVO v5)
    get_win_probability        → probabilidades en vivo desde Core API /probabilities
    get_match_summary_bpi      → ESPN BPI pre-partido desde /summary (NUEVO v5)
    get_plays                  → play-by-play para xG aproximado
    get_team_ids               → mapa displayName → team_id
    get_league_top_scorers     → goleadores de una liga (NUEVO v5)
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
    ESPN_INJURIES_ENABLED, ESPN_BPI_ENABLED,
    ESPN_STANDINGS_FEATURES,
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

# Gravedad de lesiones para ponderar impacto en el modelo
_INJURY_SEVERITY = {
    "out":        1.0,   # baja segura
    "doubtful":   0.6,   # probable baja
    "questionable": 0.3, # posible baja
    "day-to-day": 0.2,   # baja menor
    "probable":   0.1,   # muy probable que juegue
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
            level = logging.DEBUG if e.response.status_code in (400, 404) else logging.WARNING
            log.log(level, f"HTTP {e.response.status_code} → {url}")
            return None
        except Exception as e:
            log.warning(f"Error ESPN GET {url}: {e}")
            return None

    @property
    def remaining(self):
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
    """
    if score is None:
        return None
    if isinstance(score, dict):
        dv = score.get("displayValue")
        if dv is not None:
            try:
                return int(dv)
            except (ValueError, TypeError):
                pass
        val = score.get("value")
        if val is None:
            return None
        try:
            return int(float(str(val).replace(",", ".")))
        except (ValueError, TypeError):
            return None
    try:
        return int(float(str(score).replace(",", ".")))
    except (ValueError, TypeError):
        return None


def _safe_int(val, default: int = 0) -> int:
    """FIX B1: convierte valores con coma decimal a int sin explotar."""
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

        # FIX punto 3: usar _parse_score() en lugar de int() directo.
        parsed_home = _parse_score(home_score)
        parsed_away = _parse_score(away_score)

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
            "home_goals":    parsed_home,
            "away_goals":    parsed_away,
            "home_ht":       None,
            "away_ht":       None,
            "home_ft":       parsed_home if status in FINISHED_STATUSES and parsed_home is not None else None,
            "away_ft":       parsed_away if status in FINISHED_STATUSES and parsed_away is not None else None,
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


def get_fixtures_hoy(client=None, target_date: str = None) -> list[dict]:
    """Fixtures del día para SELECCIONES. Mismo contrato que la versión anterior."""
    if client is None:
        client = ESPNClient()
    return list(
        get_fixtures_today(client, SLUGS_SELECCIONES, target_date)
        .to_dict("records")
    )


# ─── LESIONES (NUEVO v5 — E1) ─────────────────────────────────────────────────

def get_team_injuries(client: ESPNClient, slug: str,
                      team_id: str) -> list[dict]:
    """
    E1: Descarga el parte de lesiones de un equipo.
    Endpoint: /apis/site/v2/sports/soccer/{slug}/teams/{id}/injuries

    Retorna lista de dicts:
        {player_name, status, injury_type, severity_weight}

    severity_weight: float 0-1 para ponderar impacto en el modelo.
        1.0 = baja segura (out)
        0.6 = dudosa (doubtful)
        0.3 = questionable
        0.2 = day-to-day
        0.1 = probable
    """
    url  = f"{ESPN_SITE_V2}/{slug}/teams/{team_id}/injuries"
    data = client.get(url)
    if not data:
        return []

    injuries = []
    for item in data.get("injuries", []):
        athlete = item.get("athlete", {})
        status_raw = (item.get("status") or "").lower().strip()
        severity   = _INJURY_SEVERITY.get(status_raw, 0.5)

        injuries.append({
            "player_id":       str(athlete.get("id", "")),
            "player_name":     athlete.get("displayName", ""),
            "position":        (athlete.get("position") or {}).get("abbreviation", ""),
            "injury_type":     (item.get("type") or {}).get("name", ""),
            "injury_detail":   item.get("detail", ""),
            "status":          status_raw,
            "severity_weight": severity,
            "date":            item.get("date", ""),
        })

    log.debug(f"Lesiones equipo {team_id} ({slug}): {len(injuries)} jugadores")
    return injuries


def _aggregate_injury_score(injuries: list[dict]) -> dict:
    """
    Agrega las lesiones de un equipo en features numéricas.

    Retorna:
        injured_count      → número total de lesionados
        injury_score       → suma ponderada por severidad
        out_count          → bajas seguras (out)
        doubtful_count     → dudosas
    """
    if not injuries:
        return {
            "injured_count":  0,
            "injury_score":   0.0,
            "out_count":      0,
            "doubtful_count": 0,
        }
    return {
        "injured_count":  len(injuries),
        "injury_score":   round(sum(i["severity_weight"] for i in injuries), 2),
        "out_count":      sum(1 for i in injuries if i["status"] == "out"),
        "doubtful_count": sum(1 for i in injuries if i["status"] == "doubtful"),
    }


def enrich_fixtures_with_injuries(client: ESPNClient,
                                   fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    E3: Enriquece fixtures_df con features de lesiones para ambos equipos.

    Columnas añadidas:
        home_injured_count   → número de lesionados del equipo local
        home_injury_score    → score ponderado por severidad (local)
        home_out_count       → bajas seguras (local)
        away_injured_count   → ídem para visitante
        away_injury_score    → ídem
        away_out_count       → ídem

    Solo se ejecuta si ESPN_INJURIES_ENABLED=true en settings.
    Si el endpoint falla para un equipo, rellena con 0 sin romper el pipeline.
    """
    if not ESPN_INJURIES_ENABLED:
        log.debug("ESPN_INJURIES_ENABLED=false — saltando lesiones")
        return fixtures_df

    if fixtures_df.empty:
        return fixtures_df

    df = fixtures_df.copy()

    # Inicializar columnas con 0
    for prefix in ("home", "away"):
        df[f"{prefix}_injured_count"]  = 0
        df[f"{prefix}_injury_score"]   = 0.0
        df[f"{prefix}_out_count"]      = 0
        df[f"{prefix}_doubtful_count"] = 0

    n_ok = 0
    for idx, row in df.iterrows():
        slug = _ID_A_SLUG.get(int(row.get("league_id", 0)))
        if not slug:
            continue

        for prefix, id_col in [("home", "home_team_id"), ("away", "away_team_id")]:
            team_id = str(row.get(id_col, ""))
            if not team_id:
                continue
            try:
                injuries = get_team_injuries(client, slug, team_id)
                agg = _aggregate_injury_score(injuries)
                df.at[idx, f"{prefix}_injured_count"]  = agg["injured_count"]
                df.at[idx, f"{prefix}_injury_score"]   = agg["injury_score"]
                df.at[idx, f"{prefix}_out_count"]      = agg["out_count"]
                df.at[idx, f"{prefix}_doubtful_count"] = agg["doubtful_count"]
                if agg["injured_count"] > 0:
                    log.info(
                        f"Lesiones {row.get(f'{prefix}_team', '')}: "
                        f"{agg['out_count']} bajas seguras, "
                        f"{agg['doubtful_count']} dudosas"
                    )
                n_ok += 1
            except Exception as e:
                log.debug(f"Error lesiones {prefix} team {team_id}: {e}")

    log.info(f"Lesiones ESPN: {n_ok} equipos enriquecidos de {len(df)*2} posibles")
    return df


# ─── ESPN BPI PRE-PARTIDO (NUEVO v5 — E2) ────────────────────────────────────

def get_match_summary_bpi(client: ESPNClient, slug: str,
                           event_id: str) -> dict | None:
    """
    E2: Extrae ESPN BPI (win probability pre-partido) desde el game summary.
    Endpoint: /apis/site/v2/sports/soccer/{slug}/summary?event={id}

    ESPN BPI ('predictor.gameProjection') es una probabilidad independiente
    calculada por ESPN que podemos usar como feature adicional o para
    validar cruzadamente nuestras predicciones DC+XGBoost.

    Retorna:
        {
          "bpi_home_pct": float,   # probabilidad de victoria local (0-100)
          "bpi_away_pct": float,   # probabilidad de victoria visitante
          "bpi_home_prob": float,  # normalizado 0-1
          "bpi_away_prob": float,
          "source": "espn_bpi"
        }
    o None si no está disponible.
    """
    if not ESPN_BPI_ENABLED:
        return None

    url  = f"{ESPN_SITE_V2}/{slug}/summary"
    data = client.get(url, params={"event": event_id})
    if not data:
        return None

    try:
        predictor = data.get("predictor", {})
        home_proj = predictor.get("homeTeam", {}).get("gameProjection")
        away_proj = predictor.get("awayTeam", {}).get("teamChanceLoss")

        if home_proj is None:
            return None

        home_pct = float(str(home_proj).replace(",", "."))
        # teamChanceLoss del visitante = probabilidad de perder del visitante
        # = probabilidad de ganar del local → usarlo solo como validación cruzada
        # Calcular away como complemento simple (empate incluido en el residuo)
        away_pct = float(str(away_proj).replace(",", ".")) if away_proj else (100.0 - home_pct)

        return {
            "bpi_home_pct":  round(home_pct, 2),
            "bpi_away_pct":  round(away_pct, 2),
            "bpi_home_prob": round(home_pct / 100.0, 4),
            "bpi_away_prob": round(away_pct / 100.0, 4),
            "source": "espn_bpi",
        }
    except (TypeError, ValueError, KeyError) as e:
        log.debug(f"Error parseando BPI para evento {event_id}: {e}")
        return None


def enrich_fixtures_with_bpi(client: ESPNClient,
                              fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade ESPN BPI como feature a fixtures_df.

    Columnas añadidas:
        espn_bpi_home_prob  → probabilidad ESPN de victoria local (0-1)
        espn_bpi_away_prob  → probabilidad ESPN de victoria visitante (0-1)
        espn_bpi_available  → bool
    """
    if not ESPN_BPI_ENABLED or fixtures_df.empty:
        return fixtures_df

    df = fixtures_df.copy()
    df["espn_bpi_home_prob"]  = None
    df["espn_bpi_away_prob"]  = None
    df["espn_bpi_available"]  = False

    n_ok = 0
    for idx, row in df.iterrows():
        slug     = _ID_A_SLUG.get(int(row.get("league_id", 0)))
        event_id = str(row.get("fixture_id", ""))
        if not slug or not event_id:
            continue

        bpi = get_match_summary_bpi(client, slug, event_id)
        if bpi:
            df.at[idx, "espn_bpi_home_prob"] = bpi["bpi_home_prob"]
            df.at[idx, "espn_bpi_away_prob"] = bpi["bpi_away_prob"]
            df.at[idx, "espn_bpi_available"] = True
            n_ok += 1
            log.info(
                f"BPI {row.get('home_team','?')} vs {row.get('away_team','?')}: "
                f"home={bpi['bpi_home_pct']}% away={bpi['bpi_away_pct']}%"
            )

    log.info(f"ESPN BPI disponible: {n_ok}/{len(df)} partidos")
    return df


# ─── GOLEADORES DE LIGA (NUEVO v5 — E4) ──────────────────────────────────────

def get_league_top_scorers(client: ESPNClient, slug: str,
                            season: int = None) -> list[dict]:
    """
    E4: Descarga los goleadores y asistentes líderes de una liga.
    Endpoint: /v2/sports/soccer/leagues/{slug}/seasons/{season}/leaders
    o          /v2/sports/soccer/leagues/{slug}/leaders (temporada actual)

    Útil como contexto del rendimiento ofensivo de los equipos en la liga.
    Retorna lista de dicts: {player_name, team_name, goals, assists, stat_name}
    """
    if season:
        url = f"{ESPN_CORE_V2}/{slug}/seasons/{season}/leaders"
    else:
        url = f"{ESPN_CORE_V2}/{slug}/leaders"

    data = client.get(url)
    if not data:
        return []

    leaders = []
    for category in data.get("categories", []):
        stat_name = category.get("name", "")
        for leader in category.get("leaders", []):
            athlete_ref = leader.get("athlete", {})
            team_ref    = leader.get("team", {})
            leaders.append({
                "stat_name":   stat_name,
                "player_id":   str(athlete_ref.get("id", "")),
                "player_name": athlete_ref.get("displayName", ""),
                "team_id":     str(team_ref.get("id", "")),
                "team_name":   team_ref.get("displayName", ""),
                "value":       leader.get("value", 0),
                "display":     leader.get("displayValue", ""),
                "slug":        slug,
            })

    log.info(f"Top scorers {slug}: {len(leaders)} líderes estadísticos")
    return leaders


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
        if v >= 100:     return round(v / 100 + 1, 3)
        elif v <= -100:  return round(-100 / v + 1, 3)
        elif v > 1:      return round(v, 3)
        return None
    except (TypeError, ValueError):
        return None


def get_match_odds(client, slug: str, event_id: str) -> dict | None:
    """
    Extrae cuotas del Core API /odds.
    Itera providers por prioridad ascendente.
    """
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

            ou_line    = item.get("overUnder")
            over_odds  = item.get("overOdds")
            under_odds = item.get("underOdds")
            spread     = item.get("spread")
            spread_h   = home_odds_raw.get("spreadOdds")
            spread_a   = away_odds_raw.get("spreadOdds")

            open_data       = item.get("open") or {}
            open_total_line = (open_data.get("over") or {}).get("value")
            open_spread_h   = ((open_data.get("spread") or {}).get("home") or {}).get("line")

            return {
                "home":             h,
                "draw":             d,
                "away":             a,
                "provider":         item.get("provider", {}).get("name", ""),
                "total_line":       float(str(ou_line).replace(",", ".")) if ou_line else None,
                "over_odds":        _to_decimal(over_odds),
                "under_odds":       _to_decimal(under_odds),
                "spread_line":      float(str(spread).replace(",", ".")) if spread else None,
                "spread_home_odds": _to_decimal(spread_h),
                "spread_away_odds": _to_decimal(spread_a),
                "open_total_line":  float(str(open_total_line).replace(",", ".")) if open_total_line else None,
                "open_spread_home": float(str(open_spread_h).replace(",", ".")) if open_spread_h else None,
            }
        except Exception as e:
            log.debug(f"Error parseando odds item: {e}")
            continue

    return None


def enrich_fixtures_with_odds(client: ESPNClient,
                               fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquece fixtures_df con cuotas ESPN en tiempo real.
    Columnas añadidas: espn_odds_home/draw/away, espn_total_line, etc.
    """
    if fixtures_df.empty:
        return fixtures_df

    try:
        from src.espn_collector import _ID_A_SLUG
    except ImportError:
        from config.settings import LIGAS_ESPN, COMPETICIONES_NACIONALES_ESPN
        _todos = {**LIGAS_ESPN, **COMPETICIONES_NACIONALES_ESPN}
        _ID_A_SLUG = {v[0]: k for k, v in _todos.items()}

    df = fixtures_df.copy()

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
    """Play-by-play: goles, remates, tarjetas, sustituciones."""
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
    """Estima xG aproximado desde play-by-play."""
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
    Por defecto usa las últimas 3 temporadas.
    """
    if seasons is None:
        yr = date.today().year
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
    Construye histórico de partidos para una liga vía /schedule de cada equipo.
    v5: sin cambios en lógica, pero ahora soporta el catálogo ampliado de ligas.
    """
    log.info(f"Construyendo histórico ESPN: {league_name} ({slug})")

    team_ids = get_team_ids(client, slug)
    if not team_ids:
        log.warning(f"Sin equipos para {league_name}")
        return pd.DataFrame()

    if seasons is None:
        yr = date.today().year
        seasons = [yr - 2, yr - 1, yr]

    log.info(f"{league_name}: descargando temporadas {seasons}...")

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
            "B365H": None, "B365D": None, "B365A": None,
            "PSH":   None, "PSD":   None, "PSA":   None,
        }

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

    path = os.path.join(DATA_RAW, f"espn_{slug.replace('.', '_')}.csv")
    df.to_csv(path, index=False)
    log.info(f"{league_name}: {len(df)} partidos históricos ({seasons}) → {path}")
    return df


# ─── STANDINGS ────────────────────────────────────────────────────────────────

def get_standings(client: ESPNClient = None,
                  slugs: dict = None) -> dict[int, list]:
    """
    Posiciones por competición.
    CORRECCIÓN: usa ESPN_SITE_V2B (/apis/v2/) en lugar de /site/v2/ (devuelve {}).
    FIX B1/B2: _safe_int en todos los campos numéricos.
    v5: sin cambios de lógica — funciona igual con las nuevas ligas.
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

        if data.get("standings", {}).get("entries"):
            entry_lists.append(data["standings"]["entries"])
        for child in data.get("children", []):
            entries = child.get("standings", {}).get("entries", [])
            if entries:
                entry_lists.append(entries)

        for entries in entry_lists:
            for entry in entries:
                team  = entry.get("team", {})
                stats = {s["name"]: s.get("value", 0) for s in entry.get("stats", [])}

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


# ─── CONTEXTO DE MOTIVACIÓN (NUEVO v6 — F1/F2) ───────────────────────────────

# Número de partidos al final de la temporada en que se aplica "presión de corte"
_JORNADAS_FINALES = 6

# Umbral de puntos para considerar que un equipo está "en la pelea"
_PUNTOS_CERCA_CORTE = 4


def _compute_standings_features(entries: list[dict],
                                 league_id: int) -> dict[str, dict]:
    """
    Dado el listado de entradas de standings (ya parseadas por get_standings),
    calcula features de motivación para cada equipo.

    Lógica de cortes por liga:
      - Top 1          → title_race (pelea por el título)
      - Top 4 (EU) / Top 8 LATAM  → european_race / clasificacion_copa
      - Bottom 3 EU / Bottom 4 LATAM → relegation_threat (zona de descenso)

    Retorna dict {team_name_lower → feature_dict}.
    """
    if not entries:
        return {}

    # Ordenar por rank para calcular posiciones relativas
    entries_sorted = sorted(entries, key=lambda e: e.get("rank", 99))
    n_teams = len(entries_sorted)
    if n_teams == 0:
        return {}

    # Determinar zonas según tamaño de la liga
    # Ligas LATAM suelen tener 16-20 equipos; EU tiene 18-20
    is_latam = 500 <= league_id < 530
    relegation_zone   = 4 if is_latam else 3
    classification_zone = min(8, n_teams // 2) if is_latam else 4

    # Calcular partidos jugados mediana para estimar progreso de temporada
    played_values = [e.get("played", 0) for e in entries_sorted if e.get("played", 0) > 0]
    median_played = sorted(played_values)[len(played_values) // 2] if played_values else 0

    # Estimar total de jornadas según liga (aprox)
    # Liga BetPlay: ~36 | Brasileirao: ~38 | Arg: ~27 | EU grandes: ~38
    total_jornadas_est = {
        501: 36, 502: 27, 503: 38, 504: 30, 505: 30,
        506: 30, 507: 27, 508: 22, 509: 22, 510: 18,
        514: 38, 515: 36, 516: 36,  # copas UEFA (fase de grupos ~8)
    }.get(league_id, 34)

    jornadas_restantes = max(0, total_jornadas_est - median_played)
    season_progress    = round(median_played / total_jornadas_est, 3) if total_jornadas_est > 0 else 0.5
    es_tramo_final     = jornadas_restantes <= _JORNADAS_FINALES

    # Puntos del líder y del equipo en el límite de descenso
    pts_lider        = entries_sorted[0].get("points", 0) if entries_sorted else 0
    idx_relegacion   = n_teams - relegation_zone
    pts_corte_desc   = entries_sorted[idx_relegacion].get("points", 0) if idx_relegacion >= 0 else 0
    pts_corte_clasif = entries_sorted[classification_zone - 1].get("points", 0) if classification_zone <= n_teams else 0

    result: dict[str, dict] = {}

    for entry in entries_sorted:
        team_name = entry.get("team", "")
        if not team_name:
            continue

        rank   = entry.get("rank", n_teams)
        points = entry.get("points", 0)
        played = entry.get("played", 0)

        # ── Distancia a cada corte ─────────────────────────────────────────
        points_to_title    = pts_lider - points          # 0 si es el líder
        points_to_safe     = points - pts_corte_desc     # negativo = en zona de descenso
        points_to_clasif   = pts_corte_clasif - points   # negativo = ya clasificado

        # ── Flags de situación ────────────────────────────────────────────
        title_race       = int(rank == 1 or (points_to_title <= _PUNTOS_CERCA_CORTE and es_tramo_final))
        clasif_race      = int(rank <= classification_zone or
                               (points_to_clasif <= _PUNTOS_CERCA_CORTE and points_to_clasif >= 0))
        relegation_threat = int(rank > idx_relegacion or
                                (0 <= points_to_safe <= _PUNTOS_CERCA_CORTE))

        # ── Forma reciente desde el record W-D-L ──────────────────────────
        # Aproximada: (wins - losses) / played, normalizada a [-1, 1]
        won   = entry.get("won", 0)
        drawn = entry.get("drawn", 0)
        lost  = entry.get("lost", 0)
        pts_por_partido = round(points / played, 3) if played > 0 else 0.0
        forma_reciente  = round((won - lost) / played, 3) if played > 0 else 0.0

        # ── Score de intensidad compuesto ────────────────────────────────
        # Combina urgencia de situación + factor de tramo final
        # Rango: 0.0 (sin presión) → 1.5 (máxima presión, tramo final + descenso)
        intensidad = 0.0
        if relegation_threat:
            urgencia = min(1.0, max(0.0, 1.0 - points_to_safe / (_PUNTOS_CERCA_CORTE + 1)))
            intensidad += urgencia * (1.5 if es_tramo_final else 1.0)
        elif title_race:
            urgencia = min(1.0, max(0.0, 1.0 - points_to_title / (_PUNTOS_CERCA_CORTE + 1)))
            intensidad += urgencia * (1.2 if es_tramo_final else 0.8)
        elif clasif_race:
            urgencia = min(1.0, max(0.0, 1.0 - points_to_clasif / (_PUNTOS_CERCA_CORTE + 1)))
            intensidad += urgencia * (1.0 if es_tramo_final else 0.6)

        result[team_name.lower()] = {
            # Posición y puntos
            "standing_rank":          rank,
            "standing_points":        points,
            "standing_played":        played,
            "standing_pts_per_game":  pts_por_partido,
            "standing_goal_diff":     entry.get("goal_diff", 0),
            # Distancias a cortes (puntos)
            "points_to_title":        round(points_to_title, 1),
            "points_to_safety":       round(points_to_safe, 1),
            "points_to_clasif":       round(points_to_clasif, 1),
            # Flags de situación (0/1)
            "title_race":             title_race,
            "clasif_race":            clasif_race,
            "relegation_threat":      relegation_threat,
            # Contexto temporal
            "season_progress":        season_progress,
            "jornadas_restantes":     jornadas_restantes,
            "es_tramo_final":         int(es_tramo_final),
            # Forma implícita
            "forma_wdl":              forma_reciente,
            # Score de intensidad compuesto
            "motivation_score":       round(min(intensidad, 1.5), 4),
        }

    return result


def get_standings_context(client: ESPNClient,
                           slugs: dict = None) -> dict[int, dict[str, dict]]:
    """
    F1: Descarga standings y convierte a features de motivación por equipo.

    Retorna:
        {league_id: {team_name_lower: {feature_dict}}}

    Se llama una vez por día al inicio del pipeline y el resultado se pasa
    a enrich_fixtures_with_standings. No requiere fixture_id ni event_id.

    Usa el mismo endpoint que get_standings() — /apis/v2/ — que ya está
    confirmado como funcional (col.1, eng.1, etc.).
    """
    if not ESPN_STANDINGS_FEATURES:
        log.debug("ESPN_STANDINGS_FEATURES=false — saltando contexto de motivación")
        return {}

    if client is None:
        client = ESPNClient()
    if slugs is None:
        slugs = {k: v for k, v in LIGAS_ESPN.items() if k in LIGAS_ESPN_ACTIVAS}

    context: dict[int, dict[str, dict]] = {}

    for slug, (league_id, league_name) in slugs.items():
        url  = f"{ESPN_SITE_V2B}/{slug}/standings"
        data = client.get(url)
        if not data:
            log.debug(f"Sin standings para contexto: {league_name}")
            continue

        # Parsear entries — misma lógica que get_standings()
        raw_entries: list[dict] = []
        if data.get("standings", {}).get("entries"):
            raw_entries.extend(data["standings"]["entries"])
        for child in data.get("children", []):
            entries = child.get("standings", {}).get("entries", [])
            raw_entries.extend(entries)

        if not raw_entries:
            log.debug(f"Standings vacíos para {league_name}")
            continue

        # Convertir entries crudas al formato de get_standings()
        parsed_entries = []
        for entry in raw_entries:
            team  = entry.get("team", {})
            stats = {s["name"]: s.get("value", 0) for s in entry.get("stats", [])}

            overall_str = next(
                (s.get("displayValue", "") for s in entry.get("stats", [])
                 if s.get("name") == "overall"), ""
            )
            won = drawn = lost = 0
            if "-" in overall_str:
                parts = overall_str.split("-")
                if len(parts) == 3:
                    won, drawn, lost = (_safe_int(p) for p in parts)

            parsed_entries.append({
                "team":      team.get("displayName", ""),
                "team_id":   str(team.get("id", "")),
                "rank":      _safe_int(stats.get("rank", 0)),
                "points":    _safe_int(stats.get("points", 0)),
                "played":    _safe_int(stats.get("gamesPlayed", 0)),
                "won":       won,
                "drawn":     drawn,
                "lost":      lost,
                "goal_diff": _safe_int(stats.get("pointDifferential", 0)),
            })

        features = _compute_standings_features(parsed_entries, league_id)
        if features:
            context[league_id] = features
            log.info(
                f"Standings context {league_name}: "
                f"{len(features)} equipos · "
                f"amenazados={sum(1 for f in features.values() if f['relegation_threat'])} · "
                f"título={sum(1 for f in features.values() if f['title_race'])}"
            )

    return context


def enrich_fixtures_with_standings(fixtures_df: pd.DataFrame,
                                    standings_context: dict[int, dict[str, dict]],
                                    fuzzy_threshold: int = 80) -> pd.DataFrame:
    """
    F2: Enriquece fixtures_df con features de motivación usando standings_context.

    Parámetros:
        fixtures_df        → DataFrame con columnas home_team, away_team, league_id
        standings_context  → output de get_standings_context()
        fuzzy_threshold    → score mínimo para fuzzy match de nombres (0-100)

    Columnas añadidas por equipo (prefijo home_ / away_):
        standing_rank          → posición en tabla
        standing_points        → puntos
        standing_pts_per_game  → puntos por partido
        standing_goal_diff     → diferencia de goles
        points_to_safety       → puntos sobre la zona de descenso (neg = en descenso)
        points_to_clasif       → puntos para zona de clasificación (neg = ya clasificado)
        title_race             → 1 si pelea por el título
        clasif_race            → 1 si pelea por clasificación
        relegation_threat      → 1 si en riesgo de descenso
        motivation_score       → score compuesto 0-1.5

    Columnas diferenciales (home - away):
        rank_diff              → diferencia de posición (positivo = visitante mejor)
        points_diff_standing   → diferencia de puntos en tabla
        motivation_diff        → diferencia de motivation_score
        pressure_asymmetry     → |relegation_threat_home - relegation_threat_away|
    """
    if fixtures_df.empty or not standings_context:
        return fixtures_df

    try:
        from rapidfuzz import process as fz_process
    except ImportError:
        log.warning("rapidfuzz no disponible — standings context sin fuzzy match")
        fz_process = None

    df = fixtures_df.copy()

    # Inicializar columnas con NaN para distinguir "no encontrado" de "0"
    feature_keys = [
        "standing_rank", "standing_points", "standing_pts_per_game",
        "standing_goal_diff", "points_to_safety", "points_to_clasif",
        "title_race", "clasif_race", "relegation_threat",
        "season_progress", "es_tramo_final", "motivation_score",
    ]
    for prefix in ("home", "away"):
        for key in feature_keys:
            df[f"{prefix}_{key}"] = float("nan")

    for col in ("rank_diff", "points_diff_standing",
                "motivation_diff", "pressure_asymmetry"):
        df[col] = float("nan")

    n_matched = 0

    for idx, row in df.iterrows():
        league_id = int(row.get("league_id", 0))
        liga_ctx  = standings_context.get(league_id)
        if not liga_ctx:
            continue

        candidates = list(liga_ctx.keys())   # nombres en minúscula

        for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
            team_name = str(row.get(team_col, "")).lower().strip()
            if not team_name:
                continue

            # Búsqueda exacta primero
            features = liga_ctx.get(team_name)

            # Fuzzy match si no hay coincidencia exacta
            if features is None and fz_process and candidates:
                match, score, _ = fz_process.extractOne(team_name, candidates)
                if score >= fuzzy_threshold:
                    features = liga_ctx[match]
                    log.debug(
                        f"Standings fuzzy: '{team_name}' → '{match}' "
                        f"(score={score}, liga={league_id})"
                    )

            if features is None:
                log.debug(
                    f"Standings context: equipo '{team_name}' no encontrado "
                    f"en liga {league_id}"
                )
                continue

            for key in feature_keys:
                if key in features:
                    df.at[idx, f"{prefix}_{key}"] = features[key]
            n_matched += 1

        # ── Diferenciales ─────────────────────────────────────────────────
        h_rank  = df.at[idx, "home_standing_rank"]
        a_rank  = df.at[idx, "away_standing_rank"]
        h_pts   = df.at[idx, "home_standing_points"]
        a_pts   = df.at[idx, "away_standing_points"]
        h_mot   = df.at[idx, "home_motivation_score"]
        a_mot   = df.at[idx, "away_motivation_score"]
        h_rel   = df.at[idx, "home_relegation_threat"]
        a_rel   = df.at[idx, "away_relegation_threat"]

        import math
        if not math.isnan(h_rank) and not math.isnan(a_rank):
            df.at[idx, "rank_diff"]           = float(h_rank - a_rank)
        if not math.isnan(h_pts) and not math.isnan(a_pts):
            df.at[idx, "points_diff_standing"] = float(h_pts - a_pts)
        if not math.isnan(h_mot) and not math.isnan(a_mot):
            df.at[idx, "motivation_diff"]      = float(h_mot - a_mot)
        if not math.isnan(h_rel) and not math.isnan(a_rel):
            df.at[idx, "pressure_asymmetry"]   = abs(float(h_rel) - float(a_rel))

    log.info(
        f"Standings enrichment: {n_matched} equipos enriquecidos "
        f"de {len(df) * 2} posibles"
    )
    return df


# ─── RESULTADOS DEL DÍA ───────────────────────────────────────────────────────

def get_results_espn(target_date: str = None,
                     slugs: dict = None) -> dict:
    """
    Resultados finalizados del día.
    v5: sin cambios — funciona igual con el catálogo ampliado.
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
    """Polling en vivo enriquecido con win probability del Core API."""
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
    """Carga histórico de selecciones desde disco."""
    frames  = []
    raw_dir = Path(DATA_RAW)

    for pattern in ["nacional_results_*.json", "live_results_*.json"]:
        for json_file in sorted(raw_dir.glob(pattern)):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                frames.append(pd.DataFrame(data))
            except Exception as e:
                log.warning(f"Error cargando {json_file}: {e}")

    for slug in SLUGS_SELECCIONES:
        csv_path = raw_dir / f"espn_{slug.replace('.', '_')}.csv"
        if csv_path.exists():
            try:
                frames.append(pd.read_csv(csv_path))
            except Exception as e:
                log.warning(f"Error cargando {csv_path}: {e}")

    manual_path = os.path.join(DATA_RAW, "nacional_historical.csv")
    if os.path.exists(manual_path):
        frames.append(pd.read_csv(manual_path))

    if not frames:
        log.warning("Sin datos históricos de selecciones nacionales.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    date_col = "date" if "date" in df.columns else "match_date"
    df["match_date"] = pd.to_datetime(df[date_col], errors="coerce")

    if "status" in df.columns:
        df = df[df["status"].isin(FINISHED_STATUSES)].copy()

    df = df.dropna(subset=["match_date", "home_goals", "away_goals"])
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])

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

    print("\n=== TEST LESIONES Col.1 ===")
    team_ids = get_team_ids(client, "col.1")
    if team_ids:
        first_name, first_id = next(iter(team_ids.items()))
        injuries = get_team_injuries(client, "col.1", first_id)
        print(f"  {first_name}: {len(injuries)} lesionados")
        for inj in injuries[:3]:
            print(f"    {inj['player_name']} ({inj['status']}) — {inj['injury_type']}")

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