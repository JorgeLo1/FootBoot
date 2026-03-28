"""
espn_collector.py — v2
Fuente unificada ESPN para FOOTBOT. Sin API key. Sin límite oficial.

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

def get_match_odds(client: ESPNClient, slug: str,
                   event_id: str) -> dict | None:
    """
    Cuotas reales del partido desde el Core API.
    Endpoint: /events/{id}/competitions/{id}/odds

    Retorna dict {home, draw, away, provider, source} en formato decimal,
    o None si no hay cuotas disponibles para este partido/liga.

    Nota: ESPN suele tener cuotas para ligas grandes (PL, LaLiga, UCL).
    Para Col.1 y ligas menores puede devolver lista vacía.
    """
    url  = f"{ESPN_CORE_V2}/{slug}/events/{event_id}/competitions/{event_id}/odds"
    data = client.get(url)
    if not data:
        return None

    items = data.get("items", [])
    if not items:
        return None

    # Ordenar por prioridad del proveedor (menor número = mayor prioridad)
    def _priority(item):
        return item.get("provider", {}).get("priority", 999)

    best = sorted(items, key=_priority)[0]

    try:
        home_odds = best.get("homeTeamOdds", {})
        away_odds = best.get("awayTeamOdds", {})
        draw_odds = best.get("drawOdds", {})

        def to_decimal(val) -> float | None:
            """Convierte moneyline americano a cuota decimal."""
            if val is None:
                return None
            try:
                v = float(val)
                if v >= 100:     return round(v / 100 + 1, 3)   # +200 → 3.00
                elif v <= -100:  return round(-100 / v + 1, 3)  # -150 → 1.67
                elif v > 1:      return round(v, 3)              # ya decimal
                return None
            except (TypeError, ValueError):
                return None

        # ESPN puede dar moneyLine, odds o value según el proveedor
        h = to_decimal(
            home_odds.get("moneyLine") or home_odds.get("odds") or home_odds.get("value")
        )
        a = to_decimal(
            away_odds.get("moneyLine") or away_odds.get("odds") or away_odds.get("value")
        )
        d = to_decimal(
            draw_odds.get("moneyLine") or draw_odds.get("odds") or draw_odds.get("value")
        )

        if not h or not a:
            return None

        return {
            "home":     h,
            "draw":     d or 3.40,
            "away":     a,
            "provider": best.get("provider", {}).get("name", "ESPN"),
            "source":   "espn_odds",
        }

    except (TypeError, ValueError, KeyError) as e:
        log.debug(f"Error parseando odds {event_id}: {e}")
        return None


def enrich_fixtures_with_odds(client: ESPNClient,
                               fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade cuotas reales ESPN a los fixtures del día.
    Se llama en el scheduler ANTES del value detector.

    Añade columnas:
        espn_odds_home, espn_odds_draw, espn_odds_away,
        espn_odds_provider, espn_odds_available (bool)
    """
    if fixtures_df.empty:
        return fixtures_df

    df = fixtures_df.copy()
    df["espn_odds_home"]      = None
    df["espn_odds_draw"]      = None
    df["espn_odds_away"]      = None
    df["espn_odds_provider"]  = None
    df["espn_odds_available"] = False

    for idx, row in df.iterrows():
        slug     = _ID_A_SLUG.get(int(row.get("league_id", 0)))
        event_id = str(row.get("fixture_id", ""))
        if not slug or not event_id:
            continue

        odds = get_match_odds(client, slug, event_id)
        if odds:
            df.at[idx, "espn_odds_home"]      = odds["home"]
            df.at[idx, "espn_odds_draw"]      = odds["draw"]
            df.at[idx, "espn_odds_away"]      = odds["away"]
            df.at[idx, "espn_odds_provider"]  = odds["provider"]
            df.at[idx, "espn_odds_available"] = True
            log.info(
                f"Odds {row['home_team']} vs {row['away_team']}: "
                f"{odds['home']}/{odds['draw']}/{odds['away']} ({odds['provider']})"
            )
        else:
            log.debug(f"Sin odds ESPN: {row.get('home_team')} vs {row.get('away_team')}")

    n = int(df["espn_odds_available"].sum())
    log.info(f"Odds ESPN disponibles: {n}/{len(df)} partidos")
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
        h = float(latest.get("homeWinPercentage", 0))
        a = float(latest.get("awayWinPercentage", 0))
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

    Los type_text de ESPN varían por liga — el matching es por keywords.
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
                      team_id: str, team_name: str) -> list[dict]:
    """
    Histórico de partidos de un equipo (pasados y futuros).
    Endpoint: /teams/{id}/schedule (Site API v2)
    """
    url  = f"{ESPN_SITE_V2}/{slug}/teams/{team_id}/schedule"
    data = client.get(url)
    if not data:
        return []

    events = []
    for event in data.get("events", []):
        try:
            comp        = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c["homeAway"] == "home"), None)
            away = next((c for c in competitors if c["homeAway"] == "away"), None)
            if not home or not away:
                continue

            raw_status = comp.get("status", {}).get("type", {}).get("name", "")
            status     = _norm_status(raw_status)

            events.append({
                "match_id":     str(event["id"]),
                "date":         event.get("date", ""),
                "slug":         slug,
                "home_team":    home["team"]["displayName"],
                "home_team_id": str(home["team"]["id"]),
                "away_team":    away["team"]["displayName"],
                "away_team_id": str(away["team"]["id"]),
                "home_score":   home.get("score"),
                "away_score":   away.get("score"),
                "status":       status,
            })
        except (KeyError, TypeError):
            continue

    log.debug(f"{team_name}: {len(events)} partidos en schedule")
    return events


# ─── HISTÓRICO POR LIGA ───────────────────────────────────────────────────────

def build_historical_espn(client: ESPNClient,
                           slug: str,
                           league_id: int,
                           league_name: str,
                           fetch_plays: bool = False,
                           max_per_team: int = 30) -> pd.DataFrame:
    """
    Construye el histórico de una liga ESPN desde cero.

    Flujo:
      1. /teams          → IDs de todos los equipos
      2. /teams/{id}/schedule → partidos (se deduplica por match_id)
      3. Opcional /plays  → xG aproximado (fetch_plays=True, +1 req/partido)

    Cuotas históricas: ESPN no las tiene → columnas B365H/PSH = None.
    Las cuotas en tiempo real se obtienen en el día del partido via get_match_odds().

    Guarda el resultado en data/raw/espn_{slug_normalizado}.csv
    """
    log.info(f"Construyendo histórico ESPN: {league_name} ({slug})")

    team_ids = get_team_ids(client, slug)
    if not team_ids:
        log.warning(f"Sin equipos para {league_name}")
        return pd.DataFrame()

    # Recopilar eventos únicos via schedules de cada equipo
    seen:   set[str]   = set()
    events: list[dict] = []

    for team_name, team_id in team_ids.items():
        for event in get_team_schedule(client, slug, team_id, team_name)[:max_per_team]:
            if event["match_id"] not in seen:
                seen.add(event["match_id"])
                events.append(event)
        time.sleep(0.3)

    log.info(f"{league_name}: {len(events)} partidos únicos encontrados")

    # Solo finalizados para entrenamiento
    finished = [
        e for e in events
        if e["status"] in FINISHED_STATUSES
        and e["home_score"] is not None
        and e["away_score"] is not None
    ]

    rows = []
    for i, event in enumerate(finished):
        row = {
            "match_id":     event["match_id"],
            "match_date":   pd.to_datetime(event["date"], utc=True, errors="coerce"),
            "league_id":    league_id,
            "league_name":  league_name,
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
    log.info(f"{league_name}: {len(df)} partidos históricos → {path}")
    return df


# ─── STANDINGS (fix: usa /apis/v2/ no /site/v2/) ─────────────────────────────

def get_standings(client: ESPNClient = None,
                  slugs: dict = None) -> dict[int, list]:
    """
    Posiciones por competición.
    CORRECCIÓN: usa ESPN_SITE_V2B (/apis/v2/) en lugar de /site/v2/
    que devuelve {} vacío para soccer standings.
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
                stats = {s["name"]: s.get("value", 0) for s in entry.get("stats", [])}
                rows.append({
                    "league_id":     league_id,
                    "league_name":   league_name,
                    "rank":          int(stats.get("rank", 0)),
                    "team":          team.get("displayName", ""),
                    "team_id":       str(team.get("id", "")),
                    "points":        int(stats.get("points", 0)),
                    "played":        int(stats.get("gamesPlayed", 0)),
                    "won":           int(stats.get("wins", 0)),
                    "drawn":         int(stats.get("ties", 0)),
                    "lost":          int(stats.get("losses", 0)),
                    "goals_for":     int(stats.get("pointsFor", 0)),
                    "goals_against": int(stats.get("pointsAgainst", 0)),
                    "goal_diff":     int(stats.get("pointDifferential", 0)),
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

    print("\n=== HISTÓRICO Col.1 (max 5 partidos/equipo) ===")
    hist = build_historical_espn(
        client, "col.1", 501, "Liga BetPlay",
        fetch_plays=False, max_per_team=5,
    )
    print(f"  {len(hist)} partidos descargados")
    if not hist.empty:
        print(hist[["match_date", "home_team", "home_goals",
                     "away_goals", "away_team"]].tail(5).to_string(index=False))