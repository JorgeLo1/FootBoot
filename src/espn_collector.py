"""
espn_collector.py  —  reemplaza nacional_collector.py
Fuente única de fixtures ESPN para selecciones Y clubes.

Exporta los mismos símbolos que scheduler_nacional.py importa:
    ApifootballClient, get_fixtures_hoy, get_standings,
    load_historical_nacional, run_live_polling, run_daily, FINISHED_STATUSES

Exporta adicionalmente para scheduler.py (clubes):
    get_fixtures_today_espn   → reemplaza get_fixtures_today() de _01_data_collector
    get_results_espn          → reemplaza get_results_fdorg() de _01_data_collector
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
from config.settings import DATA_RAW, DATA_PROCESSED, LOGS_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── SSOT DE SLUGS ESPN ───────────────────────────────────────────────────────
#
# Formato: slug_espn → (league_id_interno, nombre_display)
#
# league_id_interno: mismos IDs que config/settings.py LIGAS y
# COMPETICIONES_NACIONALES para que Supabase, features y model_engine
# no cambien absolutamente nada.
#
# Verificados en PowerShell + doc pseudo-r/Public-ESPN-API marzo 2026.

SLUGS_SELECCIONES: dict[str, tuple[int, str]] = {
    # Activo solo en ventanas FIFA (oct, nov, mar) — da HTTP 400 el resto del año
    "fifa.world_cup_qualifying.conmebol": (361, "Eliminatorias CONMEBOL"),
    # Activo todo el año (próximos fixtures del Mundial 2026 ya visibles)
    "fifa.world"                        : (1,   "Copa del Mundo"),
    # Activo solo durante el torneo — inactivo entre ediciones
    "conmebol.america"                  : (271, "Copa América"),
}

SLUGS_CLUBES: dict[str, tuple[int, str]] = {
    "col.1"                : (239, "Liga BetPlay"),
    "conmebol.libertadores": (13,  "Copa Libertadores"),
    "uefa.champions"       : (2,   "Champions League"),
    "arg.1"                : (128, "Liga Profesional Argentina"),
    "bra.1"                : (71,  "Brasileirão Serie A"),
}

# Unificado — para búsquedas y polling sin importar categoría
TODOS_LOS_SLUGS: dict[str, tuple[int, str]] = {
    **SLUGS_SELECCIONES,
    **SLUGS_CLUBES,
}

# Inverso: league_id → slug (para _get_fixture_live)
_ID_A_SLUG: dict[int, str] = {v[0]: k for k, v in TODOS_LOS_SLUGS.items()}

ESPN_SITE_V2 = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# ─── STATUSES ────────────────────────────────────────────────────────────────

LIVE_STATUSES      = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"}
FINISHED_STATUSES  = {"FT", "AET", "PEN"}
SCHEDULED_STATUSES = {"TBD", "NS"}

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

POLL_INTERVAL_S = 5 * 60
POLL_INTERVAL_L = 10 * 60

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; FOOTBOT/1.0)",
    "Accept":     "application/json",
})


# ─── CLIENTE ESPN ─────────────────────────────────────────────────────────────

class ApifootballClient:
    """
    Reemplaza ApifootballClient de nacional_collector.py.
    Sin API key. Sin límite oficial.
    Interfaz idéntica al original.
    """

    def __init__(self):
        self._calls = 0
        log.info("ESPNClient listo — sin API key, sin límite oficial")

    @property
    def remaining(self):
        return None

    def get(self, url: str, params: dict = None) -> dict | None:
        try:
            resp = _SESSION.get(url, params=params, timeout=12)
            resp.raise_for_status()
            self._calls += 1
            time.sleep(0.5)
            return resp.json()
        except requests.exceptions.HTTPError as e:
            # 400/404 son esperados para slugs fuera de temporada — no es un error real
            level = logging.DEBUG if e.response.status_code in (400, 404) else logging.WARNING
            log.log(level, f"HTTP {e.response.status_code} → {url}")
            return None
        except Exception as e:
            log.warning(f"Error ESPN GET {url}: {e}")
            return None

    def status(self) -> str:
        return f"ESPN API: {self._calls} llamadas esta sesión"


# ─── PARSERS INTERNOS ─────────────────────────────────────────────────────────

def _norm_status(raw: str) -> str:
    return _STATUS_MAP.get(raw, raw)


def _parse_fixture(event: dict, league_id: int, league_name: str) -> dict | None:
    """
    Convierte un evento del scoreboard ESPN al schema interno.
    Schema de salida idéntico al _parse_fixture() de nacional_collector.py.
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
        home_score = home.get("score")
        away_score = away.get("score")
        venue_data = comp.get("venue", {})

        return {
            "fixture_id":   str(event.get("id", "")),
            "date":         event.get("date", ""),
            "timestamp":    event.get("date", ""),
            "status":       status,
            "status_long":  raw_status,
            "elapsed":      elapsed,
            "league_id":    league_id,
            "league_name":  league_name,
            "season":       event.get("season", {}).get("year"),
            "round":        (comp.get("notes") or [{}])[0].get("headline", ""),
            "home_team":    home.get("team", {}).get("displayName", ""),
            "home_team_id": home.get("team", {}).get("id", ""),
            "away_team":    away.get("team", {}).get("displayName", ""),
            "away_team_id": away.get("team", {}).get("id", ""),
            "home_goals":   int(home_score) if home_score is not None else None,
            "away_goals":   int(away_score) if away_score is not None else None,
            "home_ht":      None,
            "away_ht":      None,
            "home_ft":      int(home_score) if status in FINISHED_STATUSES and home_score is not None else None,
            "away_ft":      int(away_score) if status in FINISHED_STATUSES and away_score is not None else None,
            "venue":        venue_data.get("fullName", ""),
            "venue_city":   (venue_data.get("address") or {}).get("city", ""),
        }
    except (KeyError, TypeError, ValueError) as e:
        log.debug(f"Error parseando fixture ESPN: {e}")
        return None


# ─── FUNCIÓN GENÉRICA DE SCOREBOARD ──────────────────────────────────────────

def _fetch_scoreboard(client: ApifootballClient,
                      slug: str,
                      league_id: int,
                      league_name: str,
                      date_param: str) -> list[dict]:
    """
    Descarga el scoreboard ESPN para un slug y fecha.
    date_param: YYYYMMDD sin guiones (formato ESPN).

    Algunos slugs (ej. fifa.world_cup_qualifying.conmebol) devuelven HTTP 400
    fuera de sus ventanas activas. Se tratan como "sin partidos" sin abortar
    el pipeline — el cliente ya loguea el warning a nivel HTTP.
    """
    url  = f"{ESPN_SITE_V2}/{slug}/scoreboard"
    data = client.get(url, params={"dates": date_param})

    # None = error HTTP (400/404/5xx) → slug inactivo esta semana, ignorar
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


# ─── FIXTURES SELECCIONES ─────────────────────────────────────────────────────

def get_fixtures_hoy(client: ApifootballClient,
                     target_date: str = None) -> list[dict]:
    """
    Fixtures del día para SELECCIONES.
    Mismo contrato que get_fixtures_hoy() de nacional_collector.py.
    """
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    date_param = target_date.replace("-", "")
    fixtures   = []

    for slug, (league_id, league_name) in SLUGS_SELECCIONES.items():
        fixtures.extend(_fetch_scoreboard(client, slug, league_id, league_name, date_param))

    if fixtures:
        path = os.path.join(DATA_RAW, f"nacional_fixtures_{target_date}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fixtures, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"Fixtures selecciones guardados → {path}")

    return fixtures


# ─── FIXTURES CLUBES ──────────────────────────────────────────────────────────

def get_fixtures_today_espn(client: ApifootballClient = None,
                             target_date: str = None) -> pd.DataFrame:
    """
    Fixtures del día para CLUBES.
    Reemplaza get_fixtures_today() de _01_data_collector.py.
    Retorna pd.DataFrame con el mismo schema que usaba el pipeline de clubes.
    """
    if client is None:
        client = ApifootballClient()
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    date_param   = target_date.replace("-", "")
    all_fixtures = []

    for slug, (league_id, league_name) in SLUGS_CLUBES.items():
        all_fixtures.extend(_fetch_scoreboard(client, slug, league_id, league_name, date_param))

    df = pd.DataFrame(all_fixtures)
    if not df.empty:
        path = os.path.join(DATA_RAW, f"fixtures_{target_date}.csv")
        df.to_csv(path, index=False)
        log.info(f"Fixtures clubes guardados: {len(df)} partidos → {path}")
    else:
        log.info("No hay partidos de clubes hoy.")

    return df


# ─── RESULTADOS DEL DÍA ───────────────────────────────────────────────────────

def get_results_espn(target_date: str = None,
                     slugs: dict = None) -> dict:
    """
    Resultados finalizados del día.
    Reemplaza get_results_fdorg() de _01_data_collector.py.
    Retorna {(home_team, away_team): {home_goals, away_goals, fixture_id, status}}

    slugs: por defecto TODOS_LOS_SLUGS.
           Pasar SLUGS_SELECCIONES o SLUGS_CLUBES para filtrar.
    """
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")
    if slugs is None:
        slugs = TODOS_LOS_SLUGS

    client     = ApifootballClient()
    date_param = target_date.replace("-", "")
    results    = {}

    for slug, (league_id, league_name) in slugs.items():
        for f in _fetch_scoreboard(client, slug, league_id, league_name, date_param):
            if f["status"] in FINISHED_STATUSES:
                results[(f["home_team"], f["away_team"])] = {
                    "home_goals": f["home_goals"] or 0,
                    "away_goals": f["away_goals"] or 0,
                    "fixture_id": f["fixture_id"],
                    "status":     f["status"],
                }

    log.info(f"Resultados ESPN ({target_date}): {len(results)} finalizados")
    return results


# ─── STANDINGS ────────────────────────────────────────────────────────────────

def get_standings(client: ApifootballClient,
                  slugs: dict = None) -> dict[int, list]:
    """
    Posiciones por competición.
    Mismo contrato que get_standings() de nacional_collector.py.
    slugs: por defecto SLUGS_SELECCIONES.
    """
    if slugs is None:
        slugs = SLUGS_SELECCIONES

    standings = {}

    for slug, (league_id, league_name) in slugs.items():
        url  = f"https://site.api.espn.com/apis/v2/sports/soccer/{slug}/standings"
        data = client.get(url)
        if not data:
            log.warning(f"Sin standings ESPN: {league_name}")
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
                rows.append({
                    "league_id":     league_id,
                    "league_name":   league_name,
                    "rank":          int(stats.get("rank", 0)),
                    "team":          team.get("displayName", ""),
                    "team_id":       team.get("id", ""),
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

        standings[league_id] = rows
        log.info(f"Standings {league_name}: {len(rows)} equipos")

    all_rows = [r for rows in standings.values() for r in rows]
    if all_rows:
        path = os.path.join(DATA_RAW, "nacional_standings.csv")
        pd.DataFrame(all_rows).to_csv(path, index=False)
        log.info(f"Standings guardados → {path}")

    return standings


# ─── POLLING EN VIVO ─────────────────────────────────────────────────────────

def _get_fixture_live(client: ApifootballClient, fixture: dict) -> dict | None:
    """
    Refresca un partido re-descargando el scoreboard del día.
    1 llamada por competición, no por partido.
    """
    slug = _ID_A_SLUG.get(fixture.get("league_id"))
    if not slug:
        return None

    today_param       = date.today().strftime("%Y%m%d")
    url               = f"{ESPN_SITE_V2}/{slug}/scoreboard"
    data              = client.get(url, params={"dates": today_param})
    if not data:
        return None

    fid                  = str(fixture["fixture_id"])
    league_id, league_name = TODOS_LOS_SLUGS.get(slug, (fixture["league_id"], ""))

    for event in data.get("events", []):
        if str(event.get("id", "")) == fid:
            return _parse_fixture(event, league_id, league_name)

    return None


def run_live_polling(client: ApifootballClient,
                     fixtures: list[dict],
                     on_update=None) -> list[dict]:
    """Mismo contrato que el original."""
    pending  = {f["fixture_id"]: f for f in fixtures
                if f["status"] in LIVE_STATUSES | SCHEDULED_STATUSES}
    finished = {f["fixture_id"]: f for f in fixtures
                if f["status"] in FINISHED_STATUSES}

    if not pending:
        log.info("Sin partidos activos para polling.")
        return list(finished.values())

    log.info(f"Polling ESPN: {len(pending)} partidos pendientes")
    last_scores = {fid: (f.get("home_goals"), f.get("away_goals"))
                   for fid, f in pending.items()}

    while pending:
        n_active = sum(1 for f in pending.values() if f["status"] in LIVE_STATUSES)
        time.sleep(POLL_INTERVAL_S if n_active <= 2 else POLL_INTERVAL_L)

        to_remove = []
        for fid, fixture in list(pending.items()):
            updated = _get_fixture_live(client, fixture)
            if not updated:
                continue

            new_score = (updated.get("home_goals"), updated.get("away_goals"))
            if new_score != last_scores.get(fid) and None not in new_score:
                log.info(
                    f"GOL: {updated['home_team']} {new_score[0]}-"
                    f"{new_score[1]} {updated['away_team']} "
                    f"(min {updated['elapsed']})"
                )
                last_scores[fid] = new_score
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
    _save_results(all_results)
    return all_results


def _save_results(fixtures: list[dict]):
    if not fixtures:
        return
    today = date.today().strftime("%Y-%m-%d")
    path  = os.path.join(DATA_RAW, f"nacional_results_{today}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"Resultados guardados → {path}")


# ─── HISTÓRICO SELECCIONES ────────────────────────────────────────────────────

def load_historical_nacional() -> pd.DataFrame:
    """Sin cambios — lee los mismos JSON/CSV que la versión anterior."""
    frames  = []
    raw_dir = Path(DATA_RAW)

    for json_file in sorted(raw_dir.glob("nacional_results_*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            frames.append(pd.DataFrame(data))
        except Exception as e:
            log.warning(f"Error cargando {json_file}: {e}")

    csv_path = os.path.join(DATA_RAW, "nacional_historical.csv")
    if os.path.exists(csv_path):
        frames.append(pd.read_csv(csv_path))

    if not frames:
        log.warning("Sin datos históricos de selecciones nacionales.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"].isin(FINISHED_STATUSES)].copy()
    df["match_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["match_date", "home_goals", "away_goals"])
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])
    df = df.drop_duplicates(subset=["fixture_id"])

    log.info(f"Histórico selecciones: {len(df)} partidos")
    return df.sort_values("match_date").reset_index(drop=True)


# ─── run_daily ────────────────────────────────────────────────────────────────

def run_daily(target_date: str = None) -> dict:
    """Punto de entrada para scheduler_nacional.py --mode calendar."""
    client = ApifootballClient()
    log.info(client.status())

    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    fixtures  = get_fixtures_hoy(client, target_date)
    standings = get_standings(client)

    log.info(f"run_daily: {len(fixtures)} fixtures · {client.status()}")
    return {"fixtures": fixtures, "standings": standings, "date": target_date}


# ─── PRUEBA RÁPIDA ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = ApifootballClient()
    today  = date.today().strftime("%Y-%m-%d")

    print("\n=== SELECCIONES ===")
    for f in get_fixtures_hoy(client, today):
        print(f"  [{f['status']:3s}] {f['league_name']:35s} "
              f"{f['home_team']} vs {f['away_team']}")

    print("\n=== CLUBES ===")
    for _, row in get_fixtures_today_espn(client, today).iterrows():
        print(f"  [{row['status']:3s}] {row['league_name']:35s} "
              f"{row['home_team']} vs {row['away_team']}")