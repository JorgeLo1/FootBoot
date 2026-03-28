import os
import sys
import json
import time
import logging
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    API_FOOTBALL_KEY, DATA_RAW, DATA_PROCESSED, LOGS_DIR,
    API_USAGE_FILE,
)
from src.utils import ApiRateLimiter
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)
 
# ─── CONSTANTES ───────────────────────────────────────────────────────────────
 
API_BASE        = "https://v3.football.api-sports.io"
DAILY_LIMIT     = 100
POLL_INTERVAL_S = 5 * 60      # 5 min (≤2 partidos activos)
POLL_INTERVAL_L = 10 * 60     # 10 min (≥3 partidos activos)
MATCH_WINDOW    = 115          # minutos desde kickoff hasta considerar finalizado
 
# Competiciones de selecciones nacionales
COMPETICIONES = {
    361: {"nombre": "Eliminatorias CONMEBOL", "temporada": 2026},
    271: {"nombre": "Copa América",           "temporada": 2024},
    1:   {"nombre": "Copa del Mundo",         "temporada": 2026},
}
 
# Archivo de uso diario separado para API-Football (nacional)
NACIONAL_USAGE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "api_usage_nacional.json",
)
 
# Statuses de partido "activo" (en juego)
LIVE_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"}
# Statuses de partido "finalizado"
FINISHED_STATUSES = {"FT", "AET", "PEN"}
# Statuses de partido "programado"
SCHEDULED_STATUSES = {"TBD", "NS"}
 
 
# ─── CLIENTE API ─────────────────────────────────────────────────────────────
 
class ApifootballClient:
    """
    Cliente HTTP para API-Football con rate limiting integrado.
    Cada instancia comparte el mismo contador de uso diario.
    """
 
    def __init__(self):
        if not API_FOOTBALL_KEY:
            raise ValueError(
                "API_FOOTBALL_KEY no configurada. "
                "Obtén una key gratuita en api-sports.io y añádela al .env"
            )
        self._headers = {
            "x-rapidapi-host": "v3.football.api-sports.io",
            "x-rapidapi-key":  API_FOOTBALL_KEY,
        }
        os.makedirs(LOGS_DIR, exist_ok=True)
        self._limiter = ApiRateLimiter(
            usage_file=NACIONAL_USAGE_FILE,
            daily_limit=DAILY_LIMIT,
        )
 
    @property
    def remaining(self) -> int:
        return self._limiter.remaining
 
    def get(self, endpoint: str, params: dict = None,
            cost: int = 1) -> dict | None:
        """
        Realiza un GET a la API consumiendo `cost` requests del presupuesto.
        Retorna el JSON de respuesta o None si hay error / límite alcanzado.
        """
        if not self._limiter.can_request(cost):
            log.warning(
                f"Límite diario alcanzado "
                f"({self._limiter._state['count']}/{DAILY_LIMIT}). "
                f"Abortando request a {endpoint}."
            )
            return None
 
        url = f"{API_BASE}/{endpoint.lstrip('/')}"
        try:
            resp = requests.get(
                url, headers=self._headers,
                params=params, timeout=10,
            )
            resp.raise_for_status()
            self._limiter.consume(cost)
            data = resp.json()
 
            errors = data.get("errors", {})
            if errors:
                log.error(f"API error en {endpoint}: {errors}")
                return None
 
            return data
 
        except requests.exceptions.HTTPError as e:
            log.error(f"HTTP {e.response.status_code} en {endpoint}: {e}")
            return None
        except Exception as e:
            log.error(f"Error en {endpoint}: {e}")
            return None
 
    def status(self) -> str:
        return self._limiter.status()
 
 
# ─── FIXTURES ────────────────────────────────────────────────────────────────
 
def get_fixtures_hoy(client: ApifootballClient,
                     target_date: str = None) -> list[dict]:
    """
    Obtiene todos los partidos de selecciones para una fecha.
    1 request por competición activa → máximo 3 requests.
    """
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")
 
    fixtures = []
    for league_id, meta in COMPETICIONES.items():
        data = client.get(
            "fixtures",
            params={
                "league":  league_id,
                "season":  meta["temporada"],
                "date":    target_date,
            },
        )
        if not data:
            continue
 
        for f in data.get("response", []):
            fixture = _parse_fixture(f, league_id, meta["nombre"])
            if fixture:
                fixtures.append(fixture)
 
        log.info(
            f"{meta['nombre']}: "
            f"{sum(1 for x in fixtures if x['league_id'] == league_id)} partidos"
        )
 
    # Guardar en disco
    if fixtures:
        path = os.path.join(DATA_RAW, f"nacional_fixtures_{target_date}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fixtures, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"Fixtures guardados → {path}")
 
    return fixtures
 
 
def get_fixture_live(client: ApifootballClient,
                     fixture_id: int) -> dict | None:
    """
    Obtiene el estado en tiempo real de un partido específico.
    1 request. Usado durante el polling.
    """
    data = client.get("fixtures", params={"id": fixture_id})
    if not data or not data.get("response"):
        return None
    return _parse_fixture(data["response"][0],
                          league_id=None, league_name=None)
 
 
def _parse_fixture(raw: dict,
                   league_id: int | None,
                   league_name: str | None) -> dict | None:
    """Normaliza un fixture de la API al formato interno."""
    try:
        f    = raw["fixture"]
        lg   = raw.get("league", {})
        home = raw["teams"]["home"]
        away = raw["teams"]["away"]
        goals= raw.get("goals", {})
        score= raw.get("score", {})
 
        return {
            "fixture_id":   f["id"],
            "date":         f["date"],
            "timestamp":    f["timestamp"],
            "status":       f["status"]["short"],
            "status_long":  f["status"]["long"],
            "elapsed":      f["status"].get("elapsed") or 0,
            "league_id":    league_id or lg.get("id"),
            "league_name":  league_name or lg.get("name", ""),
            "season":       lg.get("season"),
            "round":        lg.get("round", ""),
            "home_team":    home["name"],
            "home_team_id": home["id"],
            "away_team":    away["name"],
            "away_team_id": away["id"],
            "home_goals":   goals.get("home"),
            "away_goals":   goals.get("away"),
            "home_ht":      (score.get("halftime") or {}).get("home"),
            "away_ht":      (score.get("halftime") or {}).get("away"),
            "home_ft":      (score.get("fulltime") or {}).get("home"),
            "away_ft":      (score.get("fulltime") or {}).get("away"),
            "venue":        (f.get("venue") or {}).get("name", ""),
            "venue_city":   (f.get("venue") or {}).get("city", ""),
        }
    except (KeyError, TypeError) as e:
        log.debug(f"Error parseando fixture: {e}")
        return None
 
 
# ─── POSICIONES ──────────────────────────────────────────────────────────────
 
def get_standings(client: ApifootballClient) -> dict[int, list]:
    """
    Descarga tabla de posiciones para cada competición.
    1 request por competición → máximo 3 requests.
    Retorna {league_id: [filas de posición]}.
    """
    standings = {}
    for league_id, meta in COMPETICIONES.items():
        data = client.get(
            "standings",
            params={
                "league": league_id,
                "season": meta["temporada"],
            },
        )
        if not data:
            continue
 
        rows = []
        for standing_group in data.get("response", []):
            for group in standing_group.get("league", {}).get("standings", []):
                for entry in group:
                    rows.append({
                        "league_id":    league_id,
                        "league_name":  meta["nombre"],
                        "rank":         entry.get("rank"),
                        "team":         entry["team"]["name"],
                        "team_id":      entry["team"]["id"],
                        "points":       entry.get("points", 0),
                        "played":       entry.get("all", {}).get("played", 0),
                        "won":          entry.get("all", {}).get("win",    0),
                        "drawn":        entry.get("all", {}).get("draw",   0),
                        "lost":         entry.get("all", {}).get("lose",   0),
                        "goals_for":    entry.get("all", {}).get("goals", {}).get("for",     0),
                        "goals_against":entry.get("all", {}).get("goals", {}).get("against", 0),
                        "goal_diff":    entry.get("goalsDiff", 0),
                        "form":         entry.get("form", ""),
                        "group":        entry.get("group", ""),
                        "updated_at":   str(date.today()),
                    })
 
        standings[league_id] = rows
        log.info(f"Posiciones {meta['nombre']}: {len(rows)} equipos")
 
    # Persistir
    all_rows = [r for rows in standings.values() for r in rows]
    if all_rows:
        path = os.path.join(DATA_RAW, "nacional_standings.csv")
        pd.DataFrame(all_rows).to_csv(path, index=False)
        log.info(f"Posiciones guardadas → {path}")
 
    return standings
 
 
# ─── ESTADÍSTICAS POR EQUIPO ─────────────────────────────────────────────────
 
def get_team_stats(client: ApifootballClient,
                   team_id: int,
                   league_id: int,
                   season: int) -> dict | None:
    """
    Stats de temporada de un equipo en una competición.
    1 request. Se llama solo durante el análisis pre-partido (no polling).
    """
    data = client.get(
        "teams/statistics",
        params={
            "team":   team_id,
            "league": league_id,
            "season": season,
        },
    )
    if not data or not data.get("response"):
        return None
 
    r   = data["response"]
    fix = r.get("fixtures", {})
    gls = r.get("goals", {})
 
    played_home  = fix.get("played",  {}).get("home",  0) or 0
    played_away  = fix.get("played",  {}).get("away",  0) or 0
    played_total = fix.get("played",  {}).get("total", 0) or 0
 
    gf_home  = gls.get("for",     {}).get("total", {}).get("home",  0) or 0
    gf_away  = gls.get("for",     {}).get("total", {}).get("away",  0) or 0
    ga_home  = gls.get("against", {}).get("total", {}).get("home",  0) or 0
    ga_away  = gls.get("against", {}).get("total", {}).get("away",  0) or 0
 
    return {
        "team_id":           team_id,
        "league_id":         league_id,
        "season":            season,
        "played_home":       played_home,
        "played_away":       played_away,
        "played_total":      played_total,
        "gf_home":           gf_home,
        "gf_away":           gf_away,
        "ga_home":           ga_home,
        "ga_away":           ga_away,
        "avg_gf_home":       round(gf_home / max(played_home, 1), 3),
        "avg_gf_away":       round(gf_away / max(played_away, 1), 3),
        "avg_ga_home":       round(ga_home / max(played_home, 1), 3),
        "avg_ga_away":       round(ga_away / max(played_away, 1), 3),
        "biggest_win_home":  r.get("biggest", {}).get("wins",   {}).get("home",  ""),
        "biggest_win_away":  r.get("biggest", {}).get("wins",   {}).get("away",  ""),
        "biggest_lose_home": r.get("biggest", {}).get("loses",  {}).get("home",  ""),
        "biggest_lose_away": r.get("biggest", {}).get("loses",  {}).get("away",  ""),
        "clean_sheets_home": r.get("clean_sheet", {}).get("home",  0) or 0,
        "clean_sheets_away": r.get("clean_sheet", {}).get("away",  0) or 0,
        "form":              r.get("form", ""),
    }
 
 
def get_h2h(client: ApifootballClient,
            home_team_id: int,
            away_team_id: int,
            last: int = 10) -> list[dict]:
    """
    Head-to-head entre dos selecciones (últimos N partidos).
    1 request. Se llama solo en el análisis pre-partido.
    """
    data = client.get(
        "fixtures/headtohead",
        params={
            "h2h":  f"{home_team_id}-{away_team_id}",
            "last": last,
        },
    )
    if not data:
        return []
 
    results = []
    for f in data.get("response", []):
        parsed = _parse_fixture(f, None, None)
        if parsed:
            results.append(parsed)
 
    return results
 
 
# ─── POLLING EN VIVO ─────────────────────────────────────────────────────────
 
def _poll_interval(n_active: int) -> int:
    """Devuelve el intervalo de polling en segundos según partidos activos."""
    return POLL_INTERVAL_S if n_active <= 2 else POLL_INTERVAL_L
 
 
def run_live_polling(client: ApifootballClient,
                     fixtures: list[dict],
                     on_update=None) -> list[dict]:
    """
    Hace polling de los partidos activos hasta que todos terminen.
 
    on_update: callable(fixture_dict) llamado cada vez que un partido
               actualiza su marcador. Útil para notificaciones Telegram.
 
    Retorna la lista de fixtures con resultados finales.
    """
    # Filtrar solo los de hoy que están programados o en curso
    pending = {
        f["fixture_id"]: f for f in fixtures
        if f["status"] in LIVE_STATUSES | SCHEDULED_STATUSES
    }
    finished = {
        f["fixture_id"]: f for f in fixtures
        if f["status"] in FINISHED_STATUSES
    }
 
    if not pending:
        log.info("Ningún partido activo para hacer polling.")
        return list(finished.values())
 
    log.info(f"Iniciando polling: {len(pending)} partidos pendientes/activos")
    last_scores = {fid: (f.get("home_goals"), f.get("away_goals"))
                   for fid, f in pending.items()}
 
    while pending:
        n_active = sum(
            1 for f in pending.values()
            if f["status"] in LIVE_STATUSES
        )
        interval = _poll_interval(n_active)
 
        if client.remaining < 5:
            log.warning(
                "Menos de 5 requests restantes. "
                "Deteniendo polling para preservar presupuesto."
            )
            break
 
        time.sleep(interval)
 
        to_remove = []
        for fid in list(pending.keys()):
            updated = get_fixture_live(client, fid)
            if not updated:
                continue
 
            # Detectar gol
            new_score = (updated.get("home_goals"), updated.get("away_goals"))
            if new_score != last_scores.get(fid) and None not in new_score:
                log.info(
                    f"GOL: {updated['home_team']} {new_score[0]} - "
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
                    f"Partido finalizado: "
                    f"{updated['home_team']} {updated['home_goals']} - "
                    f"{updated['away_goals']} {updated['away_team']}"
                )
                finished[fid] = updated
                to_remove.append(fid)
 
        for fid in to_remove:
            del pending[fid]
 
        log.info(
            f"Polling: {len(pending)} activos | "
            f"{len(finished)} finalizados | "
            f"{client.remaining} req restantes"
        )
 
    # Guardar resultados finales
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
 
 
# ─── HISTORIAL ───────────────────────────────────────────────────────────────
 
def load_historical_nacional() -> pd.DataFrame:
    """
    Carga todos los resultados históricos de selecciones desde disco.
    Combina archivos daily results + cualquier CSV pre-cargado.
    """
    frames = []
 
    # Archivos JSON diarios
    raw_dir = Path(DATA_RAW)
    for json_file in sorted(raw_dir.glob("nacional_results_*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            frames.append(pd.DataFrame(data))
        except Exception as e:
            log.warning(f"Error cargando {json_file}: {e}")
 
    # CSV pre-cargado (opcional, para datos históricos más antiguos)
    csv_path = os.path.join(DATA_RAW, "nacional_historical.csv")
    if os.path.exists(csv_path):
        frames.append(pd.read_csv(csv_path))
 
    if not frames:
        log.warning("Sin datos históricos de selecciones nacionales.")
        return pd.DataFrame()
 
    df = pd.concat(frames, ignore_index=True)
 
    # Normalizar
    df = df[df["status"].isin(FINISHED_STATUSES)].copy()
    df["match_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["match_date", "home_goals", "away_goals"])
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])
    df = df.drop_duplicates(subset=["fixture_id"])
 
    log.info(f"Histórico selecciones: {len(df)} partidos")
    return df.sort_values("match_date").reset_index(drop=True)
 
 
# ─── RUNNER DIARIO ───────────────────────────────────────────────────────────
 
def run_daily(target_date: str = None) -> dict:
    """
    Punto de entrada del scheduler_nacional.py.
    Descarga fixtures y posiciones del día. No hace polling (eso es run_live).
    """
    if not API_FOOTBALL_KEY:
        log.error("API_FOOTBALL_KEY no configurada.")
        return {}
 
    client = ApifootballClient()
    log.info(f"API-Football Nacional: {client.status()}")
 
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")
 
    fixtures  = get_fixtures_hoy(client, target_date)
    standings = get_standings(client)
 
    log.info(
        f"Run daily completado: {len(fixtures)} fixtures | "
        f"{client.status()}"
    )
 
    return {
        "fixtures":  fixtures,
        "standings": standings,
        "date":      target_date,
    }
 
 
if __name__ == "__main__":
    result = run_daily()
    log.info(f"Resultado: {len(result.get('fixtures', []))} partidos hoy")