"""
_01_data_collector.py
Descarga y normaliza todas las fuentes de datos.

FIXES v3:
  - download_football_data: retry con backoff exponencial para WinError 10054
    (servidor Football-Data.co.uk cerrando conexiones). Usa Session con
    headers apropiados y pausa entre ligas para evitar bloqueo.
  - _get_fixtures_fdorg: mismo retry para fixtures del día.
  - get_weather_for_fixture: delegado a utils.py (coordenadas LATAM).

Fuentes:
  - Fixtures del día        : football-data.org (free, 10 req/min) +
                              ESPN API (sin key, backup/ampliación)
  - Datos históricos EU     : Football-Data.co.uk (CSVs gratuitos + cuotas)
  - Datos históricos otros  : ESPN API (via build_historical_espn)
  - ELO ratings             : ClubElo.com (sin key)
  - xG                      : StatsBomb Open Data (sin key)
  - Clima                   : Open-Meteo (sin key)
"""

import os
import sys
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    FOOTBALL_DATA_ORG_KEY, FOOTBALL_DATA_ORG_URL,
    LIGAS, DATA_RAW, DATA_STATSBOMB,
    FOOTBALL_DATA_URL, FOOTBALL_DATA_SEASONS,
    CLUBELO_URL, CURRENT_SEASON,
    LIGAS_ESPN, LIGAS_ESPN_ACTIVAS,
)
from src.utils import get_weather_for_fixture, ApiRateLimiter  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ─── SESSION CON RETRY ───────────────────────────────────────────────────────

def _make_session(retries: int = 4, backoff: float = 1.5) -> requests.Session:
    """
    Session con retry automático y backoff exponencial.

    Motivo: Football-Data.co.uk cierra conexiones activamente (WinError 10054)
    cuando recibe requests muy seguidas o con user-agent genérico.
    Con retry + backoff + headers realistas esto se resuelve en la mayoría
    de los casos sin intervención manual.

    retries=4, backoff=1.5 → esperas de 1.5, 3, 6, 12 segundos.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection":      "keep-alive",
    })
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://",  adapter)
    session.mount("https://", adapter)
    return session


# Sesión reutilizable para Football-Data.co.uk
_FD_SESSION = _make_session()


# ─── FIXTURES DEL DÍA ────────────────────────────────────────────────────────

def get_fixtures_today() -> pd.DataFrame:
    """
    Obtiene los partidos del día combinando:
      1. football-data.org para las 7 ligas europeas (fuente principal con cuotas)
      2. ESPN API para ligas adicionales (Col, Arg, Bra, Libertadores, etc.)

    Retorna un DataFrame unificado con el schema estándar de FOOTBOT.
    """
    frames = []

    # ── Fuente 1: football-data.org (7 ligas EU) ──────────────────────────
    fdorg_df = _get_fixtures_fdorg()
    if not fdorg_df.empty:
        frames.append(fdorg_df)

    # ── Fuente 2: ESPN (ligas adicionales) ────────────────────────────────
    espn_df = _get_fixtures_espn_today()
    if not espn_df.empty:
        ligas_eu_ids = set(LIGAS.keys())
        espn_df = espn_df[~espn_df["league_id"].isin(ligas_eu_ids)]
        if not espn_df.empty:
            frames.append(espn_df)

    if not frames:
        log.info("No hay partidos hoy en ninguna fuente.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(
        subset=["home_team", "away_team"],
        keep="first"
    ).reset_index(drop=True)

    today = date.today().strftime("%Y-%m-%d")
    path  = os.path.join(DATA_RAW, f"fixtures_{today}.csv")
    df.to_csv(path, index=False)
    log.info(f"Fixtures totales: {len(df)} partidos → {path}")
    return df


def _get_fixtures_fdorg() -> pd.DataFrame:
    """
    Fixtures desde football-data.org (7 ligas EU, 10 req/min).
    FIX: usa _make_session() con retry para WinError 10054.
    """
    if not FOOTBALL_DATA_ORG_KEY:
        log.warning(
            "FOOTBALL_DATA_ORG_KEY no configurada. "
            "Registrate en football-data.org (gratis) y añádela al .env"
        )
        return pd.DataFrame()

    today   = date.today().strftime("%Y-%m-%d")
    headers = {"X-Auth-Token": FOOTBALL_DATA_ORG_KEY}
    rows    = []
    session = _make_session()

    for league_id, (league_name, _, fdorg_id) in LIGAS.items():
        for attempt in range(3):
            try:
                resp = session.get(
                    f"{FOOTBALL_DATA_ORG_URL}/competitions/{fdorg_id}/matches",
                    headers=headers,
                    params={"dateFrom": today, "dateTo": today},
                    timeout=15,
                )
                resp.raise_for_status()

                for match in resp.json().get("matches", []):
                    if match["status"] not in ("SCHEDULED", "TIMED", "IN_PLAY", "FINISHED"):
                        continue
                    rows.append({
                        "fixture_id":  match["id"],
                        "date":        match["utcDate"],
                        "league_id":   league_id,
                        "league_name": league_name,
                        "home_team":   match["homeTeam"]["name"],
                        "away_team":   match["awayTeam"]["name"],
                        "venue":       "",
                        "venue_city":  "",
                        "status":      match["status"],
                        "source":      "fdorg",
                    })
                log.info(f"✓ Fixtures {league_name}: OK")
                break  # éxito — salir del loop de reintentos

            except Exception as e:
                wait = 2 ** attempt * 3
                log.warning(
                    f"Intento {attempt+1}/3 fallido para fixtures {league_name}: {e} "
                    f"— reintentando en {wait}s"
                )
                if attempt < 2:
                    time.sleep(wait)
                else:
                    log.error(f"Fixtures {league_name}: abandonado tras 3 intentos")

        time.sleep(6)  # respetar 10 req/min del free tier

    return pd.DataFrame(rows)


def _get_fixtures_espn_today() -> pd.DataFrame:
    """Fixtures desde ESPN para ligas activas no cubiertas por fd.org."""
    try:
        from src.espn_collector import ESPNClient, get_fixtures_today as espn_get_today
        client = ESPNClient(delay=0.5)
        slugs_activos = {
            k: v for k, v in LIGAS_ESPN.items()
            if k in LIGAS_ESPN_ACTIVAS
        }
        df = espn_get_today(client, slugs=slugs_activos)
        if not df.empty:
            df["source"] = "espn"
        return df
    except Exception as e:
        log.warning(f"ESPN fixtures falló (no crítico): {e}")
        return pd.DataFrame()


# ─── RESULTADOS DEL DÍA ──────────────────────────────────────────────────────

def get_results_fdorg(target_date: str) -> dict:
    """
    Resultados finalizados de football-data.org para una fecha.
    FIX: usa session con retry para WinError 10054.
    Retorna {(home_team, away_team): {home_goals, away_goals, fixture_id, status}}
    """
    if not FOOTBALL_DATA_ORG_KEY:
        return {}

    headers = {"X-Auth-Token": FOOTBALL_DATA_ORG_KEY}
    results = {}
    session = _make_session()

    for league_id, (league_name, _, fdorg_id) in LIGAS.items():
        for attempt in range(3):
            try:
                resp = session.get(
                    f"{FOOTBALL_DATA_ORG_URL}/competitions/{fdorg_id}/matches",
                    headers=headers,
                    params={"dateFrom": target_date, "dateTo": target_date,
                            "status": "FINISHED"},
                    timeout=15,
                )
                resp.raise_for_status()

                for match in resp.json().get("matches", []):
                    score = match.get("score", {})
                    full  = score.get("fullTime", {})
                    hg    = full.get("home") or 0
                    ag    = full.get("away") or 0
                    results[(match["homeTeam"]["name"], match["awayTeam"]["name"])] = {
                        "home_goals": int(hg),
                        "away_goals": int(ag),
                        "fixture_id": match["id"],
                        "status":     match["status"],
                    }
                break  # éxito

            except Exception as e:
                wait = 2 ** attempt * 3
                log.warning(
                    f"Intento {attempt+1}/3 fallido resultados {league_name}: {e} "
                    f"— reintentando en {wait}s"
                )
                if attempt < 2:
                    time.sleep(wait)

        time.sleep(6)

    log.info(f"Resultados fd.org: {len(results)} partidos finalizados")
    return results


def get_results_today(target_date: str = None) -> dict:
    """
    Resultados del día fusionando fd.org (EU) + ESPN (resto).
    Usado por _05_result_updater como fuente unificada.
    """
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    results = {}

    try:
        results.update(get_results_fdorg(target_date))
    except Exception as e:
        log.warning(f"fd.org resultados falló: {e}")

    try:
        from src.espn_collector import get_results_espn, TODOS_LOS_SLUGS, LIGAS_ESPN
        ligas_eu_ids = set(LIGAS.keys())
        slugs_extra  = {
            k: v for k, v in LIGAS_ESPN.items()
            if v[0] not in ligas_eu_ids
        }
        espn_results = get_results_espn(target_date, slugs=slugs_extra)
        for key, val in espn_results.items():
            if key not in results:
                results[key] = val
    except Exception as e:
        log.warning(f"ESPN resultados falló (no crítico): {e}")

    log.info(f"Resultados totales ({target_date}): {len(results)} partidos")
    return results


# ─── DATOS HISTÓRICOS EU (Football-Data.co.uk) ───────────────────────────────

def download_football_data() -> dict:
    """
    Descarga CSVs de Football-Data.co.uk para las 7 ligas EU.

    FIX: retry con backoff exponencial para WinError 10054.
    El servidor cierra conexiones cuando recibe requests muy seguidas.
    Estrategia:
      - Session con User-Agent de browser real
      - 3 intentos por CSV con espera 3s, 6s, 12s
      - Pausa de 2s entre ligas (además del retry)
    """
    dfs = {}

    for league_id, (league_name, fd_code, _) in LIGAS.items():
        frames = []

        for season in FOOTBALL_DATA_SEASONS:
            url = f"{FOOTBALL_DATA_URL}/{season}/{fd_code}.csv"

            for attempt in range(3):
                try:
                    # Usar _FD_SESSION con User-Agent de browser
                    resp = _FD_SESSION.get(url, timeout=20)
                    resp.raise_for_status()

                    # Leer CSV desde el contenido en memoria (evita re-request)
                    import io
                    df = pd.read_csv(
                        io.StringIO(resp.content.decode("latin-1")),
                        on_bad_lines="skip"
                    )
                    df["season"]      = season
                    df["league_name"] = league_name
                    frames.append(df)
                    log.info(f"✓ {league_name} {season}: {len(df)} partidos")
                    break  # éxito

                except Exception as e:
                    wait = (2 ** attempt) * 3  # 3s, 6s, 12s
                    if attempt < 2:
                        log.warning(
                            f"Intento {attempt+1}/3 fallido {league_name} {season}: {e} "
                            f"— reintentando en {wait}s"
                        )
                        time.sleep(wait)
                    else:
                        log.warning(f"No disponible {league_name} {season}: {e}")

            # Pausa corta entre temporadas para no saturar
            time.sleep(0.5)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            path = os.path.join(DATA_RAW, f"fd_{fd_code}.csv")
            combined.to_csv(path, index=False)
            dfs[fd_code] = combined
            log.info(f"→ {fd_code}: {len(combined)} partidos totales")

        # Pausa entre ligas
        time.sleep(2)

    return dfs


# ─── DATOS HISTÓRICOS ESPN (ligas sin cobertura EU) ──────────────────────────

def download_espn_historical(fetch_plays: bool = False,
                              max_per_team: int = None,
                              seasons: list[int] = None) -> dict:
    """
    Descarga histórico ESPN para las ligas activas que no cubre fd.co.uk.
    Se llama durante el re-entrenamiento semanal (lunes).

    FIX: pasa seasons a build_historical_espn para datos multi-temporada.
    max_per_team ya no es necesario — se mantiene por compatibilidad.
    """
    try:
        from src.espn_collector import ESPNClient, build_historical_espn
    except ImportError as e:
        log.error(f"No se pudo importar espn_collector: {e}")
        return {}

    client = ESPNClient(delay=0.5)
    dfs    = {}

    eu_slugs = {"eng.1", "esp.1", "ger.1", "ita.1", "fra.1", "ned.1", "por.1"}

    for slug, (league_id, league_name) in LIGAS_ESPN.items():
        if slug not in LIGAS_ESPN_ACTIVAS:
            continue
        if slug in eu_slugs:
            log.info(f"Saltando {league_name} — ya cubierta por fd.co.uk")
            continue

        try:
            df = build_historical_espn(
                client, slug, league_id, league_name,
                fetch_plays=fetch_plays,
                seasons=seasons,
            )
            if not df.empty:
                dfs[slug] = df
                log.info(f"✓ ESPN {league_name}: {len(df)} partidos")
        except Exception as e:
            log.warning(f"Error ESPN {league_name}: {e}")

    return dfs


def load_espn_historical() -> pd.DataFrame:
    """
    Carga todos los CSVs ESPN guardados en data/raw/espn_*.csv.
    """
    frames = []
    raw_dir = Path(DATA_RAW)

    for slug, (league_id, league_name) in LIGAS_ESPN.items():
        if slug not in LIGAS_ESPN_ACTIVAS:
            continue
        path = raw_dir / f"espn_{slug.replace('.', '_')}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                frames.append(df)
                log.debug(f"ESPN cargado: {league_name} ({len(df)} partidos)")
            except Exception as e:
                log.warning(f"Error cargando ESPN {path}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["match_date"] = pd.to_datetime(combined["match_date"], errors="coerce")

    if "status" in combined.columns:
        combined = combined[combined["status"] == "FT"].copy()

    combined = combined.dropna(subset=["match_date", "home_goals", "away_goals"])
    combined["home_goals"] = pd.to_numeric(combined["home_goals"], errors="coerce")
    combined["away_goals"] = pd.to_numeric(combined["away_goals"], errors="coerce")
    combined = combined.dropna(subset=["home_goals", "away_goals"])

    log.info(f"ESPN histórico cargado: {len(combined)} partidos de {len(frames)} ligas")
    return combined.sort_values("match_date").reset_index(drop=True)


# ─── CUOTAS HISTÓRICAS (Football-Data.co.uk) ─────────────────────────────────

def get_best_closing_odds(home_team: str, away_team: str,
                           historical: pd.DataFrame) -> dict | None:
    """
    Busca las últimas cuotas de cierre reales para un enfrentamiento.
    Prioridad: Pinnacle > Bet365 > Betway.
    Solo disponible para ligas cubiertas por Football-Data.co.uk.
    """
    from src._02_feature_builder import normalize_team_name

    h = normalize_team_name(home_team)
    a = normalize_team_name(away_team)

    if "home_team_norm" not in historical.columns:
        return None

    mask   = (historical["home_team_norm"] == h) & (historical["away_team_norm"] == a)
    recent = historical[mask].sort_values("match_date", ascending=False).head(3)

    for col_h, col_d, col_a in [
        ("PSH",   "PSD",   "PSA"),
        ("B365H", "B365D", "B365A"),
        ("BWH",   "BWD",   "BWA"),
    ]:
        if not all(c in recent.columns for c in [col_h, col_d, col_a]):
            continue
        valid = recent[[col_h, col_d, col_a]].dropna()
        if valid.empty:
            continue
        row = valid.iloc[0]
        try:
            return {
                "home":    float(row[col_h]),
                "draw":    float(row[col_d]),
                "away":    float(row[col_a]),
                "over25":  float(recent["B365>2.5"].dropna().iloc[0])
                           if "B365>2.5" in recent.columns and
                           not recent["B365>2.5"].dropna().empty else None,
                "under25": float(recent["B365<2.5"].dropna().iloc[0])
                           if "B365<2.5" in recent.columns and
                           not recent["B365<2.5"].dropna().empty else None,
                "source":  col_h[:3],
            }
        except (ValueError, TypeError):
            continue

    return None


# ─── STATSBOMB ───────────────────────────────────────────────────────────────

def download_statsbomb_data() -> dict:
    """Descarga datos de StatsBomb Open Data (gratis, sin API key)."""
    try:
        from statsbombpy import sb

        competitions = sb.competitions()
        target = competitions[
            competitions["competition_name"].isin([
                "Premier League", "La Liga", "Bundesliga",
                "Serie A", "Ligue 1", "Champions League",
            ])
        ]

        all_shots   = []
        all_summary = []

        for _, comp in target.iterrows():
            comp_id   = comp["competition_id"]
            season_id = comp["season_id"]
            try:
                matches = sb.matches(competition_id=comp_id, season_id=season_id)
                log.info(
                    f"StatsBomb: {comp['competition_name']} "
                    f"{comp['season_name']} → {len(matches)} partidos"
                )
                for _, match in matches.iterrows():
                    match_id = match["match_id"]
                    try:
                        events = sb.events(match_id=match_id)
                        shots  = events[events["type"] == "Shot"].copy()
                        if not shots.empty:
                            shots["match_id"]     = match_id
                            shots["home_team"]    = match["home_team"]
                            shots["away_team"]    = match["away_team"]
                            shots["is_home_team"] = shots["team"] == match["home_team"]
                            shots["competition"]  = comp["competition_name"]
                            shots["season"]       = comp["season_name"]
                            all_shots.append(
                                shots[["match_id", "team", "player",
                                       "shot_statsbomb_xg", "shot_outcome",
                                       "home_team", "away_team", "is_home_team",
                                       "competition", "season"]]
                            )
                        all_summary.append({
                            "match_id":     match_id,
                            "home_team":    match["home_team"],
                            "away_team":    match["away_team"],
                            "home_score":   match["home_score"],
                            "away_score":   match["away_score"],
                            "competition":  comp["competition_name"],
                            "season":       comp["season_name"],
                            "match_date":   match["match_date"],
                        })
                    except Exception:
                        continue
            except Exception as e:
                log.warning(f"Error StatsBomb {comp['competition_name']}: {e}")

        result = {}
        if all_shots:
            df   = pd.concat(all_shots, ignore_index=True)
            path = os.path.join(DATA_STATSBOMB, "shots_xg.csv")
            df.to_csv(path, index=False)
            result["shots"] = df
            log.info(f"StatsBomb shots: {len(df)} registros")

        if all_summary:
            df   = pd.DataFrame(all_summary)
            path = os.path.join(DATA_STATSBOMB, "match_summary.csv")
            df.to_csv(path, index=False)
            result["summary"] = df
            log.info(f"StatsBomb resumen: {len(df)} partidos")

        return result

    except Exception as e:
        log.error(f"Error general StatsBomb: {e}")
        return {}


# ─── ELO RATINGS ─────────────────────────────────────────────────────────────

def download_elo_ratings() -> pd.DataFrame:
    """Descarga ratings ELO de ClubElo.com con fallback a días anteriores."""
    session = _make_session()
    for days_back in range(0, 4):
        target = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            resp = session.get(f"{CLUBELO_URL}/{target}", timeout=15)
            resp.raise_for_status()
            import io
            df   = pd.read_csv(io.StringIO(resp.text))
            path = os.path.join(DATA_RAW, "elo_ratings.csv")
            df.to_csv(path, index=False)
            log.info(f"ELO ratings: {len(df)} equipos (hace {days_back}d)")
            return df
        except Exception:
            continue

    path = os.path.join(DATA_RAW, "elo_ratings.csv")
    if os.path.exists(path):
        log.warning("Usando ELO ratings del último archivo en caché.")
        return pd.read_csv(path)

    log.error("No se pudieron obtener ELO ratings.")
    return pd.DataFrame()


# ─── RUNNER PRINCIPAL ────────────────────────────────────────────────────────

def run():
    log.info("═══ FOOTBOT · Recolección de datos ═══")
    fixtures = get_fixtures_today()
    fd_data  = download_football_data()
    espn_h   = download_espn_historical()
    sb_data  = download_statsbomb_data()
    elo      = download_elo_ratings()
    log.info("═══ Recolección completada ═══")
    return {
        "fixtures":        fixtures,
        "football_data":   fd_data,
        "espn_historical": espn_h,
        "statsbomb":       sb_data,
        "elo":             elo,
    }


if __name__ == "__main__":
    run()