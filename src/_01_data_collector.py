"""
_01_data_collector.py
Descarga y normaliza todas las fuentes de datos.

Fuente de fixtures: football-data.org (free tier permanente, sin expiración).
Fuente de datos históricos + cuotas: Football-Data.co.uk (CSVs gratuitos).
Fuente de ELO: ClubElo.com (sin API key).
Fuente de clima: Open-Meteo (sin API key).
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    FOOTBALL_DATA_ORG_KEY, FOOTBALL_DATA_ORG_URL,
    LIGAS, DATA_RAW, DATA_STATSBOMB,
    FOOTBALL_DATA_URL, FOOTBALL_DATA_SEASONS,
    CLUBELO_URL, CURRENT_SEASON,
)
from src.utils import get_weather_for_fixture, ApiRateLimiter  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ─── FIXTURES DEL DÍA (football-data.org) ────────────────────────────────────

def get_fixtures_today() -> pd.DataFrame:
    """
    Obtiene los partidos del día desde football-data.org.
    Free tier permanente: 10 req/min → time.sleep(6) entre ligas.
    """
    if not FOOTBALL_DATA_ORG_KEY:
        log.error(
            "FOOTBALL_DATA_ORG_KEY no configurada. "
            "Regístrate en football-data.org (gratis) y añádela al .env"
        )
        return pd.DataFrame()

    today   = date.today().strftime("%Y-%m-%d")
    headers = {"X-Auth-Token": FOOTBALL_DATA_ORG_KEY}

    all_fixtures = []
    for league_id, (league_name, _, fdorg_id) in LIGAS.items():
        try:
            resp = requests.get(
                f"{FOOTBALL_DATA_ORG_URL}/competitions/{fdorg_id}/matches",
                headers=headers,
                params={"dateFrom": today, "dateTo": today},
                timeout=10,
            )
            resp.raise_for_status()

            for match in resp.json().get("matches", []):
                # Solo partidos programados o en juego
                if match["status"] not in ("SCHEDULED", "TIMED", "IN_PLAY", "FINISHED"):
                    continue
                all_fixtures.append({
                    "fixture_id":  match["id"],
                    "date":        match["utcDate"],
                    "league_id":   league_id,
                    "league_name": league_name,
                    "home_team":   match["homeTeam"]["name"],
                    "away_team":   match["awayTeam"]["name"],
                    "venue":       "",
                    "venue_city":  "",
                    "status":      match["status"],
                })
            log.info(f"✓ Fixtures {league_name}: OK")
        except Exception as e:
            log.warning(f"Error fixtures {league_name}: {e}")

        time.sleep(6)  # respetar 10 req/min del free tier

    df = pd.DataFrame(all_fixtures)
    if not df.empty:
        path = os.path.join(DATA_RAW, f"fixtures_{today}.csv")
        df.to_csv(path, index=False)
        log.info(f"Fixtures guardados: {len(df)} partidos → {path}")
    else:
        log.info("No hay partidos hoy en las ligas activas.")
    return df


# ─── RESULTADOS DEL DÍA (football-data.org) ──────────────────────────────────

def get_results_fdorg(target_date: str) -> dict:
    """
    Obtiene resultados finalizados de football-data.org para una fecha.
    Usado como fuente principal (y fallback) en result_updater.
    Retorna dict: {(home_team, away_team): {home_goals, away_goals, ...}}
    """
    if not FOOTBALL_DATA_ORG_KEY:
        return {}

    headers = {"X-Auth-Token": FOOTBALL_DATA_ORG_KEY}
    results = {}

    for league_id, (league_name, _, fdorg_id) in LIGAS.items():
        try:
            resp = requests.get(
                f"{FOOTBALL_DATA_ORG_URL}/competitions/{fdorg_id}/matches",
                headers=headers,
                params={"dateFrom": target_date, "dateTo": target_date,
                        "status": "FINISHED"},
                timeout=10,
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
        except Exception as e:
            log.warning(f"Error resultados fd.org {league_name}: {e}")

        time.sleep(6)

    log.info(f"Resultados fd.org: {len(results)} partidos finalizados")
    return results


# ─── DATOS HISTÓRICOS + CUOTAS (Football-Data.co.uk) ─────────────────────────

def download_football_data() -> dict:
    """
    Descarga CSVs de Football-Data.co.uk para todas las ligas y temporadas.
    Incluye cuotas reales (B365, Pinnacle) para el value detector.
    """
    dfs = {}
    for league_id, (league_name, fd_code, _) in LIGAS.items():
        frames = []
        for season in FOOTBALL_DATA_SEASONS:
            url = f"{FOOTBALL_DATA_URL}/{season}/{fd_code}.csv"
            try:
                df = pd.read_csv(url, encoding="latin-1", on_bad_lines="skip")
                df["season"]      = season
                df["league_name"] = league_name
                frames.append(df)
                log.info(f"✓ {league_name} {season}: {len(df)} partidos")
            except Exception as e:
                log.warning(f"No disponible {league_name} {season}: {e}")

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            path = os.path.join(DATA_RAW, f"fd_{fd_code}.csv")
            combined.to_csv(path, index=False)
            dfs[fd_code] = combined
            log.info(f"→ {fd_code}: {len(combined)} partidos totales")

    return dfs


def get_best_closing_odds(home_team: str, away_team: str,
                          historical: pd.DataFrame) -> dict | None:
    """
    Busca las últimas cuotas de cierre reales para un enfrentamiento.
    Prioridad: Pinnacle > Bet365 > Betway.
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


# ─── STATSBOMB OPEN DATA ─────────────────────────────────────────────────────

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
                            shots["match_id"]    = match_id
                            shots["home_team"]   = match["home_team"]
                            shots["away_team"]   = match["away_team"]
                            shots["is_home_team"]= shots["team"] == match["home_team"]
                            shots["competition"] = comp["competition_name"]
                            shots["season"]      = comp["season_name"]
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
                            "corners_home": len(events[
                                (events["type"] == "Pass") &
                                (events.get("pass_type", "") == "Corner") &
                                (events["team"] == match["home_team"])
                            ]),
                            "corners_away": len(events[
                                (events["type"] == "Pass") &
                                (events.get("pass_type", "") == "Corner") &
                                (events["team"] == match["away_team"])
                            ]),
                            "fouls_home":   len(events[
                                (events["type"] == "Foul Committed") &
                                (events["team"] == match["home_team"])
                            ]),
                            "fouls_away":   len(events[
                                (events["type"] == "Foul Committed") &
                                (events["team"] == match["away_team"])
                            ]),
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
    for days_back in range(0, 4):
        target = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            df   = pd.read_csv(f"{CLUBELO_URL}/{target}")
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


# ─── RUNNER ──────────────────────────────────────────────────────────────────

def run():
    log.info("═══ FOOTBOT · Recolección de datos ═══")
    fixtures = get_fixtures_today()
    fd_data  = download_football_data()
    sb_data  = download_statsbomb_data()
    elo      = download_elo_ratings()
    log.info("═══ Recolección completada ═══")
    return {"fixtures": fixtures, "football_data": fd_data,
            "statsbomb": sb_data, "elo": elo}


if __name__ == "__main__":
    run()