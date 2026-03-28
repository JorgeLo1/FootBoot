"""
_01_data_collector.py
Descarga y normaliza todas las fuentes de datos:
  - Fixtures del día vía API-Football (con rate-limit tracker)
  - Resultados históricos + CUOTAS REALES de Football-Data.co.uk
  - Datos de eventos (xG, tiros, corners) de StatsBomb Open Data
  - Ratings ELO de ClubElo.com
  - Clima de Open-Meteo (sin API key)
"""

import os
import sys
import logging
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    API_FOOTBALL_KEY, API_USAGE_FILE, API_FOOTBALL_DAILY_LIMIT,
    LIGAS, DATA_RAW, DATA_STATSBOMB,
    FOOTBALL_DATA_URL, FOOTBALL_DATA_SEASONS,
    CLUBELO_URL, CURRENT_SEASON,
)
from src.utils import get_weather_for_fixture, ApiRateLimiter  # noqa: F401 — re-exportar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Instancia global del rate limiter (se comparte con result_updater)
rate_limiter = ApiRateLimiter(API_USAGE_FILE, API_FOOTBALL_DAILY_LIMIT)


# ─── FIXTURES DEL DÍA ────────────────────────────────────────────────────────

def get_fixtures_today() -> pd.DataFrame:
    today = date.today().strftime("%Y-%m-%d")
    headers = {"X-Auth-Token": FOOTBALL_DATA_ORG_KEY}
    
    # IDs de football-data.org (distintos a API-Football)
    COMPETITION_IDS = {
        2021: ("Premier League",  39),
        2014: ("La Liga",        140),
        2002: ("Bundesliga",      78),
        2019: ("Serie A",        135),
        2015: ("Ligue 1",         61),
        2003: ("Eredivisie",      88),
        2017: ("Primeira Liga",   94),
    }
    
    all_fixtures = []
    for comp_id, (league_name, league_id) in COMPETITION_IDS.items():
        try:
            resp = requests.get(
                f"https://api.football-data.org/v4/competitions/{comp_id}/matches",
                headers=headers,
                params={"dateFrom": today, "dateTo": today},
                timeout=10,
            )
            resp.raise_for_status()
            
            for match in resp.json().get("matches", []):
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
            time.sleep(6)  # respetar 10 req/min
            
        except Exception as e:
            log.warning(f"Error fixtures {league_name}: {e}")
    
    df = pd.DataFrame(all_fixtures)
    if not df.empty:
        path = os.path.join(DATA_RAW, f"fixtures_{today}.csv")
        df.to_csv(path, index=False)
        log.info(f"Fixtures guardados: {len(df)} partidos")
    return df


# ─── DATOS HISTÓRICOS + CUOTAS REALES ────────────────────────────────────────

# Columnas de cuotas que interesan (múltiples casas para tener la mejor odds)
ODDS_COLS = [
    # Bet365
    "B365H", "B365D", "B365A",
    # Betway / Blue Square
    "BWH",   "BWD",   "BWA",
    # Pinnacle (las más eficientes del mercado)
    "PSH",   "PSD",   "PSA",
    # Mercados de goles
    "B365>2.5", "B365<2.5",
    # BTTS (si existe en el CSV)
    "B365AHH",
]


def download_football_data() -> dict:
    """
    Descarga CSVs de Football-Data.co.uk para todas las ligas y temporadas.
    Incluye columnas de cuotas reales (B365, Pinnacle) para el value detector.
    """
    dfs = {}
    for league_id, (league_name, fd_code) in LIGAS.items():
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
            log.info(f"→ {fd_code}: {len(combined)} partidos totales guardados")

    return dfs


def get_best_closing_odds(home_team: str, away_team: str,
                          historical: pd.DataFrame) -> dict | None:
    """
    Busca las últimas cuotas de cierre reales para un enfrentamiento.
    Prioridad: Pinnacle > Bet365 > Betway.
    Retorna dict {home, draw, away, over25, under25} o None si no hay datos.
    """
    from src._02_feature_builder import normalize_team_name

    h = normalize_team_name(home_team)
    a = normalize_team_name(away_team)

    if "home_team_norm" not in historical.columns:
        return None

    mask = (
        (historical["home_team_norm"] == h) &
        (historical["away_team_norm"] == a)
    )
    recent = historical[mask].sort_values("match_date", ascending=False).head(3)

    for col_h, col_d, col_a in [
        ("PSH", "PSD", "PSA"),
        ("B365H", "B365D", "B365A"),
        ("BWH", "BWD", "BWA"),
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
                # Over/Under 2.5 si existen
                "over25":  float(recent["B365>2.5"].dropna().iloc[0])
                           if "B365>2.5" in recent.columns and not recent["B365>2.5"].dropna().empty
                           else None,
                "under25": float(recent["B365<2.5"].dropna().iloc[0])
                           if "B365<2.5" in recent.columns and not recent["B365<2.5"].dropna().empty
                           else None,
                "source":  col_h[:3],
            }
        except (ValueError, TypeError):
            continue

    return None


# ─── STATSBOMB OPEN DATA ─────────────────────────────────────────────────────

def download_statsbomb_data() -> dict:
    """
    Descarga datos de StatsBomb Open Data (gratis, sin API key).
    Importante: solo cubre temporadas específicas históricas.
    Los datos se guardan con timestamp para no mezclar con features actuales.
    """
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

                        shots = events[events["type"] == "Shot"].copy()
                        if not shots.empty:
                            shots["match_id"]    = match_id
                            shots["home_team"]   = match["home_team"]
                            shots["away_team"]   = match["away_team"]
                            shots["competition"] = comp["competition_name"]
                            shots["season"]      = comp["season_name"]
                            all_shots.append(
                                shots[["match_id", "team", "player",
                                       "shot_statsbomb_xg", "shot_outcome",
                                       "home_team", "away_team",
                                       "competition", "season"]]
                            )

                        all_summary.append({
                            "match_id":      match_id,
                            "home_team":     match["home_team"],
                            "away_team":     match["away_team"],
                            "home_score":    match["home_score"],
                            "away_score":    match["away_score"],
                            "competition":   comp["competition_name"],
                            "season":        comp["season_name"],
                            "match_date":    match["match_date"],
                            "corners_home":  len(events[
                                (events["type"] == "Pass") &
                                (events["pass_type"] == "Corner") &
                                (events["team"] == match["home_team"])
                            ]),
                            "corners_away":  len(events[
                                (events["type"] == "Pass") &
                                (events["pass_type"] == "Corner") &
                                (events["team"] == match["away_team"])
                            ]),
                            "fouls_home":    len(events[
                                (events["type"] == "Foul Committed") &
                                (events["team"] == match["home_team"])
                            ]),
                            "fouls_away":    len(events[
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
            df = pd.concat(all_shots, ignore_index=True)
            path = os.path.join(DATA_STATSBOMB, "shots_xg.csv")
            df.to_csv(path, index=False)
            result["shots"] = df
            log.info(f"StatsBomb shots: {len(df)} registros guardados")

        if all_summary:
            df = pd.DataFrame(all_summary)
            path = os.path.join(DATA_STATSBOMB, "match_summary.csv")
            df.to_csv(path, index=False)
            result["summary"] = df
            log.info(f"StatsBomb resumen: {len(df)} partidos guardados")

        return result

    except Exception as e:
        log.error(f"Error general StatsBomb: {e}")
        return {}


# ─── ELO RATINGS ─────────────────────────────────────────────────────────────

def download_elo_ratings() -> pd.DataFrame:
    """
    Descarga ratings ELO actuales de ClubElo.com (gratis, sin API key).
    Tiene fallback al día anterior si el server no responde.
    """
    for days_back in range(0, 4):  # intenta hoy, ayer, anteayer, etc.
        target = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            url = f"{CLUBELO_URL}/{target}"
            df  = pd.read_csv(url)
            path = os.path.join(DATA_RAW, "elo_ratings.csv")
            df.to_csv(path, index=False)
            if days_back == 0:
                log.info(f"ELO ratings actualizados: {len(df)} equipos")
            else:
                log.info(f"ELO ratings de hace {days_back}d: {len(df)} equipos")
            return df
        except Exception:
            continue

    # Último recurso: archivo en caché
    path = os.path.join(DATA_RAW, "elo_ratings.csv")
    if os.path.exists(path):
        log.warning("Usando ELO ratings del último archivo en caché.")
        return pd.read_csv(path)

    log.error("No se pudieron obtener ELO ratings.")
    return pd.DataFrame()


# ─── RUNNER ──────────────────────────────────────────────────────────────────

def run():
    log.info("═══ FOOTBOT · Recolección de datos ═══")
    fixtures    = get_fixtures_today()
    fd_data     = download_football_data()
    sb_data     = download_statsbomb_data()
    elo         = download_elo_ratings()
    log.info("═══ Recolección completada ═══")
    return {"fixtures": fixtures, "football_data": fd_data,
            "statsbomb": sb_data, "elo": elo}


if __name__ == "__main__":
    run()