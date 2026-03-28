"""
01_data_collector.py
Descarga y normaliza todas las fuentes de datos:
  - Fixtures del día vía API-Football
  - Resultados históricos de Football-Data.co.uk
  - Datos de eventos (xG, tiros, corners) de StatsBomb Open Data
  - Ratings ELO de ClubElo.com
  - Clima de Open-Meteo
"""

import os
import sys
import json
import logging
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    API_FOOTBALL_KEY, LIGAS, DATA_RAW, DATA_STATSBOMB,
    FOOTBALL_DATA_URL, FOOTBALL_DATA_SEASONS,
    OPENMETEO_URL, CLUBELO_URL
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─── ESTADIOS CON COORDENADAS (para clima) ───────────────────────────────────
ESTADIOS = {
    "Arsenal":          (51.5549, -0.1084),
    "Chelsea":          (51.4816, -0.1910),
    "Manchester City":  (53.4831, -2.2004),
    "Manchester United":(53.4631, -2.2913),
    "Liverpool":        (53.4308, -2.9608),
    "Tottenham":        (51.6043, -0.0665),
    "Real Madrid":      (40.4530, -3.6883),
    "Barcelona":        (41.3809,  2.1228),
    "Bayern Munich":    (48.2188, 11.6248),
    "Borussia Dortmund":(51.4926,  7.4519),
    "Juventus":         (45.1096,  7.6412),
    "AC Milan":         (45.4781,  9.1240),
    "Paris SG":         (48.8414,  2.2530),
}
DEFAULT_COORDS = (51.5074, -0.1278)  # Londres como fallback


def get_fixtures_today() -> pd.DataFrame:
    """Obtiene los partidos del día desde API-Football."""
    today = date.today().strftime("%Y-%m-%d")
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key": API_FOOTBALL_KEY
    }
    
    all_fixtures = []
    for league_id, (league_name, _) in LIGAS.items():
        try:
            resp = requests.get(
                url,
                headers=headers,
                params={"date": today, "league": league_id, "season": 2024},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            
            for f in data.get("response", []):
                fixture = f["fixture"]
                teams   = f["teams"]
                all_fixtures.append({
                    "fixture_id":      fixture["id"],
                    "date":            fixture["date"],
                    "league_id":       league_id,
                    "league_name":     league_name,
                    "home_team":       teams["home"]["name"],
                    "away_team":       teams["away"]["name"],
                    "venue":           fixture.get("venue", {}).get("name", ""),
                    "venue_city":      fixture.get("venue", {}).get("city", ""),
                    "status":          fixture["status"]["short"],
                })
        except Exception as e:
            log.warning(f"Error obteniendo fixtures {league_name}: {e}")
    
    df = pd.DataFrame(all_fixtures)
    if not df.empty:
        path = os.path.join(DATA_RAW, f"fixtures_{today}.csv")
        df.to_csv(path, index=False)
        log.info(f"Fixtures guardados: {len(df)} partidos → {path}")
    else:
        log.info("No hay partidos hoy en las ligas activas.")
    return df


def download_football_data() -> dict:
    """
    Descarga CSVs de Football-Data.co.uk para todas las ligas y temporadas.
    Retorna dict {liga_code: DataFrame} con datos históricos.
    """
    dfs = {}
    for league_id, (league_name, fd_code) in LIGAS.items():
        frames = []
        for season in FOOTBALL_DATA_SEASONS:
            url = f"{FOOTBALL_DATA_URL}/{season}/{fd_code}.csv"
            try:
                df = pd.read_csv(url, encoding="latin-1", on_bad_lines="skip")
                df["season"] = season
                df["league_name"] = league_name
                frames.append(df)
                log.info(f"Descargado {league_name} {season}: {len(df)} partidos")
            except Exception as e:
                log.warning(f"No disponible {league_name} {season}: {e}")
        
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            path = os.path.join(DATA_RAW, f"fd_{fd_code}.csv")
            combined.to_csv(path, index=False)
            dfs[fd_code] = combined
    
    return dfs


def download_statsbomb_data() -> dict:
    """
    Descarga datos de StatsBomb Open Data desde GitHub.
    Obtiene xG, tiros, corners y presión por equipo.
    """
    try:
        from statsbombpy import sb
        
        competitions = sb.competitions()
        # Filtrar competiciones relevantes disponibles en open data
        target = competitions[
            competitions["competition_name"].isin([
                "Premier League", "La Liga", "Bundesliga",
                "Serie A", "Ligue 1", "Champions League"
            ])
        ]
        
        all_shots = []
        all_events_summary = []
        
        for _, comp in target.iterrows():
            comp_id   = comp["competition_id"]
            season_id = comp["season_id"]
            
            try:
                matches = sb.matches(competition_id=comp_id, season_id=season_id)
                log.info(f"StatsBomb: {comp['competition_name']} {comp['season_name']} → {len(matches)} partidos")
                
                for _, match in matches.iterrows():
                    match_id = match["match_id"]
                    try:
                        events = sb.events(match_id=match_id)
                        
                        # Extraer tiros con xG
                        shots = events[events["type"] == "Shot"].copy()
                        if not shots.empty:
                            shots["match_id"]    = match_id
                            shots["home_team"]   = match["home_team"]
                            shots["away_team"]   = match["away_team"]
                            shots["competition"] = comp["competition_name"]
                            shots["season"]      = comp["season_name"]
                            all_shots.append(shots[["match_id","team","player",
                                                      "shot_statsbomb_xg","shot_outcome",
                                                      "home_team","away_team",
                                                      "competition","season"]])
                        
                        # Resumen por partido: corners, faltas, presión
                        summary = {
                            "match_id":       match_id,
                            "home_team":      match["home_team"],
                            "away_team":      match["away_team"],
                            "home_score":     match["home_score"],
                            "away_score":     match["away_score"],
                            "competition":    comp["competition_name"],
                            "season":         comp["season_name"],
                            "match_date":     match["match_date"],
                            "corners_home":   len(events[(events["type"]=="Pass") &
                                                         (events["pass_type"]=="Corner") &
                                                         (events["team"]==match["home_team"])]),
                            "corners_away":   len(events[(events["type"]=="Pass") &
                                                         (events["pass_type"]=="Corner") &
                                                         (events["team"]==match["away_team"])]),
                            "fouls_home":     len(events[(events["type"]=="Foul Committed") &
                                                         (events["team"]==match["home_team"])]),
                            "fouls_away":     len(events[(events["type"]=="Foul Committed") &
                                                         (events["team"]==match["away_team"])]),
                            "pressure_home":  len(events[(events["type"]=="Pressure") &
                                                         (events["team"]==match["home_team"])]),
                            "pressure_away":  len(events[(events["type"]=="Pressure") &
                                                         (events["team"]==match["away_team"])]),
                        }
                        all_events_summary.append(summary)
                        
                    except Exception:
                        continue
                        
            except Exception as e:
                log.warning(f"Error StatsBomb {comp['competition_name']}: {e}")
                continue
        
        result = {}
        if all_shots:
            df_shots = pd.concat(all_shots, ignore_index=True)
            path = os.path.join(DATA_STATSBOMB, "shots_xg.csv")
            df_shots.to_csv(path, index=False)
            result["shots"] = df_shots
            log.info(f"StatsBomb shots guardados: {len(df_shots)} registros")
        
        if all_events_summary:
            df_summary = pd.DataFrame(all_events_summary)
            path = os.path.join(DATA_STATSBOMB, "match_summary.csv")
            df_summary.to_csv(path, index=False)
            result["summary"] = df_summary
            log.info(f"StatsBomb resumen guardado: {len(df_summary)} partidos")
        
        return result
        
    except Exception as e:
        log.error(f"Error general StatsBomb: {e}")
        return {}


def download_elo_ratings() -> pd.DataFrame:
    """Descarga ratings ELO actuales de ClubElo.com."""
    try:
        today = date.today().strftime("%Y-%m-%d")
        url = f"{CLUBELO_URL}/{today}"
        df = pd.read_csv(url)
        path = os.path.join(DATA_RAW, "elo_ratings.csv")
        df.to_csv(path, index=False)
        log.info(f"ELO ratings descargados: {len(df)} equipos")
        return df
    except Exception as e:
        log.warning(f"Error descargando ELO: {e}")
        # Intentar con el CSV de la semana anterior
        try:
            yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
            df = pd.read_csv(f"{CLUBELO_URL}/{yesterday}")
            return df
        except Exception:
            path = os.path.join(DATA_RAW, "elo_ratings.csv")
            if os.path.exists(path):
                log.info("Usando ELO ratings del último archivo disponible")
                return pd.read_csv(path)
            return pd.DataFrame()


def get_weather_for_fixture(home_team: str, match_datetime: str) -> dict:
    """
    Obtiene pronóstico del clima para un partido usando Open-Meteo.
    Retorna dict con temperatura, lluvia y viento.
    """
    coords = ESTADIOS.get(home_team, DEFAULT_COORDS)
    lat, lon = coords
    
    try:
        match_dt = datetime.fromisoformat(match_datetime.replace("Z", "+00:00"))
        match_date_str = match_dt.strftime("%Y-%m-%d")
        
        resp = requests.get(
            OPENMETEO_URL,
            params={
                "latitude":            lat,
                "longitude":           lon,
                "daily":               "precipitation_sum,windspeed_10m_max,temperature_2m_max",
                "forecast_days":       3,
                "start_date":          match_date_str,
                "end_date":            match_date_str,
                "timezone":            "auto"
            },
            timeout=8
        )
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        
        return {
            "temp_max":     daily.get("temperature_2m_max", [15])[0] or 15,
            "precipitation":daily.get("precipitation_sum",  [0])[0]  or 0,
            "wind_max":     daily.get("windspeed_10m_max",  [10])[0] or 10,
            "rain_flag":    1 if (daily.get("precipitation_sum", [0])[0] or 0) > 2 else 0,
            "wind_flag":    1 if (daily.get("windspeed_10m_max", [10])[0] or 10) > 30 else 0,
        }
    except Exception as e:
        log.warning(f"Error clima para {home_team}: {e}")
        return {"temp_max": 15, "precipitation": 0, "wind_max": 10,
                "rain_flag": 0, "wind_flag": 0}


def run():
    """Ejecuta la recolección completa de datos."""
    log.info("═══ FOOTBOT · Recolección de datos ═══")
    
    # 1. Fixtures del día
    log.info("── 1/4 Descargando fixtures del día...")
    fixtures = get_fixtures_today()
    
    # 2. Datos históricos Football-Data
    log.info("── 2/4 Descargando Football-Data.co.uk...")
    fd_data = download_football_data()
    
    # 3. StatsBomb Open Data
    log.info("── 3/4 Descargando StatsBomb Open Data...")
    sb_data = download_statsbomb_data()
    
    # 4. ELO ratings
    log.info("── 4/4 Descargando ELO ratings...")
    elo = download_elo_ratings()
    
    log.info("═══ Recolección completada ═══")
    return {
        "fixtures": fixtures,
        "football_data": fd_data,
        "statsbomb": sb_data,
        "elo": elo
    }


if __name__ == "__main__":
    run()
