"""
02_feature_builder.py
Construye todas las variables predictivas para cada partido:
  - xG ofensivo y defensivo (StatsBomb + Football-Data)
  - Forma reciente con decaimiento exponencial
  - Diferencia de ELO
  - Días de descanso y fatiga
  - Head-to-head histórico
  - Factores de clima
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_RAW, DATA_STATSBOMB, DATA_PROCESSED,
    VENTANA_FORMA, LAMBDA_DECAY, LIGAS
)
def get_weather_for_fixture(home_team: str, match_datetime: str) -> dict:
    """Obtiene clima para un partido (importado inline para evitar circular import)."""
    import requests as _req
    from config.settings import OPENMETEO_URL
    ESTADIOS = {
        "Arsenal": (51.5549, -0.1084), "Chelsea": (51.4816, -0.1910),
        "Manchester City": (53.4831, -2.2004), "Liverpool": (53.4308, -2.9608),
        "Real Madrid": (40.4530, -3.6883), "Barcelona": (41.3809, 2.1228),
        "Bayern Munich": (48.2188, 11.6248), "Borussia Dortmund": (51.4926, 7.4519),
        "Juventus": (45.1096, 7.6412), "AC Milan": (45.4781, 9.1240),
        "Paris SG": (48.8414, 2.2530),
    }
    coords = ESTADIOS.get(home_team, (51.5074, -0.1278))
    lat, lon = coords
    try:
        dt = datetime.fromisoformat(str(match_datetime).replace("Z", "+00:00"))
        ds = dt.strftime("%Y-%m-%d")
        r = _req.get(OPENMETEO_URL, params={
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "daily": "precipitation_sum,windspeed_10m_max,temperature_2m_max",
            "forecast_days": 3, "start_date": ds, "end_date": ds,
        }, timeout=8)
        d = r.json().get("daily", {})
        prec = (d.get("precipitation_sum") or [0])[0] or 0
        wind = (d.get("windspeed_10m_max") or [10])[0] or 10
        temp = (d.get("temperature_2m_max") or [15])[0] or 15
        return {"temp_max": temp, "precipitation": prec, "wind_max": wind,
                "rain_flag": int(prec > 2), "wind_flag": int(wind > 30)}
    except Exception:
        return {"temp_max": 15, "precipitation": 0, "wind_max": 10,
                "rain_flag": 0, "wind_flag": 0}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def exponential_weight(days_ago: float, lam: float = LAMBDA_DECAY) -> float:
    """Peso exponencial: partidos recientes importan más."""
    return np.exp(-lam * max(days_ago, 0))


def normalize_team_name(name: str) -> str:
    """Normaliza nombres de equipos para hacer match entre fuentes."""
    mappings = {
        "manchester united":  "man united",
        "manchester city":    "man city",
        "tottenham hotspur":  "tottenham",
        "wolverhampton":      "wolves",
        "newcastle united":   "newcastle",
        "brighton & hove albion": "brighton",
        "nottingham forest":  "nott'm forest",
        "paris saint-germain": "paris sg",
        "paris sg":            "paris sg",
        "atletico madrid":    "atletico madrid",
        "atletico de madrid": "atletico madrid",
        "fc barcelona":       "barcelona",
        "real madrid cf":     "real madrid",
        "borussia dortmund":  "dortmund",
        "bayer leverkusen":   "leverkusen",
        "rb leipzig":         "rb leipzig",
        "eintracht frankfurt":"frankfurt",
        "internazionale":     "inter",
        "inter milan":        "inter",
    }
    normalized = name.lower().strip()
    return mappings.get(normalized, normalized)


# ─── CARGA DE DATOS HISTÓRICOS ───────────────────────────────────────────────

def load_historical_results() -> pd.DataFrame:
    """
    Carga y unifica todos los CSVs de Football-Data.co.uk.
    Columnas clave: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, B365H, B365D, B365A
    """
    frames = []
    for _, (_, fd_code) in LIGAS.items():
        path = os.path.join(DATA_RAW, f"fd_{fd_code}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
            df["fd_code"] = fd_code
            frames.append(df)
    
    if not frames:
        log.warning("No hay datos de Football-Data descargados. Ejecuta primero 01.")
        return pd.DataFrame()
    
    raw = pd.concat(frames, ignore_index=True)
    
    # Normalizar columnas esenciales
    raw = raw.rename(columns={
        "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
        "Date": "date_str",
    })
    
    # Parsear fechas (Football-Data usa DD/MM/YY y DD/MM/YYYY)
    def parse_date(d):
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                return datetime.strptime(str(d), fmt)
            except Exception:
                continue
        return pd.NaT
    
    raw["match_date"] = raw["date_str"].apply(parse_date)
    raw = raw.dropna(subset=["match_date", "home_goals", "away_goals"])
    raw["home_goals"] = pd.to_numeric(raw["home_goals"], errors="coerce")
    raw["away_goals"] = pd.to_numeric(raw["away_goals"], errors="coerce")
    raw = raw.dropna(subset=["home_goals", "away_goals"])
    
    raw["home_team_norm"] = raw["home_team"].apply(normalize_team_name)
    raw["away_team_norm"] = raw["away_team"].apply(normalize_team_name)
    
    # Cuotas de referencia (Bet365)
    for col in ["B365H","B365D","B365A"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    
    log.info(f"Datos históricos cargados: {len(raw)} partidos")
    return raw.sort_values("match_date").reset_index(drop=True)


def load_xg_data() -> pd.DataFrame:
    """Carga datos de xG de StatsBomb."""
    path = os.path.join(DATA_STATSBOMB, "shots_xg.csv")
    if not os.path.exists(path):
        log.warning("No hay datos de StatsBomb. Se usará xG estimado de goles.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["team_norm"] = df["team"].apply(normalize_team_name)
    return df


def load_match_summary() -> pd.DataFrame:
    """Carga resumen de partidos StatsBomb (corners, faltas)."""
    path = os.path.join(DATA_STATSBOMB, "match_summary.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["home_team_norm"] = df["home_team"].apply(normalize_team_name)
    df["away_team_norm"] = df["away_team"].apply(normalize_team_name)
    return df


def load_elo() -> pd.DataFrame:
    """Carga ratings ELO de ClubElo."""
    path = os.path.join(DATA_RAW, "elo_ratings.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Club" in df.columns:
        df["team_norm"] = df["Club"].apply(normalize_team_name)
    return df


# ─── CÁLCULO DE FEATURES POR EQUIPO ──────────────────────────────────────────

def compute_team_stats(
    team: str,
    is_home: bool,
    historical: pd.DataFrame,
    xg_data: pd.DataFrame,
    match_summary: pd.DataFrame,
    reference_date: datetime,
    window: int = VENTANA_FORMA
) -> dict:
    """
    Calcula todas las estadísticas históricas de un equipo
    como local o visitante, con decaimiento temporal.
    """
    team_norm = normalize_team_name(team)
    
    if is_home:
        mask = historical["home_team_norm"] == team_norm
        team_goals_col  = "home_goals"
        rival_goals_col = "away_goals"
    else:
        mask = historical["away_team_norm"] == team_norm
        team_goals_col  = "away_goals"
        rival_goals_col = "home_goals"
    
    team_matches = historical[mask].copy()
    team_matches = team_matches[team_matches["match_date"] < reference_date]
    team_matches = team_matches.sort_values("match_date", ascending=False).head(window)
    
    if team_matches.empty:
        return _empty_team_stats(team_norm, is_home)
    
    # Calcular días desde cada partido
    team_matches = team_matches.copy()
    team_matches["days_ago"] = (
        reference_date - team_matches["match_date"]
    ).dt.days
    team_matches["weight"] = team_matches["days_ago"].apply(exponential_weight)
    
    total_weight = team_matches["weight"].sum()
    if total_weight == 0:
        total_weight = 1
    
    # Goles anotados y concedidos (ponderados)
    goals_scored   = (team_matches[team_goals_col]  * team_matches["weight"]).sum() / total_weight
    goals_conceded = (team_matches[rival_goals_col] * team_matches["weight"]).sum() / total_weight
    
    # xG desde StatsBomb si disponible, si no estimar desde goles
    xg_scored   = _compute_xg(team_norm, xg_data, reference_date, is_home, "attack")
    xg_conceded = _compute_xg(team_norm, xg_data, reference_date, is_home, "defense")
    if xg_scored == 0:
        xg_scored   = goals_scored * 1.05   # ligera estimación
    if xg_conceded == 0:
        xg_conceded = goals_conceded * 1.05
    
    # Puntos recientes (forma)
    def points(row):
        scored   = row[team_goals_col]
        conceded = row[rival_goals_col]
        if scored > conceded:   return 3
        elif scored == conceded: return 1
        else:                   return 0
    
    team_matches["points"] = team_matches.apply(points, axis=1)
    forma = (team_matches["points"] * team_matches["weight"]).sum() / total_weight
    
    # BTTS histórico (ambos marcan)
    team_matches["btts"] = (
        (team_matches["home_goals"] > 0) & (team_matches["away_goals"] > 0)
    ).astype(int)
    btts_rate = (team_matches["btts"] * team_matches["weight"]).sum() / total_weight
    
    # Over 2.5
    team_matches["over25"] = (
        (team_matches["home_goals"] + team_matches["away_goals"]) > 2.5
    ).astype(int)
    over25_rate = (team_matches["over25"] * team_matches["weight"]).sum() / total_weight
    
    # Corners y faltas desde StatsBomb
    corners_avg, fouls_avg = _compute_corners_fouls(
        team_norm, match_summary, reference_date, is_home
    )
    
    # Último partido (días de descanso)
    last_match_date = team_matches["match_date"].max()
    days_rest = (reference_date - last_match_date).days if not pd.isna(last_match_date) else 7
    
    prefix = "home" if is_home else "away"
    return {
        f"{prefix}_xg_scored":    round(xg_scored, 3),
        f"{prefix}_xg_conceded":  round(xg_conceded, 3),
        f"{prefix}_goals_scored":  round(goals_scored, 3),
        f"{prefix}_goals_conceded":round(goals_conceded, 3),
        f"{prefix}_forma":         round(forma, 3),
        f"{prefix}_btts_rate":     round(btts_rate, 3),
        f"{prefix}_over25_rate":   round(over25_rate, 3),
        f"{prefix}_corners_avg":   round(corners_avg, 2),
        f"{prefix}_fouls_avg":     round(fouls_avg, 2),
        f"{prefix}_days_rest":     days_rest,
        f"{prefix}_n_matches":     len(team_matches),
    }


def _empty_team_stats(team_norm: str, is_home: bool) -> dict:
    prefix = "home" if is_home else "away"
    return {
        f"{prefix}_xg_scored":    1.2,
        f"{prefix}_xg_conceded":  1.2,
        f"{prefix}_goals_scored":  1.2,
        f"{prefix}_goals_conceded":1.2,
        f"{prefix}_forma":         1.2,
        f"{prefix}_btts_rate":     0.5,
        f"{prefix}_over25_rate":   0.5,
        f"{prefix}_corners_avg":   5.0,
        f"{prefix}_fouls_avg":     11.0,
        f"{prefix}_days_rest":     7,
        f"{prefix}_n_matches":     0,
    }


def _compute_xg(team_norm, xg_data, ref_date, is_home, mode):
    """Calcula xG promedio de un equipo desde datos StatsBomb."""
    if xg_data.empty or "shot_statsbomb_xg" not in xg_data.columns:
        return 0.0
    mask = xg_data["team_norm"] == team_norm
    df = xg_data[mask].copy()
    if df.empty:
        return 0.0
    if mode == "attack":
        return df["shot_statsbomb_xg"].sum() / max(df["match_id"].nunique(), 1)
    else:
        # xG concedido: shots del rival en esos partidos
        home_matches = df["home_team"].unique() if is_home else []
        away_matches = df["away_team"].unique() if not is_home else []
        rival_mask = (
            xg_data["away_team"].isin(home_matches) |
            xg_data["home_team"].isin(away_matches)
        ) & (xg_data["team_norm"] != team_norm)
        rival = xg_data[rival_mask]
        if rival.empty:
            return 0.0
        return rival["shot_statsbomb_xg"].sum() / max(rival["match_id"].nunique(), 1)


def _compute_corners_fouls(team_norm, match_summary, ref_date, is_home):
    """Calcula promedios de corners y faltas desde StatsBomb summary."""
    if match_summary.empty:
        return 5.0, 11.0
    
    if is_home:
        mask = match_summary["home_team_norm"] == team_norm
        c_col, f_col = "corners_home", "fouls_home"
    else:
        mask = match_summary["away_team_norm"] == team_norm
        c_col, f_col = "corners_away", "fouls_away"
    
    df = match_summary[mask]
    if df.empty:
        return 5.0, 11.0
    
    return (
        df[c_col].mean() if c_col in df.columns else 5.0,
        df[f_col].mean() if f_col in df.columns else 11.0
    )


def compute_h2h(home_team: str, away_team: str,
                historical: pd.DataFrame, ref_date: datetime,
                last_n: int = 10) -> dict:
    """Calcula estadísticas head-to-head entre dos equipos."""
    h_norm = normalize_team_name(home_team)
    a_norm = normalize_team_name(away_team)
    
    mask = (
        ((historical["home_team_norm"] == h_norm) & (historical["away_team_norm"] == a_norm)) |
        ((historical["home_team_norm"] == a_norm) & (historical["away_team_norm"] == h_norm))
    )
    h2h = historical[mask & (historical["match_date"] < ref_date)]
    h2h = h2h.sort_values("match_date", ascending=False).head(last_n)
    
    if h2h.empty:
        return {"h2h_home_wins": 0.33, "h2h_draws": 0.33, "h2h_away_wins": 0.33,
                "h2h_avg_goals": 2.5, "h2h_btts_rate": 0.5, "h2h_n": 0}
    
    total = len(h2h)
    # Calcular desde perspectiva del home_team actual
    home_wins = len(h2h[
        ((h2h["home_team_norm"] == h_norm) & (h2h["home_goals"] > h2h["away_goals"])) |
        ((h2h["away_team_norm"] == h_norm) & (h2h["away_goals"] > h2h["home_goals"]))
    ])
    away_wins = len(h2h[
        ((h2h["home_team_norm"] == a_norm) & (h2h["home_goals"] > h2h["away_goals"])) |
        ((h2h["away_team_norm"] == a_norm) & (h2h["away_goals"] > h2h["home_goals"]))
    ])
    draws = total - home_wins - away_wins
    
    h2h["total_goals"] = h2h["home_goals"] + h2h["away_goals"]
    h2h["btts"] = ((h2h["home_goals"] > 0) & (h2h["away_goals"] > 0)).astype(int)
    
    return {
        "h2h_home_wins":  round(home_wins / total, 3),
        "h2h_draws":      round(draws / total, 3),
        "h2h_away_wins":  round(away_wins / total, 3),
        "h2h_avg_goals":  round(h2h["total_goals"].mean(), 3),
        "h2h_btts_rate":  round(h2h["btts"].mean(), 3),
        "h2h_n":          total,
    }


def get_elo_diff(home_team: str, away_team: str, elo_df: pd.DataFrame) -> float:
    """Retorna diferencia de ELO (home - away). Positivo = local favorito."""
    if elo_df.empty or "team_norm" not in elo_df.columns:
        return 0.0
    elo_col = "Elo" if "Elo" in elo_df.columns else (
              "elo" if "elo" in elo_df.columns else None)
    if not elo_col:
        return 0.0
    
    h_norm = normalize_team_name(home_team)
    a_norm = normalize_team_name(away_team)
    
    home_row = elo_df[elo_df["team_norm"] == h_norm]
    away_row = elo_df[elo_df["team_norm"] == a_norm]
    
    home_elo = home_row[elo_col].values[0] if not home_row.empty else 1500.0
    away_elo = away_row[elo_col].values[0] if not away_row.empty else 1500.0
    
    return round(float(home_elo) - float(away_elo), 1)


# ─── BUILDER PRINCIPAL ───────────────────────────────────────────────────────

def build_features_for_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Genera el DataFrame de features completo para todos los partidos del día.
    Este es el input directo del modelo.
    """
    log.info("Cargando datos históricos para feature building...")
    historical    = load_historical_results()
    xg_data       = load_xg_data()
    match_summary = load_match_summary()
    elo_df        = load_elo()
    
    if historical.empty:
        log.error("Sin datos históricos. Imposible calcular features.")
        return pd.DataFrame()
    
    all_features = []
    ref_date = datetime.now()
    
    for _, fixture in fixtures.iterrows():
        home = fixture["home_team"]
        away = fixture["away_team"]
        log.info(f"Calculando features: {home} vs {away}")
        
        # Stats de cada equipo
        home_stats = compute_team_stats(home, True,  historical, xg_data,
                                        match_summary, ref_date)
        away_stats = compute_team_stats(away, False, historical, xg_data,
                                        match_summary, ref_date)
        
        # H2H
        h2h = compute_h2h(home, away, historical, ref_date)
        
        # ELO
        elo_diff = get_elo_diff(home, away, elo_df)
        
        # Clima
        weather = get_weather_for_fixture(
            home, fixture.get("date", ref_date.isoformat())
        )
        
        # Cuotas del mercado como feature (si existen)
        market_features = _extract_market_features(
            home, away, historical, ref_date
        )
        
        # Conteo de partidos mínimos
        n_home = home_stats.get("home_n_matches", 0)
        n_away = away_stats.get("away_n_matches", 0)
        
        row = {
            "fixture_id":   fixture.get("fixture_id", 0),
            "league_id":    fixture.get("league_id", 0),
            "league_name":  fixture.get("league_name", ""),
            "home_team":    home,
            "away_team":    away,
            "match_date":   fixture.get("date", str(ref_date.date())),
            "n_home_matches": n_home,
            "n_away_matches": n_away,
            "elo_diff":     elo_diff,
            **home_stats,
            **away_stats,
            **h2h,
            **weather,
            **market_features,
        }
        all_features.append(row)
    
    df = pd.DataFrame(all_features)
    
    # Features derivados
    if not df.empty:
        df["xg_diff"]       = df["home_xg_scored"] - df["away_xg_scored"]
        df["xg_total_exp"]  = df["home_xg_scored"] + df["away_xg_scored"]
        df["goals_diff"]    = df["home_goals_scored"] - df["away_goals_scored"]
        df["forma_diff"]    = df["home_forma"] - df["away_forma"]
        df["rest_diff"]     = df["away_days_rest"] - df["home_days_rest"]
        df["fatiga_flag"]   = ((df["home_days_rest"] < 4) |
                               (df["away_days_rest"] < 4)).astype(int)
        
        path = os.path.join(DATA_PROCESSED, f"features_{date.today()}.csv")
        df.to_csv(path, index=False)
        log.info(f"Features guardados: {len(df)} partidos → {path}")
    
    return df


def _extract_market_features(home, away, historical, ref_date) -> dict:
    """Extrae probabilidades implícitas de las cuotas históricas del mercado."""
    h_norm = normalize_team_name(home)
    a_norm = normalize_team_name(away)
    
    mask = (
        ((historical["home_team_norm"] == h_norm) & (historical["away_team_norm"] == a_norm))
    )
    recent = historical[mask & (historical["match_date"] < ref_date)]
    recent = recent.sort_values("match_date", ascending=False).head(5)
    
    result = {"market_prob_home": 0.45, "market_prob_draw": 0.27, "market_prob_away": 0.28}
    
    for col_h, col_d, col_a in [("B365H","B365D","B365A"),("BWH","BWD","BWA"),("PSH","PSD","PSA")]:
        if all(c in recent.columns for c in [col_h, col_d, col_a]):
            valid = recent[[col_h, col_d, col_a]].dropna()
            if not valid.empty:
                row = valid.iloc[0]
                total = 1/row[col_h] + 1/row[col_d] + 1/row[col_a]
                result["market_prob_home"]  = round((1/row[col_h]) / total, 4)
                result["market_prob_draw"]  = round((1/row[col_d]) / total, 4)
                result["market_prob_away"]  = round((1/row[col_a]) / total, 4)
                break
    
    return result


def build_training_dataset(historical: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el dataset de entrenamiento completo.
    Para cada partido histórico, calcula los features con los datos
    disponibles ANTES de ese partido (walk-forward honesto).
    """
    log.info("Construyendo dataset de entrenamiento (walk-forward)...")
    
    xg_data       = load_xg_data()
    match_summary = load_match_summary()
    elo_df        = load_elo()
    
    training_rows = []
    
    # Ordenar por fecha y tomar de la mitad en adelante (para tener suficiente historia)
    historical = historical.sort_values("match_date").reset_index(drop=True)
    start_idx = max(200, len(historical) // 4)
    
    for idx in range(start_idx, len(historical)):
        match = historical.iloc[idx]
        ref_date = match["match_date"]
        past_data = historical.iloc[:idx]  # Solo datos anteriores
        
        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))
        if not home or not away:
            continue
        
        home_stats = compute_team_stats(home, True,  past_data, xg_data,
                                        match_summary, ref_date)
        away_stats = compute_team_stats(away, False, past_data, xg_data,
                                        match_summary, ref_date)
        h2h        = compute_h2h(home, away, past_data, ref_date)
        elo_diff   = get_elo_diff(home, away, elo_df)
        market     = _extract_market_features(home, away, past_data, ref_date)
        
        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])
        
        row = {
            "home_team":   home,
            "away_team":   away,
            "match_date":  ref_date,
            "elo_diff":    elo_diff,
            **home_stats,
            **away_stats,
            **h2h,
            **market,
            # Targets
            "target_home_win": int(home_goals > away_goals),
            "target_draw":     int(home_goals == away_goals),
            "target_away_win": int(home_goals < away_goals),
            "target_btts":     int(home_goals > 0 and away_goals > 0),
            "target_over25":   int(home_goals + away_goals > 2.5),
            "home_goals_actual": home_goals,
            "away_goals_actual": away_goals,
        }
        training_rows.append(row)
        
        if idx % 500 == 0:
            log.info(f"Procesados {idx}/{len(historical)} partidos...")
    
    df = pd.DataFrame(training_rows)
    
    if not df.empty:
        df["xg_diff"]      = df["home_xg_scored"] - df["away_xg_scored"]
        df["xg_total_exp"] = df["home_xg_scored"] + df["away_xg_scored"]
        df["goals_diff"]   = df["home_goals_scored"] - df["away_goals_scored"]
        df["forma_diff"]   = df["home_forma"] - df["away_forma"]
        df["rest_diff"]    = df["away_days_rest"] - df["home_days_rest"]
        df["fatiga_flag"]  = ((df["home_days_rest"] < 4) |
                              (df["away_days_rest"] < 4)).astype(int)
        
        path = os.path.join(DATA_PROCESSED, "training_dataset.csv")
        df.to_csv(path, index=False)
        log.info(f"Dataset entrenamiento: {len(df)} partidos → {path}")
    
    return df


if __name__ == "__main__":
    # Test: cargar fixtures del día y construir features
    today = date.today().strftime("%Y-%m-%d")
    fixtures_path = os.path.join(DATA_RAW, f"fixtures_{today}.csv")
    
    if os.path.exists(fixtures_path):
        fixtures = pd.read_csv(fixtures_path)
        df = build_features_for_fixtures(fixtures)
        print(df.head())
    else:
        print("No hay fixtures del día. Ejecuta primero 01_data_collector.py")
        
    # Construir dataset de entrenamiento
    historical = load_historical_results()
    if not historical.empty:
        training = build_training_dataset(historical)
        print(f"\nDataset de entrenamiento: {len(training)} registros")
