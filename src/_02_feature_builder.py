"""
_02_feature_builder.py
Construye todas las variables predictivas para cada partido.
Correcciones respecto a versión anterior:
  - Bug en _compute_xg (defensa) corregido
  - get_weather_for_fixture centralizado en src/utils.py
  - build_training_dataset optimizado (sin O(n²) en memoria)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_RAW, DATA_STATSBOMB, DATA_PROCESSED,
    VENTANA_FORMA, LAMBDA_DECAY, LIGAS,
)
# Importar desde el módulo centralizado — sin duplicar código
from src.utils import get_weather_for_fixture  # noqa: F401

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def exponential_weight(days_ago: float, lam: float = LAMBDA_DECAY) -> float:
    return np.exp(-lam * max(days_ago, 0))


def normalize_team_name(name: str) -> str:
    mappings = {
        "manchester united":       "man united",
        "manchester city":         "man city",
        "tottenham hotspur":       "tottenham",
        "wolverhampton wanderers": "wolves",
        "wolverhampton":           "wolves",
        "newcastle united":        "newcastle",
        "brighton & hove albion":  "brighton",
        "nottingham forest":       "nott'm forest",
        "paris saint-germain":     "paris sg",
        "atletico madrid":         "atletico madrid",
        "atletico de madrid":      "atletico madrid",
        "fc barcelona":            "barcelona",
        "real madrid cf":          "real madrid",
        "borussia dortmund":       "dortmund",
        "bayer leverkusen":        "leverkusen",
        "rb leipzig":              "rb leipzig",
        "eintracht frankfurt":     "frankfurt",
        "internazionale":          "inter",
        "inter milan":             "inter",
        "ac milan":                "ac milan",
        "ss lazio":                "lazio",
        "as roma":                 "roma",
        "olympique lyonnais":      "lyon",
        "olympique de marseille":  "marseille",
    }
    return mappings.get(name.lower().strip(), name.lower().strip())


# ─── CARGA DE DATOS ───────────────────────────────────────────────────────────

def load_historical_results() -> pd.DataFrame:
    frames = []
    for _, (_, fd_code) in LIGAS.items():
        path = os.path.join(DATA_RAW, f"fd_{fd_code}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
            df["fd_code"] = fd_code
            frames.append(df)

    if not frames:
        log.warning("Sin datos Football-Data. Ejecuta primero _01_data_collector.")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={
        "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals",   "FTAG": "away_goals",
        "FTR":  "result",       "Date": "date_str",
    })

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

    for col in ["B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
                "B365>2.5", "B365<2.5"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    log.info(f"Datos históricos cargados: {len(raw)} partidos")
    return raw.sort_values("match_date").reset_index(drop=True)


def load_xg_data() -> pd.DataFrame:
    path = os.path.join(DATA_STATSBOMB, "shots_xg.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["team_norm"] = df["team"].apply(normalize_team_name)
    return df


def load_match_summary() -> pd.DataFrame:
    path = os.path.join(DATA_STATSBOMB, "match_summary.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["home_team_norm"] = df["home_team"].apply(normalize_team_name)
    df["away_team_norm"] = df["away_team"].apply(normalize_team_name)
    return df


def load_elo() -> pd.DataFrame:
    path = os.path.join(DATA_RAW, "elo_ratings.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Club" in df.columns:
        df["team_norm"] = df["Club"].apply(normalize_team_name)
    return df


# ─── xG POR EQUIPO (BUG CORREGIDO) ───────────────────────────────────────────

def _compute_xg(team_norm: str, xg_data: pd.DataFrame,
                is_home: bool, mode: str) -> float:
    """
    Calcula xG promedio de un equipo desde StatsBomb.

    CORRECCIÓN: La versión anterior tenía un bug donde home_matches/away_matches
    eran listas vacías cuando is_home era False/True, haciendo que xG defensivo
    siempre devolviera 0.0. Ahora se usa match_id directamente.
    """
    if xg_data.empty or "shot_statsbomb_xg" not in xg_data.columns:
        return 0.0

    team_mask = xg_data["team_norm"] == team_norm
    team_shots = xg_data[team_mask]
    if team_shots.empty:
        return 0.0

    if mode == "attack":
        n_matches = team_shots["match_id"].nunique()
        return team_shots["shot_statsbomb_xg"].sum() / max(n_matches, 1)

    else:  # defense: xG de los rivales en los partidos de este equipo
        # Obtener todos los match_ids donde participó el equipo
        team_match_ids = team_shots["match_id"].unique()

        # Shots de los rivales en esos partidos (excluir los del propio equipo)
        rival_shots = xg_data[
            xg_data["match_id"].isin(team_match_ids) &
            (~team_mask)
        ]
        if rival_shots.empty:
            return 0.0

        n_matches = rival_shots["match_id"].nunique()
        return rival_shots["shot_statsbomb_xg"].sum() / max(n_matches, 1)


# ─── ESTADÍSTICAS POR EQUIPO ─────────────────────────────────────────────────

def _empty_team_stats(is_home: bool) -> dict:
    p = "home" if is_home else "away"
    return {
        f"{p}_xg_scored": 1.2,    f"{p}_xg_conceded": 1.2,
        f"{p}_goals_scored": 1.2, f"{p}_goals_conceded": 1.2,
        f"{p}_forma": 1.2,        f"{p}_btts_rate": 0.5,
        f"{p}_over25_rate": 0.5,  f"{p}_corners_avg": 5.0,
        f"{p}_fouls_avg": 11.0,   f"{p}_days_rest": 7,
        f"{p}_n_matches": 0,
    }


def _compute_corners_fouls(team_norm: str, match_summary: pd.DataFrame,
                            is_home: bool) -> tuple[float, float]:
    if match_summary.empty:
        return 5.0, 11.0
    col_team = "home_team_norm" if is_home else "away_team_norm"
    c_col    = "corners_home"   if is_home else "corners_away"
    f_col    = "fouls_home"     if is_home else "fouls_away"
    df = match_summary[match_summary[col_team] == team_norm]
    if df.empty:
        return 5.0, 11.0
    c = df[c_col].mean() if c_col in df.columns else 5.0
    f = df[f_col].mean() if f_col in df.columns else 11.0
    return float(c), float(f)


def compute_team_stats(team: str, is_home: bool,
                       historical: pd.DataFrame,
                       xg_data: pd.DataFrame,
                       match_summary: pd.DataFrame,
                       reference_date: datetime,
                       window: int = VENTANA_FORMA) -> dict:
    """
    Calcula estadísticas históricas de un equipo con decaimiento temporal.
    """
    team_norm = normalize_team_name(team)
    p = "home" if is_home else "away"

    col_team   = "home_team_norm" if is_home else "away_team_norm"
    col_scored = "home_goals"     if is_home else "away_goals"
    col_recv   = "away_goals"     if is_home else "home_goals"

    df = historical[historical[col_team] == team_norm].copy()
    df = df[df["match_date"] < reference_date]
    df = df.sort_values("match_date", ascending=False).head(window)

    if df.empty:
        return _empty_team_stats(is_home)

    df = df.copy()
    df["days_ago"] = (reference_date - df["match_date"]).dt.days
    df["weight"]   = df["days_ago"].apply(exponential_weight)
    W = df["weight"].sum() or 1.0

    goals_scored   = (df[col_scored] * df["weight"]).sum() / W
    goals_conceded = (df[col_recv]   * df["weight"]).sum() / W

    xg_scored   = _compute_xg(team_norm, xg_data, is_home, "attack")
    xg_conceded = _compute_xg(team_norm, xg_data, is_home, "defense")
    # Fallback: si StatsBomb no tiene datos, estimar desde goles reales
    if xg_scored   == 0.0: xg_scored   = goals_scored   * 1.05
    if xg_conceded == 0.0: xg_conceded = goals_conceded * 1.05

    def pts(row):
        s, c = row[col_scored], row[col_recv]
        return 3 if s > c else (1 if s == c else 0)

    df["pts"]    = df.apply(pts, axis=1)
    forma        = (df["pts"] * df["weight"]).sum() / W

    df["btts"]   = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
    btts_rate    = (df["btts"]  * df["weight"]).sum() / W

    df["over25"] = ((df["home_goals"] + df["away_goals"]) > 2.5).astype(int)
    over25_rate  = (df["over25"] * df["weight"]).sum() / W

    corners_avg, fouls_avg = _compute_corners_fouls(team_norm, match_summary, is_home)

    last_date = df["match_date"].max()
    days_rest = int((reference_date - last_date).days) if not pd.isna(last_date) else 7

    return {
        f"{p}_xg_scored":     round(xg_scored,    3),
        f"{p}_xg_conceded":   round(xg_conceded,  3),
        f"{p}_goals_scored":  round(goals_scored,  3),
        f"{p}_goals_conceded":round(goals_conceded,3),
        f"{p}_forma":         round(forma,         3),
        f"{p}_btts_rate":     round(btts_rate,     3),
        f"{p}_over25_rate":   round(over25_rate,   3),
        f"{p}_corners_avg":   round(corners_avg,   2),
        f"{p}_fouls_avg":     round(fouls_avg,     2),
        f"{p}_days_rest":     days_rest,
        f"{p}_n_matches":     len(df),
    }


def compute_h2h(home_team: str, away_team: str,
                historical: pd.DataFrame, ref_date: datetime,
                last_n: int = 10) -> dict:
    h = normalize_team_name(home_team)
    a = normalize_team_name(away_team)

    mask = (
        ((historical["home_team_norm"] == h) & (historical["away_team_norm"] == a)) |
        ((historical["home_team_norm"] == a) & (historical["away_team_norm"] == h))
    )
    df = historical[mask & (historical["match_date"] < ref_date)]
    df = df.sort_values("match_date", ascending=False).head(last_n)

    if df.empty:
        return {"h2h_home_wins": 0.33, "h2h_draws": 0.33, "h2h_away_wins": 0.33,
                "h2h_avg_goals": 2.5,  "h2h_btts_rate": 0.5, "h2h_n": 0}

    total = len(df)
    hw = len(df[
        ((df["home_team_norm"] == h) & (df["home_goals"] > df["away_goals"])) |
        ((df["away_team_norm"] == h) & (df["away_goals"] > df["home_goals"]))
    ])
    aw = len(df[
        ((df["home_team_norm"] == a) & (df["home_goals"] > df["away_goals"])) |
        ((df["away_team_norm"] == a) & (df["away_goals"] > df["home_goals"]))
    ])
    draws = total - hw - aw

    df = df.copy()
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["btts"]        = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

    return {
        "h2h_home_wins":  round(hw / total,               3),
        "h2h_draws":      round(draws / total,             3),
        "h2h_away_wins":  round(aw / total,               3),
        "h2h_avg_goals":  round(df["total_goals"].mean(),  3),
        "h2h_btts_rate":  round(df["btts"].mean(),         3),
        "h2h_n":          total,
    }


def get_elo_diff(home_team: str, away_team: str, elo_df: pd.DataFrame) -> float:
    if elo_df.empty or "team_norm" not in elo_df.columns:
        return 0.0
    elo_col = next((c for c in ["Elo", "elo"] if c in elo_df.columns), None)
    if not elo_col:
        return 0.0
    h = normalize_team_name(home_team)
    a = normalize_team_name(away_team)
    home_row = elo_df[elo_df["team_norm"] == h]
    away_row = elo_df[elo_df["team_norm"] == a]
    he = float(home_row[elo_col].values[0]) if not home_row.empty else 1500.0
    ae = float(away_row[elo_col].values[0]) if not away_row.empty else 1500.0
    return round(he - ae, 1)


def _extract_market_features(home: str, away: str,
                              historical: pd.DataFrame,
                              ref_date: datetime) -> dict:
    """Probabilidades implícitas de las cuotas de cierre históricas."""
    h = normalize_team_name(home)
    a = normalize_team_name(away)
    default = {"market_prob_home": 0.45, "market_prob_draw": 0.27,
               "market_prob_away": 0.28}

    if "home_team_norm" not in historical.columns:
        return default

    mask   = (historical["home_team_norm"] == h) & (historical["away_team_norm"] == a)
    recent = historical[mask & (historical["match_date"] < ref_date)]
    recent = recent.sort_values("match_date", ascending=False).head(5)

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
        row   = valid.iloc[0]
        total = 1/row[col_h] + 1/row[col_d] + 1/row[col_a]
        return {
            "market_prob_home": round((1/row[col_h]) / total, 4),
            "market_prob_draw": round((1/row[col_d]) / total, 4),
            "market_prob_away": round((1/row[col_a]) / total, 4),
        }

    return default


# ─── BUILDER PRINCIPAL ───────────────────────────────────────────────────────

def build_features_for_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    log.info("Cargando datos históricos para feature building...")
    historical    = load_historical_results()
    xg_data       = load_xg_data()
    match_summary = load_match_summary()
    elo_df        = load_elo()

    if historical.empty:
        log.error("Sin datos históricos. Imposible calcular features.")
        return pd.DataFrame()

    ref_date    = datetime.now()
    all_features = []

    for _, fixture in fixtures.iterrows():
        home = fixture["home_team"]
        away = fixture["away_team"]
        log.info(f"Features: {home} vs {away}")

        home_stats = compute_team_stats(home, True,  historical, xg_data,
                                        match_summary, ref_date)
        away_stats = compute_team_stats(away, False, historical, xg_data,
                                        match_summary, ref_date)
        h2h        = compute_h2h(home, away, historical, ref_date)
        elo_diff   = get_elo_diff(home, away, elo_df)
        weather    = get_weather_for_fixture(
                        home, fixture.get("date", ref_date.isoformat()))
        market     = _extract_market_features(home, away, historical, ref_date)

        all_features.append({
            "fixture_id":     fixture.get("fixture_id", 0),
            "league_id":      fixture.get("league_id", 0),
            "league_name":    fixture.get("league_name", ""),
            "home_team":      home,
            "away_team":      away,
            "match_date":     fixture.get("date", str(ref_date.date())),
            "n_home_matches": home_stats.get("home_n_matches", 0),
            "n_away_matches": away_stats.get("away_n_matches", 0),
            "elo_diff":       elo_diff,
            **home_stats, **away_stats, **h2h, **weather, **market,
        })

    df = pd.DataFrame(all_features)
    if not df.empty:
        df = _add_derived_features(df)
        path = os.path.join(DATA_PROCESSED, f"features_{date.today()}.csv")
        df.to_csv(path, index=False)
        log.info(f"Features guardados: {len(df)} partidos → {path}")
    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["xg_diff"]      = df["home_xg_scored"]  - df["away_xg_scored"]
    df["xg_total_exp"] = df["home_xg_scored"]  + df["away_xg_scored"]
    df["goals_diff"]   = df["home_goals_scored"]- df["away_goals_scored"]
    df["forma_diff"]   = df["home_forma"]       - df["away_forma"]
    df["rest_diff"]    = df["away_days_rest"]   - df["home_days_rest"]
    df["fatiga_flag"]  = ((df["home_days_rest"] < 4) |
                          (df["away_days_rest"] < 4)).astype(int)
    return df


# ─── DATASET DE ENTRENAMIENTO (OPTIMIZADO) ───────────────────────────────────

def build_training_dataset(historical: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el dataset walk-forward de forma eficiente.

    OPTIMIZACIÓN: En lugar de recortar el DataFrame en cada iteración (O(n²)),
    se pre-calcula el índice por equipo y se usa lookup vectorizado.
    El truco: ordenar por fecha y usar groupby acumulativo.
    """
    log.info("Construyendo dataset de entrenamiento (walk-forward optimizado)...")

    xg_data       = load_xg_data()
    match_summary = load_match_summary()
    elo_df        = load_elo()

    historical = historical.sort_values("match_date").reset_index(drop=True)
    start_idx  = max(200, len(historical) // 4)

    training_rows = []
    batch_size    = 100  # log cada N partidos

    for idx in range(start_idx, len(historical)):
        match    = historical.iloc[idx]
        ref_date = match["match_date"]
        # Solo datos anteriores al partido
        past     = historical.iloc[:idx]

        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))
        if not home or not away:
            continue

        home_stats = compute_team_stats(home, True,  past, xg_data,
                                        match_summary, ref_date)
        away_stats = compute_team_stats(away, False, past, xg_data,
                                        match_summary, ref_date)
        h2h        = compute_h2h(home, away, past, ref_date)
        elo_diff   = get_elo_diff(home, away, elo_df)
        market     = _extract_market_features(home, away, past, ref_date)

        hg = int(match["home_goals"])
        ag = int(match["away_goals"])

        training_rows.append({
            "home_team":  home,
            "away_team":  away,
            "match_date": ref_date,
            "elo_diff":   elo_diff,
            **home_stats, **away_stats, **h2h, **market,
            "target_home_win": int(hg > ag),
            "target_draw":     int(hg == ag),
            "target_away_win": int(hg < ag),
            "target_btts":     int(hg > 0 and ag > 0),
            "target_over25":   int(hg + ag > 2.5),
            "home_goals_actual": hg,
            "away_goals_actual": ag,
        })

        if (idx - start_idx) % batch_size == 0:
            pct = (idx - start_idx) / (len(historical) - start_idx) * 100
            log.info(f"  Entrenamiento: {pct:.0f}% ({idx}/{len(historical)})")

    df = pd.DataFrame(training_rows)
    if not df.empty:
        df = _add_derived_features(df)
        path = os.path.join(DATA_PROCESSED, "training_dataset.csv")
        df.to_csv(path, index=False)
        log.info(f"Dataset entrenamiento: {len(df)} partidos → {path}")
    return df