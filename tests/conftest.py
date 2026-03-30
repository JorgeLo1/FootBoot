"""
conftest.py — Fixtures compartidos para todos los tests de FOOTBOT.

Uso:
    pytest tests/                      # todos los tests rápidos
    pytest tests/ -m slow              # incluye entrenamiento de modelos
    pytest tests/ --cov=src            # con cobertura
"""

import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Añade la raíz del proyecto al path para que "from src.xxx import" funcione
# independientemente de desde dónde se ejecute pytest.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, date, timedelta


# ─── FIXTURE: historical (EU style, con cuotas) ───────────────────────────────

@pytest.fixture
def sample_historical():
    """
    DataFrame histórico de 240 partidos con el schema de Football-Data.co.uk.
    Suficiente para DixonColesModel (MIN_MATCHES_PER_LIGA = 200).
    Incluye cuotas B365 y Pinnacle para tests de build_odds_dict.
    """
    np.random.seed(42)
    n = 240
    teams = [
        "Arsenal", "Chelsea", "Manchester City", "Liverpool",
        "Tottenham", "Manchester United", "Aston Villa", "Newcastle",
    ]

    records = []
    base = datetime(2022, 8, 1)
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 3) % len(teams)]
        if home == away:
            away = teams[(i + 4) % len(teams)]
        hg = int(np.random.poisson(1.5))
        ag = int(np.random.poisson(1.1))
        records.append({
            "home_team":      home,
            "away_team":      away,
            "home_goals":     hg,
            "away_goals":     ag,
            "match_date":     base + timedelta(days=i * 3),
            "league_name":    "Premier League",
            "league_id":      39,
            "source":         "fd_uk",
            "B365H":          round(np.random.uniform(1.5, 4.0), 2),
            "B365D":          round(np.random.uniform(2.8, 4.0), 2),
            "B365A":          round(np.random.uniform(1.5, 5.0), 2),
            "PSH":            round(np.random.uniform(1.5, 4.0), 2),
            "PSD":            round(np.random.uniform(2.8, 4.0), 2),
            "PSA":            round(np.random.uniform(1.5, 5.0), 2),
            "B365>2.5":       round(np.random.uniform(1.70, 2.20), 2),
            "B365<2.5":       round(np.random.uniform(1.70, 2.20), 2),
        })

    df = pd.DataFrame(records)
    df["home_team_norm"] = df["home_team"].str.lower().str.strip()
    df["away_team_norm"] = df["away_team"].str.lower().str.strip()
    return df


# ─── FIXTURE: training_df (con targets y features para XGBoost) ───────────────

@pytest.fixture
def sample_training_df():
    """
    Dataset de entrenamiento mínimo con todas las columnas que necesita
    FootbotEnsemble.fit(): FEATURE_COLS + targets + home/away_team_norm.
    """
    np.random.seed(0)
    n = 300
    teams = [f"Club_{i:02d}" for i in range(10)]

    rows = []
    base = datetime(2021, 1, 1)
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 4) % len(teams)]
        hg   = int(np.random.poisson(1.4))
        ag   = int(np.random.poisson(1.1))
        rows.append({
            # Identidad
            "home_team":           home,
            "away_team":           away,
            "home_team_norm":      home.lower(),
            "away_team_norm":      away.lower(),
            "home_goals":          hg,
            "away_goals":          ag,
            "league_name":         "Test League",
            "match_date":          base + timedelta(days=i * 3),
            # FEATURE_COLS
            "home_xg_scored":      np.random.uniform(0.8, 2.5),
            "home_xg_conceded":    np.random.uniform(0.8, 1.8),
            "home_goals_scored":   np.random.uniform(0.8, 2.5),
            "home_goals_conceded": np.random.uniform(0.6, 1.8),
            "home_forma":          np.random.uniform(0.5, 2.5),
            "home_btts_rate":      np.random.uniform(0.3, 0.7),
            "home_over25_rate":    np.random.uniform(0.3, 0.7),
            "home_corners_avg":    np.random.uniform(3.0, 8.0),
            "home_fouls_avg":      np.random.uniform(8.0, 15.0),
            "home_days_rest":      float(np.random.randint(3, 14)),
            "away_xg_scored":      np.random.uniform(0.8, 2.0),
            "away_xg_conceded":    np.random.uniform(0.8, 1.8),
            "away_goals_scored":   np.random.uniform(0.8, 2.0),
            "away_goals_conceded": np.random.uniform(0.8, 1.8),
            "away_forma":          np.random.uniform(0.5, 2.5),
            "away_btts_rate":      np.random.uniform(0.3, 0.7),
            "away_over25_rate":    np.random.uniform(0.3, 0.7),
            "away_corners_avg":    np.random.uniform(3.0, 8.0),
            "away_fouls_avg":      np.random.uniform(8.0, 15.0),
            "away_days_rest":      float(np.random.randint(3, 14)),
            "h2h_home_wins":       np.random.uniform(0.2, 0.5),
            "h2h_draws":           np.random.uniform(0.1, 0.4),
            "h2h_away_wins":       np.random.uniform(0.2, 0.5),
            "h2h_avg_goals":       np.random.uniform(2.0, 3.5),
            "h2h_btts_rate":       np.random.uniform(0.3, 0.7),
            "elo_diff":            np.random.uniform(-200.0, 200.0),
            "xg_diff":             np.random.uniform(-1.0, 1.0),
            "xg_total_exp":        np.random.uniform(1.5, 4.0),
            "goals_diff":          np.random.uniform(-1.0, 1.0),
            "forma_diff":          np.random.uniform(-1.0, 1.0),
            "rest_diff":           float(np.random.randint(-5, 5)),
            "fatiga_flag":         float(np.random.randint(0, 2)),
            "rain_flag":           float(np.random.randint(0, 2)),
            "wind_flag":           float(np.random.randint(0, 2)),
            # Targets
            "target_home_win":     int(hg > ag),
            "target_draw":         int(hg == ag),
            "target_away_win":     int(hg < ag),
            "target_btts":         int(hg > 0 and ag > 0),
            "target_over25":       int(hg + ag > 2.5),
            "home_goals_actual":   hg,
            "away_goals_actual":   ag,
        })

    return pd.DataFrame(rows)


# ─── FIXTURE: fixture_row (features calculadas para un partido) ───────────────

@pytest.fixture
def sample_fixture_row():
    """Fila de features lista para pasar a analyze_fixture / predict_match."""
    return pd.Series({
        "fixture_id":           12345,
        "league_id":            39,
        "league_name":          "Premier League",
        "home_team":            "Arsenal",
        "away_team":            "Chelsea",
        "match_date":           str(date.today()),
        "source":               "fdorg",
        "n_home_matches":       50,
        "n_away_matches":       50,
        # Home stats
        "home_xg_scored":       1.80,
        "home_xg_conceded":     1.00,
        "home_goals_scored":    1.90,
        "home_goals_conceded":  0.90,
        "home_forma":           2.10,
        "home_btts_rate":       0.65,
        "home_over25_rate":     0.60,
        "home_corners_avg":     6.20,
        "home_fouls_avg":       10.50,
        "home_days_rest":       7,
        "home_n_matches":       10,
        "home_n_matches_total": 50,
        # Away stats
        "away_xg_scored":       1.40,
        "away_xg_conceded":     1.30,
        "away_goals_scored":    1.50,
        "away_goals_conceded":  1.20,
        "away_forma":           1.80,
        "away_btts_rate":       0.55,
        "away_over25_rate":     0.55,
        "away_corners_avg":     5.50,
        "away_fouls_avg":       11.20,
        "away_days_rest":       7,
        "away_n_matches":       10,
        "away_n_matches_total": 50,
        # H2H
        "h2h_home_wins":        0.40,
        "h2h_draws":            0.30,
        "h2h_away_wins":        0.30,
        "h2h_avg_goals":        2.80,
        "h2h_btts_rate":        0.60,
        "h2h_n":                10,
        # Derivadas
        "elo_diff":             50.0,
        "xg_diff":              0.40,
        "xg_total_exp":         3.20,
        "goals_diff":           0.40,
        "forma_diff":           0.30,
        "rest_diff":            0,
        "fatiga_flag":          0,
        "rain_flag":            0,
        "wind_flag":            0,
        # Market probs (solo inferencia)
        "market_prob_home":     0.45,
        "market_prob_draw":     0.28,
        "market_prob_away":     0.27,
        # ESPN odds (no disponibles)
        "espn_odds_available":  False,
    })


# ─── FIXTURE: predictions (salida de predict_match) ──────────────────────────

@pytest.fixture
def sample_predictions():
    """Dict de predicciones blended DC+XGBoost para un partido."""
    return {
        "prob_home_win":        0.48,
        "prob_draw":            0.27,
        "prob_away_win":        0.25,
        "prob_btts":            0.58,
        "prob_over25":          0.54,
        "prob_under25":         0.46,
        "dc_prob_home":         0.47,
        "dc_prob_draw":         0.28,
        "dc_prob_away":         0.25,
        "dc_prob_btts":         0.56,
        "dc_prob_over25":       0.52,
        "dc_exp_home_goals":    1.60,
        "dc_exp_away_goals":    1.20,
        "dc_exp_total_goals":   2.80,
        "top_features": {
            "home_win": ["home_forma", "elo_diff", "home_xg_scored"],
            "btts":     ["h2h_btts_rate", "home_over25_rate"],
            "over25":   ["xg_total_exp", "h2h_avg_goals"],
        },
    }


# ─── FIXTURE: bets_df (value bets de ejemplo para telegram) ──────────────────

@pytest.fixture
def sample_bets_df():
    """DataFrame con una apuesta de cada nivel de confianza."""
    today = str(date.today())
    rows  = []

    configs = [
        ("alta",  "home_win",  "Victoria Arsenal",  0.62, 2.10, 30.2, 3.5, "espn_live"),
        ("media", "over25",    "Over 2.5 goles",    0.58, 2.05, 18.9, 1.2, "exact_match"),
        ("baja",  "btts_si",   "Ambos marcan — SÍ", 0.53, 1.90,  5.7, 0.4, "espn_live"),
    ]

    for conf, market, display, prob, odds, edge, kelly, source in configs:
        rows.append({
            "home_team":        "Arsenal",
            "away_team":        "Chelsea",
            "league":           "Premier League",
            "match_date":       today,
            "market":           market,
            "market_display":   display,
            "model_prob":       prob,
            "model_prob_pct":   f"{prob*100:.1f}%",
            "reference_odds":   odds,
            "odds_source":      source,
            "odds_are_real":    source != "model_implied",
            "edge_pct":         edge,
            "kelly_pct":        kelly,
            "confidence":       conf,
            "explanation":      "forma local 2.1pts/PJ | ELO +50",
            "exp_home_goals":   1.60,
            "exp_away_goals":   1.20,
            "exp_total_goals":  2.80,
            "n_home_matches":   50,
            "n_away_matches":   50,
        })

    return pd.DataFrame(rows)