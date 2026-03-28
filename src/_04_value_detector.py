"""
_04_value_detector.py
Detecta value bets comparando probabilidades del modelo con cuotas del mercado.

CORRECCIÓN PRINCIPAL: Las cuotas de referencia ahora son REALES.
Se obtienen de Football-Data.co.uk (B365/Pinnacle) para el par de equipos.
Solo se usa CUOTAS_FALLBACK cuando no hay cuotas históricas disponibles,
y en ese caso el edge calculado se marca como "estimado" para transparencia.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    UMBRAL_EDGE_ALTA, UMBRAL_EDGE_MEDIA,
    UMBRAL_PROB_ALTA, UMBRAL_PROB_MEDIA,
    MIN_PARTIDOS_ALTA, MIN_PARTIDOS_MEDIA,
    KELLY_FRACCION, DATA_PROCESSED,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Cuotas promedio del mercado europeo (fallback cuando no hay datos reales).
# Se usan SOLO si no hay cuotas de Football-Data disponibles.
# Valores ajustados al overround típico (~5-6%).
CUOTAS_FALLBACK = {
    "home_win":   2.10,
    "draw":       3.40,
    "away_win":   3.60,
    "btts_si":    1.85,
    "btts_no":    1.95,
    "over25":     1.85,
    "under25":    1.95,
    "double_1x":  1.35,
    "double_x2":  1.55,
    "double_12":  1.30,
}

MARKET_TO_PROB = {
    "home_win":  "prob_home_win",
    "draw":      "prob_draw",
    "away_win":  "prob_away_win",
    "btts_si":   "prob_btts",
    "over25":    "prob_over25",
}


def calculate_edge(model_prob: float, decimal_odds: float) -> float:
    """Edge% = (prob × odds - 1) × 100. Positivo = value."""
    if decimal_odds <= 1.0 or model_prob <= 0:
        return -999.0
    return round((model_prob * decimal_odds - 1) * 100, 2)


def kelly_fraction(model_prob: float, decimal_odds: float,
                   fraction: float = KELLY_FRACCION) -> float:
    """Kelly Criterion fraccionado. Retorna % del bankroll."""
    if decimal_odds <= 1.0 or model_prob <= 0:
        return 0.0
    b     = decimal_odds - 1
    kelly = (model_prob * b - (1 - model_prob)) / b
    return round(max(kelly * fraction, 0.0) * 100, 2)


def get_model_prob_for_market(market: str, predictions: dict) -> float:
    if market == "double_1x":
        return predictions.get("prob_home_win", 0) + predictions.get("prob_draw", 0)
    if market == "double_x2":
        return predictions.get("prob_draw", 0) + predictions.get("prob_away_win", 0)
    if market == "double_12":
        return predictions.get("prob_home_win", 0) + predictions.get("prob_away_win", 0)
    if market == "btts_no":
        return 1 - predictions.get("prob_btts", 0.5)
    if market == "under25":
        return 1 - predictions.get("prob_over25", 0.5)
    key = MARKET_TO_PROB.get(market)
    return predictions.get(key, 0) if key else 0


def build_odds_dict(home_team: str, away_team: str,
                    historical: pd.DataFrame) -> tuple[dict, bool]:
    """
    Construye el diccionario de cuotas para un partido.
    Retorna (odds_dict, son_reales).

    Prioridad:
    1. Cuotas reales de Football-Data (Pinnacle > Bet365)
    2. CUOTAS_FALLBACK (estimado)
    """
    try:
        from src._01_data_collector import get_best_closing_odds
        real = get_best_closing_odds(home_team, away_team, historical)
    except Exception:
        real = None

    if real:
        odds = {
            "home_win":  real["home"],
            "draw":      real["draw"],
            "away_win":  real["away"],
            "btts_si":   CUOTAS_FALLBACK["btts_si"],   # no disponible en FD
            "btts_no":   CUOTAS_FALLBACK["btts_no"],
            "over25":    real.get("over25") or CUOTAS_FALLBACK["over25"],
            "under25":   real.get("under25") or CUOTAS_FALLBACK["under25"],
            "double_1x": round(1 / (1/real["home"] + 1/real["draw"]) * 0.94, 2),
            "double_x2": round(1 / (1/real["draw"] + 1/real["away"]) * 0.94, 2),
            "double_12": round(1 / (1/real["home"] + 1/real["away"]) * 0.94, 2),
        }
        return odds, True

    return CUOTAS_FALLBACK.copy(), False


def classify_confidence(edge: float, model_prob: float,
                        n_home: int, n_away: int) -> str | None:
    if (edge >= UMBRAL_EDGE_ALTA and
        model_prob >= UMBRAL_PROB_ALTA and
        n_home >= MIN_PARTIDOS_ALTA and
        n_away >= MIN_PARTIDOS_ALTA):
        return "alta"
    if (edge >= UMBRAL_EDGE_MEDIA and
        model_prob >= UMBRAL_PROB_MEDIA and
        n_home >= MIN_PARTIDOS_MEDIA and
        n_away >= MIN_PARTIDOS_MEDIA):
        return "media"
    return None


def build_explanation(market: str, features_row: dict,
                       top_features: list) -> str:
    templates = {
        "home_xg_scored":    lambda v: f"xG ofensivo local {v:.2f}",
        "away_xg_scored":    lambda v: f"xG ofensivo visitante {v:.2f}",
        "home_xg_conceded":  lambda v: f"xG concedido local {v:.2f}",
        "away_xg_conceded":  lambda v: f"xG concedido visitante {v:.2f}",
        "home_forma":        lambda v: f"forma local {v:.1f} pts/PJ",
        "away_forma":        lambda v: f"forma visitante {v:.1f} pts/PJ",
        "elo_diff":          lambda v: f"ventaja ELO {abs(v):.0f} pts ({'local' if v>0 else 'visitante'})",
        "h2h_btts_rate":     lambda v: f"BTTS en {v*100:.0f}% H2H",
        "h2h_avg_goals":     lambda v: f"promedio {v:.1f} goles H2H",
        "xg_total_exp":      lambda v: f"xG total {v:.2f}",
        "home_days_rest":    lambda v: f"local {v:.0f}d descanso",
        "away_days_rest":    lambda v: f"visitante {v:.0f}d descanso",
        "fatiga_flag":       lambda v: "fatiga detectada (<4d)" if v else "",
        "rain_flag":         lambda v: "lluvia prevista" if v else "",
        "home_over25_rate":  lambda v: f"local Over2.5 en {v*100:.0f}%",
        "away_over25_rate":  lambda v: f"visitante Over2.5 en {v*100:.0f}%",
    }
    parts = []
    for feat in top_features[:3]:
        val = features_row.get(feat)
        if val is not None and feat in templates:
            try:
                text = templates[feat](float(val))
                if text:
                    parts.append(text)
            except Exception:
                pass
    return " | ".join(parts) or "análisis estadístico del equipo"


def analyze_fixture(fixture_row: pd.Series, predictions: dict,
                    historical: pd.DataFrame = None) -> list:
    """
    Analiza un partido y retorna todas las value bets.
    Ahora recibe el DataFrame histórico para obtener cuotas reales.
    """
    home   = fixture_row.get("home_team", "")
    away   = fixture_row.get("away_team", "")
    n_home = int(fixture_row.get("n_home_matches", 0))
    n_away = int(fixture_row.get("n_away_matches", 0))

    # Obtener cuotas (reales si hay, estimadas si no)
    if historical is not None and not historical.empty:
        reference_odds, odds_are_real = build_odds_dict(home, away, historical)
    else:
        reference_odds, odds_are_real = CUOTAS_FALLBACK.copy(), False

    if not odds_are_real:
        log.debug(f"{home} vs {away}: usando cuotas estimadas (sin datos reales)")

    markets_to_check = ["home_win", "away_win", "btts_si", "over25"]
    if n_home >= MIN_PARTIDOS_ALTA and n_away >= MIN_PARTIDOS_ALTA:
        markets_to_check += ["draw", "double_1x", "double_x2"]

    bets = []
    for market in markets_to_check:
        model_prob = get_model_prob_for_market(market, predictions)
        if model_prob <= 0:
            continue

        odds       = reference_odds.get(market, CUOTAS_FALLBACK.get(market, 2.0))
        edge       = calculate_edge(model_prob, odds)
        confidence = classify_confidence(edge, model_prob, n_home, n_away)
        if confidence is None:
            continue

        # Si las cuotas son estimadas, aplicar umbral de edge más conservador
        if not odds_are_real and edge < UMBRAL_EDGE_ALTA:
            confidence = None  # promover solo bets con edge alto si no hay cuotas reales
        if confidence is None:
            continue

        kelly = kelly_fraction(model_prob, odds)
        top_feats = predictions.get("top_features", {}).get(
            market.replace("_si", "").replace("_no", ""), []
        )
        explanation = build_explanation(market, fixture_row.to_dict(), top_feats)

        display_map = {
            "home_win":  f"Victoria {home}",
            "draw":      "Empate",
            "away_win":  f"Victoria {away}",
            "btts_si":   "Ambos marcan — SÍ",
            "btts_no":   "Ambos marcan — NO",
            "over25":    "Over 2.5 goles",
            "under25":   "Under 2.5 goles",
            "double_1x": f"DC {home}/Empate",
            "double_x2": f"DC Empate/{away}",
            "double_12": f"DC {home}/{away}",
        }

        bets.append({
            "home_team":      home,
            "away_team":      away,
            "league":         fixture_row.get("league_name", ""),
            "match_date":     fixture_row.get("match_date", str(date.today())),
            "market":         market,
            "market_display": display_map.get(market, market),
            "model_prob":     round(model_prob, 4),
            "model_prob_pct": f"{model_prob*100:.1f}%",
            "reference_odds": odds,
            "odds_source":    "real" if odds_are_real else "estimada",
            "edge_pct":       edge,
            "kelly_pct":      kelly,
            "confidence":     confidence,
            "explanation":    explanation,
            "exp_home_goals": round(predictions.get("dc_exp_home_goals", 0), 2),
            "exp_away_goals": round(predictions.get("dc_exp_away_goals", 0), 2),
            "n_home_matches": n_home,
            "n_away_matches": n_away,
        })

    bets.sort(key=lambda x: ({"alta": 0, "media": 1}.get(x["confidence"], 9),
                               -x["edge_pct"]))
    return bets


def detect_all_value_bets(features_df: pd.DataFrame,
                           all_predictions: list,
                           historical: pd.DataFrame = None) -> pd.DataFrame:
    all_bets = []

    for i, (_, fixture) in enumerate(features_df.iterrows()):
        if i >= len(all_predictions):
            break
        preds = all_predictions[i]
        bets  = analyze_fixture(fixture, preds, historical)
        all_bets.extend(bets)

        home, away = fixture["home_team"], fixture["away_team"]
        if bets:
            log.info(f"{home} vs {away}: {len(bets)} value bet(s)")
        else:
            log.info(f"{home} vs {away}: sin valor")

    if not all_bets:
        return pd.DataFrame()

    df   = pd.DataFrame(all_bets)
    path = os.path.join(DATA_PROCESSED, f"bets_{date.today()}.csv")
    df.to_csv(path, index=False)
    log.info(f"{len(df)} value bets guardadas → {path}")
    return df


def summarize_bets(bets_df: pd.DataFrame) -> dict:
    if bets_df.empty:
        return {"total": 0, "alta": 0, "media": 0}
    return {
        "total":    len(bets_df),
        "alta":     len(bets_df[bets_df["confidence"] == "alta"]),
        "media":    len(bets_df[bets_df["confidence"] == "media"]),
        "edge_max": round(bets_df["edge_pct"].max(), 2),
        "edge_avg": round(bets_df["edge_pct"].mean(), 2),
        "real_odds_pct": round(
            len(bets_df[bets_df["odds_source"] == "real"]) / len(bets_df) * 100, 1
        ) if "odds_source" in bets_df.columns else 0,
    }