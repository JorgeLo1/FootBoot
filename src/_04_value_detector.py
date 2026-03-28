"""
04_value_detector.py
Detecta value bets comparando probabilidades del modelo con cuotas del mercado.
Aplica Kelly Criterion fraccionado y clasifica en Alta / Media confianza.
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
    KELLY_FRACCION, DATA_PROCESSED
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Cuotas de referencia estimadas por mercado (cuando no hay cuotas reales)
# Se usan solo como fallback — idealmente vendrían de Football-Data o scraping
CUOTAS_REFERENCIA_DEFAULT = {
    "home_win": 2.10,
    "draw":     3.40,
    "away_win": 3.60,
    "btts_si":  1.85,
    "btts_no":  1.95,
    "over25":   1.85,
    "under25":  1.95,
    "double_1x": 1.35,
    "double_x2": 1.55,
    "double_12": 1.30,
}

# Mapeo mercado → probabilidad del modelo
MARKET_TO_PROB = {
    "home_win":   "prob_home_win",
    "draw":       "prob_draw",
    "away_win":   "prob_away_win",
    "btts_si":    "prob_btts",
    "over25":     "prob_over25",
    "double_1x":  None,   # prob_home + prob_draw
    "double_x2":  None,   # prob_draw + prob_away
    "double_12":  None,   # prob_home + prob_away
}


def calculate_edge(model_prob: float, decimal_odds: float) -> float:
    """
    Edge% = (prob_modelo × cuota_decimal - 1) × 100
    Positivo = value bet (modelo ve más valor que el mercado)
    """
    if decimal_odds <= 1.0 or model_prob <= 0:
        return -999.0
    return round((model_prob * decimal_odds - 1) * 100, 2)


def kelly_fraction(model_prob: float, decimal_odds: float,
                   fraction: float = KELLY_FRACCION) -> float:
    """
    Kelly Criterion fraccionado.
    f* = (prob × (odds-1) - (1-prob)) / (odds-1) × fraccion
    Retorna el % del bankroll a apostar.
    """
    if decimal_odds <= 1.0 or model_prob <= 0:
        return 0.0
    b = decimal_odds - 1
    kelly = (model_prob * b - (1 - model_prob)) / b
    kelly_f = max(kelly * fraction, 0.0)
    return round(kelly_f * 100, 2)  # retorna en %


def get_model_prob_for_market(market: str, predictions: dict) -> float:
    """Extrae la probabilidad del modelo para un mercado dado."""
    if market == "double_1x":
        return predictions.get("prob_home_win", 0) + predictions.get("prob_draw", 0)
    elif market == "double_x2":
        return predictions.get("prob_draw", 0) + predictions.get("prob_away_win", 0)
    elif market == "double_12":
        return predictions.get("prob_home_win", 0) + predictions.get("prob_away_win", 0)
    elif market == "btts_no":
        return 1 - predictions.get("prob_btts", 0.5)
    elif market == "under25":
        return 1 - predictions.get("prob_over25", 0.5)
    
    prob_key = MARKET_TO_PROB.get(market)
    return predictions.get(prob_key, 0) if prob_key else 0


def classify_confidence(edge: float, model_prob: float,
                        n_home: int, n_away: int) -> str:
    """
    Clasifica la apuesta en:
    - 'alta':   edge > 8%, prob > 62%, datos suficientes
    - 'media':  edge > 4%, prob > 55%, datos mínimos
    - None:     no recomendar
    """
    if (edge >= UMBRAL_EDGE_ALTA and
        model_prob >= UMBRAL_PROB_ALTA and
        n_home >= MIN_PARTIDOS_ALTA and
        n_away >= MIN_PARTIDOS_ALTA):
        return "alta"
    
    elif (edge >= UMBRAL_EDGE_MEDIA and
          model_prob >= UMBRAL_PROB_MEDIA and
          n_home >= MIN_PARTIDOS_MEDIA and
          n_away >= MIN_PARTIDOS_MEDIA):
        return "media"
    
    return None


def build_shap_explanation(market: str, features_row: dict,
                           top_features: list) -> str:
    """
    Genera una explicación simple de la predicción basada en
    los features más importantes del modelo.
    """
    explanations = {
        "home_xg_scored":    lambda v: f"xG ofensivo local {v:.2f}",
        "away_xg_scored":    lambda v: f"xG ofensivo visitante {v:.2f}",
        "home_xg_conceded":  lambda v: f"xG concedido por local {v:.2f}",
        "away_xg_conceded":  lambda v: f"xG concedido por visitante {v:.2f}",
        "home_forma":        lambda v: f"forma local {v:.1f} pts/partido",
        "away_forma":        lambda v: f"forma visitante {v:.1f} pts/partido",
        "elo_diff":          lambda v: f"ventaja ELO de {abs(v):.0f} pts para {'local' if v>0 else 'visitante'}",
        "h2h_btts_rate":     lambda v: f"BTTS en {v*100:.0f}% de H2H históricos",
        "h2h_avg_goals":     lambda v: f"promedio {v:.1f} goles en H2H",
        "xg_total_exp":      lambda v: f"xG total esperado {v:.2f}",
        "home_days_rest":    lambda v: f"local con {v:.0f} días de descanso",
        "away_days_rest":    lambda v: f"visitante con {v:.0f} días de descanso",
        "fatiga_flag":       lambda v: "un equipo con fatiga (< 4 días descanso)" if v else "",
        "rain_flag":         lambda v: "lluvia prevista (reduce goles)" if v else "",
        "wind_flag":         lambda v: "viento fuerte previsto" if v else "",
        "market_prob_home":  lambda v: f"mercado da {v*100:.0f}% al local",
        "home_over25_rate":  lambda v: f"local Over 2.5 en {v*100:.0f}% de partidos",
        "away_over25_rate":  lambda v: f"visitante Over 2.5 en {v*100:.0f}% de partidos",
    }
    
    parts = []
    for feat in top_features[:3]:
        val = features_row.get(feat)
        if val is not None and feat in explanations:
            try:
                text = explanations[feat](float(val))
                if text:
                    parts.append(text)
            except Exception:
                continue
    
    return " | ".join(parts) if parts else "análisis histórico del equipo"


def analyze_fixture(fixture_row: pd.Series,
                    predictions: dict,
                    reference_odds: dict = None) -> list:
    """
    Analiza un partido y retorna todas las value bets encontradas.
    """
    home = fixture_row.get("home_team", "")
    away = fixture_row.get("away_team", "")
    n_home = int(fixture_row.get("n_home_matches", 0))
    n_away = int(fixture_row.get("n_away_matches", 0))
    
    if reference_odds is None:
        reference_odds = CUOTAS_REFERENCIA_DEFAULT
    
    # Seleccionar mercados a analizar según nivel de datos
    markets_to_check = ["home_win", "away_win", "btts_si", "over25"]
    if n_home >= MIN_PARTIDOS_ALTA and n_away >= MIN_PARTIDOS_ALTA:
        markets_to_check += ["draw", "double_1x", "double_x2"]
    
    bets = []
    for market in markets_to_check:
        model_prob = get_model_prob_for_market(market, predictions)
        if model_prob <= 0:
            continue
        
        odds = reference_odds.get(market, CUOTAS_REFERENCIA_DEFAULT.get(market, 2.0))
        edge = calculate_edge(model_prob, odds)
        confidence = classify_confidence(edge, model_prob, n_home, n_away)
        
        if confidence is None:
            continue
        
        kelly = kelly_fraction(model_prob, odds)
        
        # Generar explicación
        top_feats = predictions.get("top_features", {}).get(
            market.replace("_si","").replace("_no",""), []
        )
        explanation = build_shap_explanation(
            market, fixture_row.to_dict(), top_feats
        )
        
        # Nombre legible del mercado
        market_display = {
            "home_win":  f"Victoria {home}",
            "draw":      "Empate",
            "away_win":  f"Victoria {away}",
            "btts_si":   "Ambos marcan — SÍ",
            "btts_no":   "Ambos marcan — NO",
            "over25":    "Over 2.5 goles",
            "under25":   "Under 2.5 goles",
            "double_1x": f"Doble oportunidad {home}/Empate",
            "double_x2": f"Doble oportunidad Empate/{away}",
            "double_12": f"Doble oportunidad {home}/{away}",
        }.get(market, market)
        
        bets.append({
            "home_team":      home,
            "away_team":      away,
            "league":         fixture_row.get("league_name", ""),
            "match_date":     fixture_row.get("match_date", str(date.today())),
            "market":         market,
            "market_display": market_display,
            "model_prob":     round(model_prob, 4),
            "model_prob_pct": f"{model_prob*100:.1f}%",
            "reference_odds": odds,
            "edge_pct":       edge,
            "kelly_pct":      kelly,
            "confidence":     confidence,
            "explanation":    explanation,
            "exp_home_goals": round(predictions.get("dc_exp_home_goals", 0), 2),
            "exp_away_goals": round(predictions.get("dc_exp_away_goals", 0), 2),
            "n_home_matches": n_home,
            "n_away_matches": n_away,
        })
    
    # Ordenar por confianza y luego por edge
    confidence_order = {"alta": 0, "media": 1}
    bets.sort(key=lambda x: (confidence_order.get(x["confidence"], 9), -x["edge_pct"]))
    
    return bets


def detect_all_value_bets(features_df: pd.DataFrame,
                           all_predictions: list,
                           reference_odds: dict = None) -> pd.DataFrame:
    """
    Corre el detector para todos los partidos del día.
    Retorna DataFrame con todas las value bets encontradas.
    """
    all_bets = []
    
    for i, (_, fixture) in enumerate(features_df.iterrows()):
        if i >= len(all_predictions):
            break
        preds = all_predictions[i]
        bets  = analyze_fixture(fixture, preds, reference_odds)
        all_bets.extend(bets)
        
        if bets:
            log.info(f"{fixture['home_team']} vs {fixture['away_team']}: "
                     f"{len(bets)} value bet(s) encontradas")
        else:
            log.info(f"{fixture['home_team']} vs {fixture['away_team']}: sin valor")
    
    if not all_bets:
        log.info("No se encontraron value bets hoy.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_bets)
    
    # Guardar en processed
    path = os.path.join(DATA_PROCESSED, f"bets_{date.today()}.csv")
    df.to_csv(path, index=False)
    log.info(f"Value bets guardadas: {len(df)} → {path}")
    
    return df


def summarize_bets(bets_df: pd.DataFrame) -> dict:
    """Resumen estadístico de las apuestas del día."""
    if bets_df.empty:
        return {"total": 0, "alta": 0, "media": 0}
    
    return {
        "total":        len(bets_df),
        "alta":         len(bets_df[bets_df["confidence"] == "alta"]),
        "media":        len(bets_df[bets_df["confidence"] == "media"]),
        "edge_max":     round(bets_df["edge_pct"].max(), 2),
        "edge_avg":     round(bets_df["edge_pct"].mean(), 2),
        "kelly_avg":    round(bets_df["kelly_pct"].mean(), 2),
    }


if __name__ == "__main__":
    # Test con datos simulados
    fake_predictions = {
        "prob_home_win": 0.55,
        "prob_draw":     0.25,
        "prob_away_win": 0.20,
        "prob_btts":     0.68,
        "prob_over25":   0.70,
        "dc_exp_home_goals": 1.8,
        "dc_exp_away_goals": 1.1,
        "top_features": {
            "home_win": ["home_xg_scored","elo_diff","home_forma"],
            "btts":     ["h2h_btts_rate","xg_total_exp","home_xg_scored"],
            "over25":   ["xg_total_exp","home_over25_rate","away_over25_rate"],
        }
    }
    fake_fixture = pd.Series({
        "home_team": "Arsenal", "away_team": "Chelsea",
        "league_name": "Premier League", "match_date": str(date.today()),
        "n_home_matches": 35, "n_away_matches": 35,
        "home_xg_scored": 1.9, "away_xg_conceded": 1.4,
        "elo_diff": 80, "home_forma": 2.1,
        "h2h_btts_rate": 0.72, "xg_total_exp": 2.8,
    })
    
    bets = analyze_fixture(fake_fixture, fake_predictions)
    for bet in bets:
        print(f"\n[{bet['confidence'].upper()}] {bet['market_display']}")
        print(f"  Prob modelo: {bet['model_prob_pct']} | Cuota: {bet['reference_odds']}")
        print(f"  Edge: {bet['edge_pct']}% | Kelly: {bet['kelly_pct']}% bankroll")
        print(f"  Motivo: {bet['explanation']}")
