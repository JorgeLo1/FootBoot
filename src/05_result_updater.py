"""
05_result_updater.py
Cierra las predicciones del día anterior con el resultado real.
Actualiza Supabase con ganadas/perdidas y calcula estadísticas del modelo.
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
    API_FOOTBALL_KEY, SUPABASE_URL, SUPABASE_KEY, DATA_PROCESSED
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def init_supabase():
    """Inicializa el cliente de Supabase."""
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        log.error(f"Error conectando Supabase: {e}")
        return None


def get_results_from_api(target_date: str) -> dict:
    """
    Obtiene los resultados reales de la fecha dada desde API-Football.
    Retorna dict: {(home_team, away_team): {"home_goals": X, "away_goals": Y}}
    """
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key": API_FOOTBALL_KEY
    }
    
    results = {}
    
    from config.settings import LIGAS
    for league_id, (league_name, _) in LIGAS.items():
        try:
            resp = requests.get(
                url, headers=headers,
                params={"date": target_date, "league": league_id,
                        "season": 2024, "status": "FT"},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            
            for f in data.get("response", []):
                teams = f["teams"]
                goals = f["goals"]
                
                home = teams["home"]["name"]
                away = teams["away"]["name"]
                hg   = goals.get("home", 0) or 0
                ag   = goals.get("away", 0) or 0
                
                results[(home, away)] = {
                    "home_goals": int(hg),
                    "away_goals": int(ag),
                    "fixture_id": f["fixture"]["id"],
                    "status":     f["fixture"]["status"]["short"],
                }
                
        except Exception as e:
            log.warning(f"Error obteniendo resultados {league_name}: {e}")
    
    log.info(f"Resultados obtenidos: {len(results)} partidos finalizados")
    return results


def evaluate_bet(market: str, bet_prediction: str,
                 home_goals: int, away_goals: int) -> bool:
    """
    Evalúa si una apuesta fue ganada o perdida según el resultado real.
    """
    total = home_goals + away_goals
    btts  = home_goals > 0 and away_goals > 0
    
    evaluators = {
        "home_win":  lambda: home_goals > away_goals,
        "draw":      lambda: home_goals == away_goals,
        "away_win":  lambda: home_goals < away_goals,
        "btts_si":   lambda: btts,
        "btts_no":   lambda: not btts,
        "over25":    lambda: total > 2.5,
        "under25":   lambda: total < 2.5,
        "over35":    lambda: total > 3.5,
        "double_1x": lambda: home_goals >= away_goals,
        "double_x2": lambda: away_goals >= home_goals,
        "double_12": lambda: home_goals != away_goals,
    }
    
    evaluator = evaluators.get(market)
    if evaluator:
        return bool(evaluator())
    return False


def save_predictions_to_supabase(bets_df: pd.DataFrame, sb) -> int:
    """
    Guarda las predicciones del día en Supabase (tabla: predicciones).
    Retorna número de registros insertados.
    """
    if bets_df.empty or sb is None:
        return 0
    
    inserted = 0
    for _, bet in bets_df.iterrows():
        try:
            record = {
                "partido":        f"{bet['home_team']} vs {bet['away_team']}",
                "home_team":      bet["home_team"],
                "away_team":      bet["away_team"],
                "liga":           bet.get("league", ""),
                "fecha_partido":  str(bet.get("match_date", date.today())),
                "mercado":        bet["market"],
                "mercado_display": bet["market_display"],
                "prediccion":     bet["market_display"],
                "prob_modelo":    float(bet["model_prob"]),
                "cuota_referencia": float(bet["reference_odds"]),
                "edge_pct":       float(bet["edge_pct"]),
                "kelly_pct":      float(bet["kelly_pct"]),
                "confianza":      bet["confidence"],
                "explicacion":    bet.get("explanation", ""),
                "resultado":      None,    # pendiente
                "ganada":         None,    # pendiente
                "creado_en":      datetime.utcnow().isoformat(),
                "modelo_version": date.today().strftime("%Y%m%d"),
            }
            sb.table("predicciones").insert(record).execute()
            inserted += 1
        except Exception as e:
            log.warning(f"Error insertando predicción en Supabase: {e}")
    
    log.info(f"Predicciones guardadas en Supabase: {inserted}")
    return inserted


def update_results_in_supabase(target_date: str, real_results: dict, sb) -> int:
    """
    Cierra las predicciones del target_date con el resultado real.
    """
    if sb is None:
        return 0
    
    try:
        # Obtener predicciones pendientes de esa fecha
        response = (
            sb.table("predicciones")
            .select("*")
            .eq("fecha_partido", target_date)
            .is_("ganada", "null")
            .execute()
        )
        pending = response.data
    except Exception as e:
        log.error(f"Error consultando Supabase: {e}")
        return 0
    
    updated = 0
    for pred in pending:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        
        # Buscar resultado real
        result = real_results.get((home, away))
        if not result:
            # Intentar con orden inverso o nombre normalizado
            for (h, a), r in real_results.items():
                if (home.lower() in h.lower() or h.lower() in home.lower()) and \
                   (away.lower() in a.lower() or a.lower() in away.lower()):
                    result = r
                    break
        
        if not result:
            log.debug(f"Sin resultado para {home} vs {away}")
            continue
        
        hg = result["home_goals"]
        ag = result["away_goals"]
        won = evaluate_bet(pred["mercado"], pred["prediccion"], hg, ag)
        
        resultado_str = f"{hg}-{ag}"
        
        try:
            sb.table("predicciones").update({
                "resultado":      resultado_str,
                "home_goals":     hg,
                "away_goals":     ag,
                "ganada":         won,
                "cerrado_en":     datetime.utcnow().isoformat(),
            }).eq("id", pred["id"]).execute()
            updated += 1
            
            status = "GANADA" if won else "perdida"
            log.info(f"{home} vs {away} | {pred['mercado']} → {resultado_str} → {status}")
        except Exception as e:
            log.warning(f"Error actualizando Supabase: {e}")
    
    return updated


def compute_model_stats(sb) -> dict:
    """
    Calcula estadísticas del modelo desde Supabase y las guarda.
    Retorna dict con ROI, tasa de acierto, etc.
    """
    if sb is None:
        return {}
    
    try:
        response = (
            sb.table("predicciones")
            .select("*")
            .not_.is_("ganada", "null")
            .execute()
        )
        df = pd.DataFrame(response.data)
    except Exception as e:
        log.error(f"Error consultando estadísticas: {e}")
        return {}
    
    if df.empty:
        return {"total": 0}
    
    total    = len(df)
    ganadas  = df["ganada"].sum()
    tasa     = ganadas / total if total > 0 else 0
    
    # ROI simulado (apostando el Kelly sugerido sobre bankroll base)
    df["ganada"] = df["ganada"].astype(bool)
    df["roi_unidad"] = df.apply(
        lambda r: (r["cuota_referencia"] - 1) if r["ganada"] else -1, axis=1
    )
    roi = df["roi_unidad"].mean() * 100
    
    # Por confianza
    stats = {
        "total":     total,
        "ganadas":   int(ganadas),
        "tasa_pct":  round(tasa * 100, 1),
        "roi_pct":   round(roi, 2),
    }
    
    for nivel in ["alta", "media"]:
        sub = df[df["confianza"] == nivel]
        if not sub.empty:
            sub_gan = sub["ganada"].sum()
            sub_roi = sub["roi_unidad"].mean() * 100
            stats[f"tasa_{nivel}_pct"] = round(sub_gan / len(sub) * 100, 1)
            stats[f"roi_{nivel}_pct"]  = round(sub_roi, 2)
    
    # Por mercado
    for market, group in df.groupby("mercado"):
        g = group["ganada"].sum()
        t = len(group)
        stats[f"tasa_{market}_pct"] = round(g / t * 100, 1)
    
    log.info(f"Stats modelo → Tasa: {stats['tasa_pct']}% | ROI: {stats['roi_pct']}%")
    
    # Guardar en tabla estadísticas
    try:
        sb.table("estadisticas_modelo").insert({
            "fecha":           str(date.today()),
            "total_predicciones": total,
            "tasa_acierto":    round(tasa, 4),
            "roi_acumulado":   round(roi, 4),
            "calculado_en":    datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"Error guardando stats en Supabase: {e}")
    
    return stats


def create_supabase_tables_sql() -> str:
    """
    Retorna el SQL para crear las tablas en Supabase.
    Ejecutar una sola vez en el SQL Editor de Supabase.
    """
    return """
-- Tabla principal de predicciones
CREATE TABLE IF NOT EXISTS predicciones (
    id               UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    partido          TEXT NOT NULL,
    home_team        TEXT NOT NULL,
    away_team        TEXT NOT NULL,
    liga             TEXT,
    fecha_partido    DATE NOT NULL,
    mercado          TEXT NOT NULL,
    mercado_display  TEXT,
    prediccion       TEXT,
    prob_modelo      NUMERIC(6,4),
    cuota_referencia NUMERIC(6,3),
    edge_pct         NUMERIC(7,2),
    kelly_pct        NUMERIC(6,2),
    confianza        TEXT CHECK (confianza IN ('alta','media')),
    explicacion      TEXT,
    resultado        TEXT,
    home_goals       INTEGER,
    away_goals       INTEGER,
    ganada           BOOLEAN,
    creado_en        TIMESTAMPTZ DEFAULT NOW(),
    cerrado_en       TIMESTAMPTZ,
    modelo_version   TEXT
);

-- Tabla de estadísticas agregadas
CREATE TABLE IF NOT EXISTS estadisticas_modelo (
    id                  UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    fecha               DATE NOT NULL,
    total_predicciones  INTEGER,
    tasa_acierto        NUMERIC(6,4),
    roi_acumulado       NUMERIC(8,4),
    calculado_en        TIMESTAMPTZ DEFAULT NOW()
);

-- Índices para queries frecuentes
CREATE INDEX IF NOT EXISTS idx_pred_fecha ON predicciones(fecha_partido);
CREATE INDEX IF NOT EXISTS idx_pred_confianza ON predicciones(confianza);
CREATE INDEX IF NOT EXISTS idx_pred_ganada ON predicciones(ganada);
CREATE INDEX IF NOT EXISTS idx_pred_mercado ON predicciones(mercado);

-- Vista de ROI por liga y mercado (útil para el dashboard)
CREATE OR REPLACE VIEW roi_por_mercado AS
SELECT
    mercado,
    confianza,
    COUNT(*) AS total,
    SUM(CASE WHEN ganada THEN 1 ELSE 0 END) AS ganadas,
    ROUND(AVG(CASE WHEN ganada THEN 1.0 ELSE 0.0 END) * 100, 1) AS tasa_acierto_pct,
    ROUND(AVG(CASE WHEN ganada THEN cuota_referencia - 1 ELSE -1 END) * 100, 2) AS roi_pct
FROM predicciones
WHERE ganada IS NOT NULL
GROUP BY mercado, confianza
ORDER BY roi_pct DESC;
"""


def run():
    """Ejecuta la actualización completa de resultados."""
    log.info("═══ FOOTBOT · Actualización de resultados ═══")
    
    sb = init_supabase()
    
    # Actualizar resultados del día anterior
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    log.info(f"Obteniendo resultados del {yesterday}...")
    
    real_results = get_results_from_api(yesterday)
    
    if real_results:
        updated = update_results_in_supabase(yesterday, real_results, sb)
        log.info(f"Predicciones cerradas: {updated}")
    
    # Calcular estadísticas acumuladas
    stats = compute_model_stats(sb)
    if stats:
        log.info(f"Estadísticas del modelo: {stats}")
    
    log.info("═══ Actualización completada ═══")
    return stats


if __name__ == "__main__":
    # Imprimir el SQL para setup inicial de Supabase
    print("═══ SQL para crear tablas en Supabase ═══")
    print(create_supabase_tables_sql())
    print("\n═══ Ejecutando actualización de resultados... ═══")
    run()
