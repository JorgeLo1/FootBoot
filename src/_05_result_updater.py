"""
_05_result_updater.py
Cierra las predicciones del día anterior con el resultado real.

FUENTES de resultados (en orden de prioridad):
  1. football-data.org (get_results_fdorg) — free permanente
  2. API-Football como fallback si está configurada
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
    API_FOOTBALL_KEY, SUPABASE_URL, SUPABASE_KEY,
    DATA_PROCESSED, LIGAS,
)

# Import al nivel del módulo — falla claro al arranque si hay problema
try:
    from src._01_data_collector import get_results_fdorg
    _HAS_FDORG = True
except ImportError as e:
    _HAS_FDORG = False
    logging.getLogger(__name__).warning(f"get_results_fdorg no disponible: {e}")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def init_supabase():
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        log.error(f"Error conectando Supabase: {e}")
        return None


def get_results_from_api(target_date: str) -> dict:
    """
    Obtiene resultados del día.
    Prioridad: football-data.org → API-Football (fallback).
    """
    results = {}

    # 1. football-data.org (fuente principal)
    if _HAS_FDORG:
        try:
            results = get_results_fdorg(target_date)
            if results:
                log.info(f"Resultados obtenidos de fd.org: {len(results)} partidos")
                return results
        except Exception as e:
            log.warning(f"fd.org falló: {e}. Intentando API-Football...")

    # 2. API-Football como fallback
    if not API_FOOTBALL_KEY:
        log.warning("Sin resultados: ni fd.org ni API_FOOTBALL_KEY disponibles.")
        return {}

    url     = "https://v3.football.api-sports.io/fixtures"
    headers = {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key":  API_FOOTBALL_KEY,
    }

    for league_id, (league_name, _, __) in LIGAS.items():
        try:
            resp = requests.get(
                url, headers=headers,
                params={"date": target_date, "league": league_id,
                        "season": 2024, "status": "FT"},
                timeout=10,
            )
            resp.raise_for_status()
            for f in resp.json().get("response", []):
                teams = f["teams"]
                goals = f["goals"]
                hg    = goals.get("home", 0) or 0
                ag    = goals.get("away", 0) or 0
                results[(teams["home"]["name"], teams["away"]["name"])] = {
                    "home_goals": int(hg),
                    "away_goals": int(ag),
                    "fixture_id": f["fixture"]["id"],
                    "status":     f["fixture"]["status"]["short"],
                }
        except Exception as e:
            log.warning(f"Error API-Football {league_name}: {e}")

    log.info(f"Resultados obtenidos de API-Football: {len(results)} partidos")
    return results


def evaluate_bet(market: str, home_goals: int, away_goals: int) -> bool:
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
    return bool(evaluator()) if evaluator else False


def save_predictions_to_supabase(bets_df: pd.DataFrame, sb) -> int:
    if bets_df.empty or sb is None:
        return 0
    inserted = 0
    for _, bet in bets_df.iterrows():
        try:
            record = {
                "partido":          f"{bet['home_team']} vs {bet['away_team']}",
                "home_team":        bet["home_team"],
                "away_team":        bet["away_team"],
                "liga":             bet.get("league", ""),
                "fecha_partido":    str(bet.get("match_date", date.today())),
                "mercado":          bet["market"],
                "mercado_display":  bet["market_display"],
                "prediccion":       bet["market_display"],
                "prob_modelo":      float(bet["model_prob"]),
                "cuota_referencia": float(bet["reference_odds"]),
                "edge_pct":         float(bet["edge_pct"]),
                "kelly_pct":        float(bet["kelly_pct"]),
                "confianza":        bet["confidence"],
                "explicacion":      bet.get("explanation", ""),
                "resultado":        None,
                "ganada":           None,
                "creado_en":        datetime.utcnow().isoformat(),
                "modelo_version":   date.today().strftime("%Y%m%d"),
            }
            sb.table("predicciones").insert(record).execute()
            inserted += 1
        except Exception as e:
            log.warning(f"Error insertando en Supabase: {e}")
    log.info(f"Predicciones guardadas en Supabase: {inserted}")
    return inserted


def update_results_in_supabase(target_date: str, real_results: dict, sb) -> int:
    if sb is None:
        return 0
    try:
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
        home   = pred.get("home_team", "")
        away   = pred.get("away_team", "")
        result = real_results.get((home, away))

        if not result:
            for (h, a), r in real_results.items():
                if (home.lower() in h.lower() or h.lower() in home.lower()) and \
                   (away.lower() in a.lower() or a.lower() in away.lower()):
                    result = r
                    break

        if not result:
            continue

        hg  = result["home_goals"]
        ag  = result["away_goals"]
        won = evaluate_bet(pred["mercado"], hg, ag)

        try:
            sb.table("predicciones").update({
                "resultado":  f"{hg}-{ag}",
                "home_goals": hg,
                "away_goals": ag,
                "ganada":     won,
                "cerrado_en": datetime.utcnow().isoformat(),
            }).eq("id", pred["id"]).execute()
            updated += 1
            log.info(
                f"{home} vs {away} | {pred['mercado']} → "
                f"{hg}-{ag} → {'GANADA' if won else 'perdida'}"
            )
        except Exception as e:
            log.warning(f"Error actualizando Supabase: {e}")

    return updated


def compute_model_stats(sb) -> dict:
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

    total   = len(df)
    ganadas = df["ganada"].sum()
    tasa    = ganadas / total if total > 0 else 0

    df["ganada"]     = df["ganada"].astype(bool)
    df["roi_unidad"] = df.apply(
        lambda r: (r["cuota_referencia"] - 1) if r["ganada"] else -1, axis=1
    )
    roi = df["roi_unidad"].mean() * 100

    stats = {
        "total":    total,
        "ganadas":  int(ganadas),
        "tasa_pct": round(tasa * 100, 1),
        "roi_pct":  round(roi, 2),
    }
    for nivel in ["alta", "media"]:
        sub = df[df["confianza"] == nivel]
        if not sub.empty:
            stats[f"tasa_{nivel}_pct"] = round(sub["ganada"].sum() / len(sub) * 100, 1)
            stats[f"roi_{nivel}_pct"]  = round(sub["roi_unidad"].mean() * 100, 2)

    for market, group in df.groupby("mercado"):
        stats[f"tasa_{market}_pct"] = round(
            group["ganada"].sum() / len(group) * 100, 1
        )

    log.info(f"Stats → Tasa: {stats['tasa_pct']}% | ROI: {stats['roi_pct']}%")

    try:
        sb.table("estadisticas_modelo").insert({
            "fecha":               str(date.today()),
            "total_predicciones":  total,
            "tasa_acierto":        round(tasa, 4),
            "roi_acumulado":       round(roi, 4),
            "calculado_en":        datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"Error guardando stats en Supabase: {e}")

    return stats


def create_supabase_tables_sql() -> str:
    return """
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

CREATE TABLE IF NOT EXISTS estadisticas_modelo (
    id                  UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    fecha               DATE NOT NULL,
    total_predicciones  INTEGER,
    tasa_acierto        NUMERIC(6,4),
    roi_acumulado       NUMERIC(8,4),
    calculado_en        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_fecha     ON predicciones(fecha_partido);
CREATE INDEX IF NOT EXISTS idx_pred_confianza ON predicciones(confianza);
CREATE INDEX IF NOT EXISTS idx_pred_ganada    ON predicciones(ganada);
CREATE INDEX IF NOT EXISTS idx_pred_mercado   ON predicciones(mercado);

CREATE OR REPLACE VIEW roi_por_mercado AS
SELECT
    mercado, confianza,
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
    log.info("═══ FOOTBOT · Actualización de resultados ═══")
    sb        = init_supabase()
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    log.info(f"Obteniendo resultados del {yesterday}...")
    real_results = get_results_from_api(yesterday)
    if real_results:
        updated = update_results_in_supabase(yesterday, real_results, sb)
        log.info(f"Predicciones cerradas: {updated}")
    stats = compute_model_stats(sb)
    if stats:
        log.info(f"Estadísticas: {stats}")
    log.info("═══ Actualización completada ═══")
    return stats


if __name__ == "__main__":
    print("═══ SQL para crear tablas en Supabase ═══")
    print(create_supabase_tables_sql())
    print("\n═══ Ejecutando actualización de resultados... ═══")
    run()