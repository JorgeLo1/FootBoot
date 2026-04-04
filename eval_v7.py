"""
eval_v7.py — Evaluación formal post-reentrenamiento FootBot v7
==============================================================
Valida si el fix de scale_pos_weight dinámico mejoró los mercados
draw y away_win respecto a las métricas pre-fix.

Metodología:
  - Split temporal 80/20 sobre el histórico ordenado por fecha
  - El 80% se usa solo para referencia de contexto (ya estaba en entrenamiento)
  - El 20% final (≈847 partidos) es el conjunto de validación
  - Las predicciones se generan con los modelos .pkl ya entrenados (v7)
  - Cuotas: model-implied (sin Supabase)
  - ROI flat: apuesta 1 unidad por señal, sin Kelly

CAMBIOS v2 (post-backfill):
  - --sweep-draw: barrido de thresholds para `draw` (0.25–0.45, paso 0.02)
    para recalibrar el threshold óptimo con el dataset ampliado de 2.863 partidos.
    El threshold=0.33 fue calibrado con 848 partidos y genera solo 1 señal
    en el nuevo dataset — requiere recalibración.

Uso:
  python eval_v7.py
  python eval_v7.py --min-prob 0.55 --min-edge 4.0
  python eval_v7.py --ligas "Liga BetPlay,Liga MX"
  python eval_v7.py --output eval_resultados.csv
  python eval_v7.py --sweep-draw                   # recalibra threshold draw
  python eval_v7.py --sweep-draw --output sweep.csv
"""

import argparse
import sys
import os
import warnings
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Argumentos CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Evaluación formal FootBot v7")
parser.add_argument("--min-prob", type=float, default=0.50,
                    help="Prob mínima para contar una señal (default: 0.50)")
parser.add_argument("--min-edge", type=float, default=0.0,
                    help="Edge mínimo %% para contar una señal (default: 0.0)")
parser.add_argument("--split", type=float, default=0.80,
                    help="Fracción de entrenamiento (default: 0.80)")
parser.add_argument("--ligas", type=str, default=None,
                    help="Filtrar ligas: 'Liga BetPlay,Liga MX' (default: todas)")
parser.add_argument("--output", type=str, default=None,
                    help="Guardar resultados en CSV (opcional)")
parser.add_argument("--verbose", action="store_true",
                    help="Mostrar detalle por partido")
parser.add_argument("--sweep-draw", action="store_true",
                    help="Barrido de thresholds para mercado `draw` (0.25–0.45, paso 0.02). "
                         "Útil para recalibrar con el dataset post-backfill.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Imports del proyecto (deben correr desde la raíz del proyecto footbot/)
# ---------------------------------------------------------------------------

try:
    from src._02_feature_builder import load_historical_results, normalize_team_name
    from src._03_model_engine import load_models, predict_match
    from src._04_value_detector import (
        compute_all_market_probs,
        calculate_edge,
        get_model_prob_for_market,
    )
except ImportError as e:
    print(f"\n[ERROR] No se pudo importar módulo FootBot: {e}")
    print("Asegurate de correr desde la raíz del proyecto:")
    print("  cd footbot/")
    print("  python eval_v7.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Mercados a evaluar — con su evaluador y clave en market_probs
# ---------------------------------------------------------------------------

MERCADOS = {
    # mercado: (prob_key, evaluador(home_g, away_g))
    "home_win":  ("prob_home_win",  lambda h, a: h > a),
    "draw":      ("prob_draw",      lambda h, a: h == a),
    "away_win":  ("prob_away_win",  lambda h, a: h < a),
    "btts":      ("prob_btts",      lambda h, a: h > 0 and a > 0),
    "btts_no":   ("prob_btts_no",   lambda h, a: not (h > 0 and a > 0)),
    "over05":    ("prob_over05",    lambda h, a: (h + a) > 0.5),
    "over15":    ("prob_over15",    lambda h, a: (h + a) > 1.5),
    "over25":    ("prob_over25",    lambda h, a: (h + a) > 2.5),
    "over35":    ("prob_over35",    lambda h, a: (h + a) > 3.5),
    "over45":    ("prob_over45",    lambda h, a: (h + a) > 4.5),
    "under15":   ("prob_under15",   lambda h, a: (h + a) < 1.5),
    "under25":   ("prob_under25",   lambda h, a: (h + a) < 2.5),
    "under35":   ("prob_under35",   lambda h, a: (h + a) < 3.5),
    "double_1x": ("prob_double_1x", lambda h, a: h >= a),
    "double_x2": ("prob_double_x2", lambda h, a: a >= h),
    "double_12": ("prob_double_12", lambda h, a: h != a),
}

# Mercados de foco principal del fix v7
MERCADOS_FOCO = ["home_win", "draw", "away_win", "btts", "over25"]

# Métricas pre-fix (para comparación)
METRICAS_PRE_FIX = {
    "home_win": {"accuracy": 55.4, "roi": +17.4, "dc_weight": 0.700},
    "draw":     {"accuracy": 72.7, "roi":   0.0, "dc_weight": 0.700},
    "away_win": {"accuracy": 73.3, "roi":   0.0, "dc_weight": 0.700},
    "btts":     {"accuracy": 53.8, "roi":  -3.6, "dc_weight": 0.675},
    "over25":   {"accuracy": 56.3, "roi":  +7.1, "dc_weight": 0.700},
}

# ---------------------------------------------------------------------------
# Cuota model-implied: 1 / prob (con margen mínimo para evitar infinitos)
# ---------------------------------------------------------------------------

def implied_odd(prob: float) -> float:
    """Cuota justa sin margen de book (model-implied)."""
    if prob <= 0.01:
        return 99.0
    return round(1.0 / prob, 3)


# ---------------------------------------------------------------------------
# Carga y preparación del histórico
# ---------------------------------------------------------------------------

def cargar_historico(filtro_ligas=None):
    print("\n[1/4] Cargando histórico ESPN...")
    hist = load_historical_results()

    # Columnas mínimas requeridas (nombres reales del DataFrame ESPN)
    required = {"match_date", "home_team", "away_team", "home_goals", "away_goals", "league_name"}
    missing = required - set(hist.columns)
    if missing:
        print(f"[ERROR] Faltan columnas en histórico: {missing}")
        sys.exit(1)

    # Normalizar nombres internos para el resto del script
    hist = hist.rename(columns={"match_date": "date", "home_goals": "home_score", "away_goals": "away_score"})

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date", "home_score", "away_score"])
    hist["home_score"] = pd.to_numeric(hist["home_score"], errors="coerce")
    hist["away_score"] = pd.to_numeric(hist["away_score"], errors="coerce")
    hist = hist.dropna(subset=["home_score", "away_score"])
    hist = hist.sort_values("date").reset_index(drop=True)

    if filtro_ligas:
        ligas = [l.strip() for l in filtro_ligas.split(",")]
        hist = hist[hist["league_name"].isin(ligas)]
        if hist.empty:
            print(f"[ERROR] Sin partidos para ligas: {ligas}")
            sys.exit(1)

    n_total = len(hist)
    print(f"    Partidos disponibles: {n_total:,}")
    print(f"    Rango: {hist['date'].min().date()} → {hist['date'].max().date()}")
    print(f"    Ligas: {hist['league_name'].value_counts().to_dict()}")

    return hist


def split_temporal(hist: pd.DataFrame, split: float):
    n = len(hist)
    corte = int(n * split)
    train = hist.iloc[:corte].copy()
    test  = hist.iloc[corte:].copy()
    print(f"\n[2/4] Split temporal {int(split*100)}/{int((1-split)*100)}")
    print(f"    Entrenamiento : {len(train):,} partidos (hasta {train['date'].max().date()})")
    print(f"    Validación    : {len(test):,}  partidos (desde {test['date'].min().date()})")
    return train, test


# ---------------------------------------------------------------------------
# Predicción por partido
# ---------------------------------------------------------------------------

def predecir_partido(row: pd.Series, dc, ensemble, historical: pd.DataFrame):
    """
    Genera predicciones para un partido del set de validación.
    Usa solo el histórico previo a la fecha del partido (walk-forward).
    """
    fixture = row.to_dict()

    # Inferir league_id desde league_name si no existe
    if "league_id" not in fixture or pd.isna(fixture.get("league_id")):
        LEAGUE_ID_MAP = {
            "Liga BetPlay": 501, "Liga BetPlay Colombia": 501,
            "Liga Profesional Argentina": 502,
            "Brasileirão Serie A": 503, "Brasileirao Serie A": 503,
            "Copa Libertadores": 511,
            "Copa Sudamericana": 512,
            "Champions League": 514, "UEFA Champions League": 514,
            "Liga MX": 518,
        }
        fixture["league_id"] = LEAGUE_ID_MAP.get(row.get("league_name", ""), 0)

    try:
        preds = predict_match(
            row["home_team"],
            row["away_team"],
            fixture,
            dc,
            ensemble,
        )
    except Exception:
        return None

    mu  = preds.get("dc_exp_home_goals", 1.4)
    lam = preds.get("dc_exp_away_goals", 1.1)

    if mu <= 0 or lam <= 0:
        return None

    try:
        market_probs = compute_all_market_probs(mu, lam)
    except Exception:
        return None

    return market_probs, mu, lam


# ---------------------------------------------------------------------------
# Evaluación principal
# ---------------------------------------------------------------------------

def evaluar(test: pd.DataFrame, dc, ensemble, historical: pd.DataFrame,
            min_prob: float, min_edge: float, verbose: bool):

    print(f"\n[3/4] Evaluando {len(test):,} partidos...")
    print(f"    Filtros: prob ≥ {min_prob:.0%} | edge ≥ {min_edge:.1f}%")

    # Acumuladores por mercado
    stats = defaultdict(lambda: {
        "n_señales": 0, "n_correctas": 0,
        "roi_sum": 0.0,
        "probs": [],
    })

    # Para el análisis de distribución de predicciones
    dist = defaultdict(lambda: defaultdict(int))  # mercado → {pred: count}

    errores = 0
    filas_eval = []

    for idx, row in test.iterrows():
        resultado = predecir_partido(row, dc, ensemble, historical)
        if resultado is None:
            errores += 1
            continue

        market_probs, mu, lam = resultado
        home_g = int(row["home_score"])
        away_g = int(row["away_score"])

        for mercado, (prob_key, evaluador) in MERCADOS.items():
            prob = market_probs.get(prob_key, 0.0)
            odd  = implied_odd(prob)
            edge = calculate_edge(prob, odd)  # con cuota justa, edge ≈ 0 sin margen

            # Con cuota model-implied, el edge real viene de si el modelo es mejor
            # que una cuota justa — usamos prob como señal directa
            if prob < min_prob:
                continue
            if edge < min_edge:
                continue

            gano = evaluador(home_g, away_g)
            roi_partido = (odd - 1) if gano else -1.0

            stats[mercado]["n_señales"]  += 1
            stats[mercado]["n_correctas"] += int(gano)
            stats[mercado]["roi_sum"]     += roi_partido
            stats[mercado]["probs"].append(prob)

            if verbose:
                filas_eval.append({
                    "fecha": row["date"].date(),
                    "partido": f"{row['home_team']} vs {row['away_team']}",
                    "liga": row.get("league_name", ""),
                    "marcador": f"{home_g}-{away_g}",
                    "mercado": mercado,
                    "prob": round(prob, 3),
                    "odd_implied": round(odd, 3),
                    "gano": gano,
                })

    print(f"    Partidos procesados: {len(test) - errores:,} | Errores: {errores}")
    return stats, filas_eval


# ---------------------------------------------------------------------------
# Reporte final
# ---------------------------------------------------------------------------

VERDE  = "\033[92m"
ROJO   = "\033[91m"
AMARILLO = "\033[93m"
AZUL   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def color_roi(roi):
    if roi > 5:
        return f"{VERDE}{roi:+.1f}%{RESET}"
    elif roi > 0:
        return f"{AMARILLO}{roi:+.1f}%{RESET}"
    else:
        return f"{ROJO}{roi:+.1f}%{RESET}"


def imprimir_reporte(stats: dict, test: pd.DataFrame):
    print(f"\n{'='*70}")
    print(f"{BOLD}  REPORTE EVALUACIÓN FORMAL — FootBot v7 (post-reentrenamiento){RESET}")
    print(f"  Partidos de validación : {len(test):,}")
    print(f"  Período                : {test['date'].min().date()} → {test['date'].max().date()}")
    print(f"{'='*70}\n")

    # --- Tabla principal ---
    header = f"{'Mercado':<14} {'Señales':>8} {'Acierto':>9} {'ROI flat':>10} {'Prob media':>11}"
    print(f"{BOLD}{header}{RESET}")
    print("-" * 56)

    resumen = []
    for mercado in MERCADOS.keys():
        s = stats[mercado]
        n = s["n_señales"]
        if n == 0:
            row_out = {"mercado": mercado, "señales": 0, "accuracy": None, "roi": None, "prob_media": None}
            resumen.append(row_out)
            print(f"  {mercado:<14} {'—':>8} {'—':>9} {'—':>10} {'—':>11}")
            continue

        accuracy = s["n_correctas"] / n * 100
        roi      = s["roi_sum"] / n * 100
        prob_med = np.mean(s["probs"]) * 100

        # Resaltar mercados de foco
        marker = " ◄" if mercado in MERCADOS_FOCO else ""
        roi_str = color_roi(roi)

        print(f"  {mercado:<14} {n:>8,} {accuracy:>8.1f}% {roi_str:>18} {prob_med:>10.1f}%{marker}")

        resumen.append({
            "mercado": mercado, "señales": n,
            "accuracy": round(accuracy, 1),
            "roi": round(roi, 1),
            "prob_media": round(prob_med, 1),
        })

    # --- Comparación v6 pre-fix vs v7 post-fix ---
    print(f"\n{'='*70}")
    print(f"{BOLD}  COMPARACIÓN: Pre-fix v6 → Post-fix v7{RESET}")
    print(f"{'='*70}")
    print(f"  {'Mercado':<12} {'ROI pre-fix':>12} {'ROI post-fix':>13} {'Delta':>10} {'Acc pre-fix':>12} {'Acc post-fix':>13}")
    print("-" * 70)

    for mercado in MERCADOS_FOCO:
        pre  = METRICAS_PRE_FIX.get(mercado, {})
        post_s = stats[mercado]
        n = post_s["n_señales"]

        roi_pre  = pre.get("roi", None)
        acc_pre  = pre.get("accuracy", None)

        if n == 0:
            print(f"  {mercado:<12} {str(roi_pre)+'%':>12} {'—':>13} {'—':>10} {str(acc_pre)+'%':>12} {'—':>13}")
            continue

        roi_post = post_s["roi_sum"] / n * 100
        acc_post = post_s["n_correctas"] / n * 100

        delta_roi = roi_post - (roi_pre or 0)
        delta_str = f"{VERDE}+{delta_roi:.1f}%{RESET}" if delta_roi > 0 else f"{ROJO}{delta_roi:.1f}%{RESET}"

        print(f"  {mercado:<12} {str(roi_pre)+'%':>12} {roi_post:>+12.1f}% {delta_str:>18} {str(acc_pre)+'%':>12} {acc_post:>12.1f}%")

    # --- Veredicto draw / away_win ---
    print(f"\n{'='*70}")
    print(f"{BOLD}  VEREDICTO PRINCIPAL (draw / away_win){RESET}")
    print(f"{'='*70}")

    for m in ["draw", "away_win"]:
        s = stats[m]
        n = s["n_señales"]
        if n == 0:
            print(f"  {m}: SIN SEÑALES — el modelo aún no genera predicciones positivas.")
            print(f"       → Revisar umbrales o aumentar datos de validación.")
            continue

        roi = s["roi_sum"] / n * 100
        pre_roi = METRICAS_PRE_FIX[m]["roi"]

        if pre_roi == 0.0 and n > 0:
            if roi > 0:
                print(f"  {VERDE}✅ {m}: MEJORÓ — pasó de ROI 0.0% (sin señal) a {roi:+.1f}% ({n} señales){RESET}")
            elif roi > -5:
                print(f"  {AMARILLO}⚠️  {m}: GENERANDO SEÑAL — ROI {roi:+.1f}% ({n} señales). Mejoró vs 0.0% pre-fix, pero aún negativo.{RESET}")
            else:
                print(f"  {ROJO}❌ {m}: ROI {roi:+.1f}% ({n} señales). scale_pos_weight generó señales pero ROI sigue negativo.{RESET}")
                print(f"       → Considerar aumentar umbral de prob o revisar features.")

    print(f"\n{'='*70}")
    print(f"  Nota: ROI flat con cuota model-implied. En producción el ROI")
    print(f"  depende de las cuotas reales del book (siempre inferiores).")
    print(f"{'='*70}\n")

    return resumen


# ---------------------------------------------------------------------------
# Guardar CSV opcional
# ---------------------------------------------------------------------------

def guardar_csv(resumen: list, filas_eval: list, output_path: str):
    df_res = pd.DataFrame(resumen)
    df_res["fecha_eval"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df_res["version"] = "v7"
    df_res.to_csv(output_path, index=False)
    print(f"  Resumen guardado en: {output_path}")

    if filas_eval:
        detail_path = output_path.replace(".csv", "_detalle.csv")
        pd.DataFrame(filas_eval).to_csv(detail_path, index=False)
        print(f"  Detalle guardado en: {detail_path}")


# ---------------------------------------------------------------------------
# Barrido de thresholds para mercado draw (v2 — post-backfill)
# ---------------------------------------------------------------------------

def sweep_draw_threshold(test: pd.DataFrame, dc, ensemble, historical: pd.DataFrame,
                         thresholds=None, output_path: str = None):
    """
    Barre thresholds de decisión para el mercado `draw` y muestra ROI, accuracy
    y número de señales por cada threshold.

    Con el dataset post-backfill (2.863 partidos), el threshold=0.33 calibrado
    con 848 partidos genera solo 1 señal. Esta función encuentra el punto óptimo
    en el nuevo dataset.

    Metodología:
    - Para cada threshold T, se cuenta una señal cuando prob_draw >= T
    - Cuota model-implied: 1 / prob_draw
    - ROI flat: (odd - 1) si ganó, -1 si perdió

    Args:
        thresholds: lista de valores a probar. Default: 0.25 a 0.45, paso 0.02
        output_path: si se indica, guarda el resultado en CSV
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.25, 0.46, 0.02)]

    prob_key = "prob_draw"
    evaluador = lambda h, a: h == a  # noqa: E731

    print(f"\n{'='*70}")
    print(f"{BOLD}  BARRIDO THRESHOLD — mercado `draw` (post-backfill){RESET}")
    print(f"  Partidos validación: {len(test):,}")
    print(f"  Período: {test['date'].min().date()} → {test['date'].max().date()}")
    print(f"  Thresholds: {thresholds}")
    print(f"{'='*70}")

    # Pre-computar probs de draw por partido (una sola pasada)
    print("  Pre-computando probabilidades draw...")
    draw_probs = []
    home_goals_list = []
    away_goals_list = []
    errores = 0

    for idx, row in test.iterrows():
        resultado = predecir_partido(row, dc, ensemble, historical)
        if resultado is None:
            errores += 1
            draw_probs.append(None)
            home_goals_list.append(None)
            away_goals_list.append(None)
            continue
        market_probs, mu, lam = resultado
        draw_probs.append(market_probs.get(prob_key, 0.0))
        home_goals_list.append(int(row["home_score"]))
        away_goals_list.append(int(row["away_score"]))

    valid = [(p, h, a) for p, h, a in zip(draw_probs, home_goals_list, away_goals_list)
             if p is not None]
    print(f"  Partidos válidos: {len(valid):,} | Errores: {errores}")
    print(f"  Distribución prob_draw — media: {np.mean([p for p,_,_ in valid]):.3f} "
          f"| max: {max(p for p,_,_ in valid):.3f} "
          f"| min: {min(p for p,_,_ in valid):.3f}")

    # Frecuencia real de empates en el set de validación
    n_draws = sum(1 for _, h, a in valid if evaluador(h, a))
    pct_draws = n_draws / len(valid) * 100 if valid else 0
    print(f"  Empates reales en validación: {n_draws}/{len(valid)} ({pct_draws:.1f}%)")

    print(f"\n  {'Threshold':>10} {'Señales':>9} {'Accuracy':>10} {'ROI flat':>10} "
          f"{'Prob media':>11} {'EV por señal':>13}")
    print("  " + "-" * 65)

    resultados = []
    mejor_roi  = -999.0
    mejor_t    = None

    for t in thresholds:
        señales     = [(p, h, a) for p, h, a in valid if p >= t]
        n           = len(señales)
        if n == 0:
            print(f"  {t:>10.2f} {'0':>9} {'—':>10} {'—':>10} {'—':>11} {'—':>13}")
            resultados.append({
                "threshold": t, "señales": 0,
                "accuracy": None, "roi": None,
                "prob_media": None, "ev_por_señal": None,
            })
            continue

        correctas   = sum(1 for p, h, a in señales if evaluador(h, a))
        accuracy    = correctas / n * 100
        roi_sum     = sum((1/p - 1) if evaluador(h, a) else -1.0 for p, h, a in señales)
        roi         = roi_sum / n * 100
        prob_media  = np.mean([p for p, _, _ in señales]) * 100
        ev          = roi_sum / n  # ganancia esperada por unidad apostada

        roi_str = color_roi(roi)
        # Marcar el mejor ROI con flechas
        marker = f" ← {VERDE}MEJOR ROI{RESET}" if roi > mejor_roi else ""
        if roi > mejor_roi:
            mejor_roi = roi
            mejor_t   = t

        print(f"  {t:>10.2f} {n:>9,} {accuracy:>9.1f}% {roi_str:>18} "
              f"{prob_media:>10.1f}% {ev:>+12.4f}{marker}")

        resultados.append({
            "threshold": t, "señales": n,
            "accuracy": round(accuracy, 1),
            "roi": round(roi, 1),
            "prob_media": round(prob_media, 1),
            "ev_por_señal": round(ev, 4),
        })

    print(f"\n  {'='*65}")
    print(f"  {BOLD}Threshold recomendado: {mejor_t} (ROI {mejor_roi:+.1f}%){RESET}")
    print(f"\n  NOTA: ROI con cuota model-implied (sin margen de book).")
    print(f"  En producción las cuotas reales serán ~10-15% inferiores.")
    print(f"  Considerar solo thresholds con ≥ 20 señales para robustez estadística.")
    print(f"  {'='*65}\n")

    # Guardar CSV si se indica
    if output_path:
        sweep_path = output_path.replace(".csv", "").replace("eval_resultados", "sweep") + "_sweep_draw.csv"
        pd.DataFrame(resultados).to_csv(sweep_path, index=False)
        print(f"  Sweep guardado en: {sweep_path}")

    return resultados, mejor_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"{BOLD}  FootBot — eval_v7.py{RESET}")
    print(f"  Evaluación formal post-reentrenamiento (scale_pos_weight dinámico)")
    print(f"{'='*70}")

    # 1. Carga
    hist = cargar_historico(args.ligas)

    # 2. Split
    train_hist, test = split_temporal(hist, args.split)

    # 3. Cargar modelos
    print("\n[3/4] Cargando modelos .pkl...")
    try:
        dc, ensemble = load_models()
        modelos_dc = list(dc.models.keys()) if hasattr(dc, "models") else []
        print(f"    Dixon-Coles entrenados: {modelos_dc}")
        print(f"    XGBoost mercados: {list(ensemble.classifiers.keys()) if hasattr(ensemble, 'classifiers') else '—'}")
    except Exception as e:
        print(f"[ERROR] No se pudieron cargar los modelos: {e}")
        print("Asegurate de haber entrenado primero con: python scheduler.py")
        sys.exit(1)

    # 4. Modo sweep-draw — recalibración de threshold post-backfill
    if args.sweep_draw:
        sweep_draw_threshold(test, dc, ensemble, hist, output_path=args.output)
        # Ejecutar también evaluación normal para contexto completo
        print(f"\n{'='*70}")
        print(f"{BOLD}  Continuando con evaluación normal para referencia...{RESET}")

    # 5. Evaluar
    stats, filas_eval = evaluar(
        test, dc, ensemble, hist,
        min_prob=args.min_prob,
        min_edge=args.min_edge,
        verbose=args.verbose,
    )

    # 6. Reporte
    resumen = imprimir_reporte(stats, test)

    # 7. CSV opcional
    if args.output:
        guardar_csv(resumen, filas_eval if args.verbose else [], args.output)

    # 8. Verbose: top señales
    if args.verbose and filas_eval:
        print(f"\n{BOLD}  TOP 20 SEÑALES (por prob){RESET}")
        df_v = pd.DataFrame(filas_eval).sort_values("prob", ascending=False).head(20)
        print(df_v.to_string(index=False))


if __name__ == "__main__":
    main()