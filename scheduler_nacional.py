import os
import sys
import logging
import argparse
import traceback
import pandas as pd
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_PROCESSED, MODELS_DIR, LOGS_DIR,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    SUPABASE_URL, SUPABASE_KEY,
    API_FOOTBALL_KEY,
)

os.makedirs(LOGS_DIR, exist_ok=True)
log_file = os.path.join(LOGS_DIR, f"nacional_{date.today()}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("footbot.nacional")

# Feature columns del modelo de selecciones
# (subconjunto de los features nacionales sin leakage)
NACIONAL_FEATURE_COLS = [
    "home_goals_scored",    "home_goals_conceded",
    "away_goals_scored",    "away_goals_conceded",
    "home_forma",           "away_forma",
    "home_forma_role",      "away_forma_role",
    "home_gf_role",         "away_gf_role",
    "home_gc_role",         "away_gc_role",
    "home_btts_rate",       "away_btts_rate",
    "home_over25_rate",     "away_over25_rate",
    "home_days_rest",       "away_days_rest",
    "home_racha",           "away_racha",
    "h2h_home_wins",        "h2h_draws",        "h2h_away_wins",
    "h2h_avg_goals",        "h2h_btts_rate",
    "fifa_diff",            "fifa_rank_diff",
    "goals_diff",           "forma_diff",
    "xg_total_exp",         "rest_diff",        "fatiga_flag",
    "competition_tier",     "is_neutral_venue", "is_knockout",
    "pts_diff_standing",    "pos_diff_standing",
]

MODEL_PATH_DC  = os.path.join(MODELS_DIR, "nacional_dixon_coles_latest.pkl")
MODEL_PATH_XGB = os.path.join(MODELS_DIR, "nacional_ensemble_latest.pkl")


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _has_models() -> bool:
    return os.path.exists(MODEL_PATH_DC) and os.path.exists(MODEL_PATH_XGB)


def _should_retrain(historical: pd.DataFrame) -> bool:
    no_model   = not _has_models()
    is_monday  = datetime.now().weekday() == 0
    enough_data = len(historical) >= 100
    return (no_model or is_monday) and enough_data


def _send_error(msg: str):
    try:
        from src.telegram_sender import send_error_notification
        send_error_notification(f"[NACIONAL] {msg}")
    except Exception:
        pass


def _format_nacional_message(bets_df: pd.DataFrame,
                              model_stats: dict = None) -> str:
    """Formatea el reporte de selecciones para Telegram."""
    today  = date.today()
    total  = len(bets_df) if not bets_df.empty else 0
    alta   = bets_df[bets_df["confidence"] == "alta"]  if not bets_df.empty else pd.DataFrame()
    media  = bets_df[bets_df["confidence"] == "media"] if not bets_df.empty else pd.DataFrame()

    lines = [
        f"🌎 *FOOTBOT SELECCIONES · {today.strftime('%d/%m/%Y')}*",
        f"_Eliminatorias · Copa América · Mundial_",
        f"_{total} apuesta{'s' if total != 1 else ''} encontrada{'s' if total != 1 else ''}_",
        "",
    ]

    def render(bet: pd.Series) -> list:
        odds_tag = "" if bet.get("odds_source") not in ("fallback", "estimada") \
            else " _(cuota estimada)_"
        return [
            f"*{bet['home_team']} vs {bet['away_team']}*",
            f"🏆 {bet.get('league', '')} · {bet.get('round', '')}",
            f"📌 {bet['market_display']}",
            (f"📊 Prob: `{bet['model_prob_pct']}`  "
             f"Cuota: `{bet['reference_odds']}`{odds_tag}  "
             f"Edge: `+{bet['edge_pct']}%`"),
            f"💰 Kelly: `{bet['kelly_pct']}%` bankroll",
            f"💡 _{bet.get('explanation', '')}_ ",
            "",
        ]

    if not alta.empty:
        lines += ["🟢 *ALTA CONFIANZA*", "─" * 28]
        for _, bet in alta.iterrows():
            lines += render(bet)

    if not media.empty:
        lines += ["🟡 *MEDIA CONFIANZA*", "─" * 28]
        for _, bet in media.iterrows():
            lines += render(bet)

    if total == 0:
        lines += [
            "❌ *Sin value bets hoy en selecciones*",
            "_No se encontró edge suficiente._",
            "",
        ]

    lines += [
        "─" * 28,
        "⚠️ _Modelo estadístico experimental. Apuesta con responsabilidad._",
    ]
    return "\n".join(lines)


def _format_live_goal(fixture: dict) -> str:
    """Mensaje de Telegram para un gol en vivo."""
    return (
        f"⚽ *GOL — {fixture['league_name']}*\n"
        f"*{fixture['home_team']}* {fixture['home_goals']} – "
        f"{fixture['away_goals']} *{fixture['away_team']}*\n"
        f"_Min. {fixture['elapsed']}_"
    )


# ─── MODOS DE EJECUCIÓN ───────────────────────────────────────────────────────

def run_calendar():
    """
    Modo 00:00 — actualiza fixtures y posiciones del día.
    Consume ~6 requests (3 fixtures + 3 standings).
    """
    log.info("=" * 55)
    log.info(f"  NACIONAL · calendar · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 55)

    if not API_FOOTBALL_KEY:
        log.error("API_FOOTBALL_KEY no configurada.")
        return

    try:
        from src.nacional_collector import run_daily
        result = run_daily()
        n_fixtures = len(result.get("fixtures", []))
        log.info(f"Calendar OK: {n_fixtures} partidos hoy")
    except Exception as e:
        log.error(f"Error en calendar: {e}\n{traceback.format_exc()}")
        _send_error(f"calendar falló:\n{e}")


def run_predict():
    """
    Modo 08:00 — genera predicciones para los partidos del día.
    """
    log.info("=" * 55)
    log.info(f"  NACIONAL · predict · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 55)

    if not API_FOOTBALL_KEY:
        log.error("API_FOOTBALL_KEY no configurada.")
        return

    from src.nacional_collector import (
        ApifootballClient, get_fixtures_hoy,
        get_standings, load_historical_nacional,
    )
    from src.nacional_features import (
        build_nacional_features, build_nacional_training_dataset,
    )
    from src._03_model_engine import (
        DixonColesEnsemble, FootbotEnsemble,
        train_and_save, load_models, predict_match, blend_predictions,
    )
    from src._04_value_detector import detect_all_value_bets, summarize_bets
    from src.telegram_sender import send_telegram

    client = ApifootballClient()

    # PASO 1 — Fixtures y posiciones
    log.info("> PASO 1 — Fixtures y posiciones")
    today    = date.today().strftime("%Y-%m-%d")
    fixtures = get_fixtures_hoy(client, today)

    if not fixtures:
        send_telegram(
            f"🌎 *FOOTBOT SELECCIONES · {today}*\n"
            "No hay partidos de selecciones hoy."
        )
        log.info("Sin partidos hoy. Pipeline detenido.")
        return

    standings = get_standings(client)
    log.info(f"Paso 1 OK: {len(fixtures)} partidos")

    # PASO 2 — Histórico y features
    log.info("> PASO 2 — Histórico y features")
    historical = load_historical_nacional()

    try:
        features_df = build_nacional_features(
            fixtures, historical, standings, client=None
        )
        if features_df.empty:
            _send_error("features_df vacío en predict nacional")
            return
        log.info(f"Paso 2 OK: {len(features_df)} partidos con features")
    except Exception as e:
        log.error(f"Error Paso 2: {e}\n{traceback.format_exc()}")
        _send_error(f"Paso 2 falló:\n{e}")
        return

    # PASO 3 — Modelo
    log.info("> PASO 3 — Modelo predictivo nacional")
    dc, ensemble = None, None
    all_predictions = []

    try:
        if _should_retrain(historical):
            log.info("Reentrenando modelo nacional...")
            training_df = build_nacional_training_dataset(historical)
            if len(training_df) >= 100:
                # Adaptar feature cols al modelo nacional
                _patch_feature_cols(ensemble)
                dc, ensemble = _train_nacional(training_df)
            else:
                log.warning(
                    f"Solo {len(training_df)} partidos históricos de selecciones. "
                    "Se necesitan ≥100 para entrenar. Usando modelo de clubes como base."
                )
                dc, ensemble = load_models()
        else:
            if _has_models():
                import joblib
                dc       = joblib.load(MODEL_PATH_DC)
                ensemble = joblib.load(MODEL_PATH_XGB)
                log.info("Modelos nacionales cargados desde disco.")
            else:
                log.warning("Sin modelo nacional. Usando modelo de clubes como base.")
                dc, ensemble = load_models()

        if dc is None or ensemble is None:
            _send_error("Modelos no disponibles en predict nacional")
            return

        for _, row in features_df.iterrows():
            pred = predict_match(
                row["home_team"], row["away_team"],
                row.to_dict(), dc, ensemble,
            )
            all_predictions.append(pred)

        log.info(f"Paso 3 OK: {len(all_predictions)} predicciones")

    except Exception as e:
        log.error(f"Error Paso 3: {e}\n{traceback.format_exc()}")
        _send_error(f"Paso 3 falló:\n{e}")
        return

    # PASO 4 — Value bets
    log.info("> PASO 4 — Value bets nacionales")
    bets_df = pd.DataFrame()
    try:
        bets_df = detect_all_value_bets(features_df, all_predictions)
        log.info(f"Paso 4 OK: {summarize_bets(bets_df)}")
    except Exception as e:
        log.error(f"Error Paso 4: {e}\n{traceback.format_exc()}")
        _send_error(f"Paso 4 falló:\n{e}")

    # PASO 5 — Supabase
    log.info("> PASO 5 — Guardando en Supabase")
    try:
        from src._05_result_updater import init_supabase, save_predictions_to_supabase
        sb = init_supabase()
        if sb and not bets_df.empty:
            save_predictions_to_supabase(bets_df, sb)
    except Exception as e:
        log.error(f"Error Paso 5: {e}")

    # PASO 6 — Telegram
    log.info("> PASO 6 — Enviando a Telegram")
    try:
        msg  = _format_nacional_message(bets_df)
        sent = send_telegram(msg)
        log.info("Paso 6 OK" if sent else "Paso 6: fallo Telegram")
    except Exception as e:
        log.error(f"Error Paso 6: {e}")

    log.info("=" * 55)
    log.info(f"  predict completado · {datetime.now().strftime('%H:%M:%S')}")
    log.info("=" * 55)


def run_live():
    """
    Modo live — polling adaptativo durante los partidos del día.
    Envía notificación de gol en tiempo real.
    Llamar justo antes del primer partido del día.
    """
    log.info("=" * 55)
    log.info(f"  NACIONAL · live · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 55)

    if not API_FOOTBALL_KEY:
        log.error("API_FOOTBALL_KEY no configurada.")
        return

    from src.nacional_collector import (
        ApifootballClient, get_fixtures_hoy, run_live_polling,
    )
    from src.telegram_sender import send_telegram

    client   = ApifootballClient()
    fixtures = get_fixtures_hoy(client)

    if not fixtures:
        log.info("Sin partidos hoy para hacer live polling.")
        return

    def on_goal(fixture):
        try:
            send_telegram(_format_live_goal(fixture))
        except Exception as e:
            log.warning(f"Error enviando gol: {e}")

    final_results = run_live_polling(client, fixtures, on_update=on_goal)
    log.info(f"Live polling completado: {len(final_results)} partidos procesados")


def run_results():
    """
    Modo 23:00 — cierra las predicciones del día con el resultado real.
    Reutiliza el result_updater de clubes adaptando la fuente.
    """
    log.info("=" * 55)
    log.info(f"  NACIONAL · results · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 55)

    if not API_FOOTBALL_KEY:
        log.error("API_FOOTBALL_KEY no configurada.")
        return

    from src.nacional_collector import (
        ApifootballClient, get_fixtures_hoy, FINISHED_STATUSES,
    )
    from src._05_result_updater import (
        init_supabase, update_results_in_supabase, compute_model_stats,
        evaluate_bet,
    )

    client    = ApifootballClient()
    today_str = date.today().strftime("%Y-%m-%d")
    fixtures  = get_fixtures_hoy(client, today_str)

    # Construir dict de resultados en el mismo formato que result_updater
    real_results = {}
    for f in fixtures:
        if f["status"] in FINISHED_STATUSES:
            real_results[(f["home_team"], f["away_team"])] = {
                "home_goals": int(f["home_goals"] or 0),
                "away_goals": int(f["away_goals"] or 0),
                "fixture_id": f["fixture_id"],
                "status":     f["status"],
            }

    log.info(f"Resultados finales: {len(real_results)} partidos")

    sb      = init_supabase()
    updated = update_results_in_supabase(today_str, real_results, sb)
    stats   = compute_model_stats(sb) or {}

    log.info(f"Predicciones cerradas: {updated} | Stats: {stats}")
    log.info("=" * 55)


# ─── ENTRENAMIENTO ESPECÍFICO ────────────────────────────────────────────────

def _train_nacional(training_df: pd.DataFrame):
    """
    Entrena y guarda los modelos específicos para selecciones nacionales.
    Usa el mismo DixonColesEnsemble + FootbotEnsemble pero con features nacionales.
    """
    import joblib, shutil
    from datetime import date as _date
    from src._03_model_engine import (
        DixonColesEnsemble, FootbotEnsemble, FEATURE_COLS,
    )

    log.info(f"Entrenando modelos nacionales con {len(training_df)} partidos...")

    # Overridear feature cols para el modelo nacional
    import src._03_model_engine as engine_mod
    original_cols     = engine_mod.FEATURE_COLS[:]
    engine_mod.FEATURE_COLS = [
        c for c in NACIONAL_FEATURE_COLS
        if c in training_df.columns
    ]

    try:
        dc       = DixonColesEnsemble().fit(training_df)
        ensemble = FootbotEnsemble().fit(training_df, dc_ensemble=dc)
    finally:
        # Restaurar siempre, incluso si falla
        engine_mod.FEATURE_COLS = original_cols

    os.makedirs(MODELS_DIR, exist_ok=True)
    version  = _date.today().strftime("%Y%m%d")
    dc_path  = os.path.join(MODELS_DIR, f"nacional_dixon_coles_{version}.pkl")
    xgb_path = os.path.join(MODELS_DIR, f"nacional_ensemble_{version}.pkl")

    joblib.dump(dc,       dc_path)
    joblib.dump(ensemble, xgb_path)
    shutil.copy2(dc_path,  MODEL_PATH_DC)
    shutil.copy2(xgb_path, MODEL_PATH_XGB)

    log.info(f"Modelos nacionales guardados: versión {version}")
    return dc, ensemble


def _patch_feature_cols(ensemble):
    """No-op — solo para claridad en el flujo. El patch real está en _train_nacional."""
    pass


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOOTBOT — Pipeline de selecciones")
    parser.add_argument(
        "--mode",
        choices=["calendar", "predict", "live", "results", "all"],
        default="predict",
        help=(
            "calendar = actualiza fixtures y posiciones (00:00)\n"
            "predict  = genera predicciones del día (08:00)\n"
            "live     = polling en tiempo real durante partidos\n"
            "results  = cierra predicciones del día (23:00)\n"
            "all      = ejecuta calendar + predict (útil para pruebas)"
        ),
    )
    args = parser.parse_args()

    if args.mode == "calendar":
        run_calendar()
    elif args.mode == "predict":
        run_predict()
    elif args.mode == "live":
        run_live()
    elif args.mode == "results":
        run_results()
    elif args.mode == "all":
        run_calendar()
        run_predict()