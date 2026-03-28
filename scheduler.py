"""
scheduler.py — FOOTBOT
Orquestador principal del pipeline diario.
"""

import os
import sys
import logging
import traceback
import pandas as pd
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_RAW, DATA_PROCESSED, MODELS_DIR, LOGS_DIR,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    FOOTBALL_DATA_ORG_KEY, SUPABASE_URL, SUPABASE_KEY,
)

os.makedirs(LOGS_DIR, exist_ok=True)
log_file = os.path.join(LOGS_DIR, f"footbot_{date.today()}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("footbot")


def validate_credentials() -> list[str]:
    placeholders = {"TU_BOT_TOKEN_AQUI", "TU_CHAT_ID_AQUI",
                    "TU_SUPABASE_URL_AQUI", "TU_SUPABASE_ANON_KEY_AQUI"}
    checks = {
        "FOOTBALL_DATA_ORG_KEY": FOOTBALL_DATA_ORG_KEY,
        "TELEGRAM_TOKEN":        TELEGRAM_TOKEN,
        "TELEGRAM_CHAT_ID":      TELEGRAM_CHAT_ID,
        "SUPABASE_URL":          SUPABASE_URL,
        "SUPABASE_KEY":          SUPABASE_KEY,
    }
    return [
        f"  ✗ {name} no configurada"
        for name, value in checks.items()
        if not value or value in placeholders
    ]


def should_retrain() -> bool:
    no_model  = not os.path.exists(os.path.join(MODELS_DIR, "ensemble_latest.pkl"))
    is_monday = datetime.now().weekday() == 0
    return no_model or is_monday


def run_pipeline():
    log.info("=" * 55)
    log.info(f"  FOOTBOT -- {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    log.info("=" * 55)

    issues = validate_credentials()
    if issues:
        log.warning("Credenciales no configuradas:")
        for issue in issues:
            log.warning(issue)

    from src.telegram_sender import send_error_notification, send_telegram

    retrain = should_retrain()
    log.info(f"Re-entrenar modelo hoy: {retrain}")

    # PASO 1 — Recoleccion de datos
    log.info("\n> PASO 1 -- Recoleccion de datos")
    try:
        from src._01_data_collector import (
            get_fixtures_today, download_football_data,
            download_statsbomb_data, download_elo_ratings,
        )
        fixtures = get_fixtures_today()
        if fixtures.empty:
            send_telegram(
                f"FOOTBOT -- {date.today()}\n"
                "No hay partidos en las ligas activas hoy."
            )
            return
        if retrain:
            log.info("Actualizando datos historicos...")
            download_football_data()
            download_statsbomb_data()
        download_elo_ratings()
        log.info(f"Paso 1 OK: {len(fixtures)} partidos")
    except Exception as e:
        log.error(f"Error Paso 1: {e}\n{traceback.format_exc()}")
        send_error_notification(f"Paso 1 fallo:\n{str(e)}")
        return

    # PASO 2 — Feature building
    log.info("\n> PASO 2 -- Construyendo features")
    historical = pd.DataFrame()
    try:
        from src._02_feature_builder import (
            build_features_for_fixtures, build_training_dataset,
            load_historical_results,
        )
        historical  = load_historical_results()
        features_df = build_features_for_fixtures(fixtures)
        if features_df.empty:
            send_error_notification("Paso 2: DataFrame de features vacio")
            return
        log.info(f"Paso 2 OK: {len(features_df)} partidos con features")
    except Exception as e:
        log.error(f"Error Paso 2: {e}\n{traceback.format_exc()}")
        send_error_notification(f"Paso 2 fallo:\n{str(e)}")
        return

    # PASO 3 — Modelo
    log.info("\n> PASO 3 -- Modelo predictivo")
    ensemble = None
    try:
        from src._03_model_engine import train_and_save, load_models, predict_match
        if retrain:
            log.info("Re-entrenando modelo...")
            try:
                training_df  = build_training_dataset(historical)
                dc, ensemble = train_and_save(training_df)
            except RuntimeError as e:
                log.error(f"Error guardando modelo: {e}")
                send_error_notification(
                    f"Modelo re-entrenado pero NO guardado:\n{e}\n"
                    "Se usaran los modelos anteriores."
                )
                dc, ensemble = load_models()
        else:
            dc, ensemble = load_models()

        if dc is None or ensemble is None:
            send_error_notification("Paso 3: modelos no disponibles")
            return

        all_predictions = [
            predict_match(row["home_team"], row["away_team"],
                          row.to_dict(), dc, ensemble)
            for _, row in features_df.iterrows()
        ]
        log.info(f"Paso 3 OK: {len(all_predictions)} predicciones")

        if retrain and ensemble is not None:
            try:
                send_telegram(ensemble.get_validation_summary())
            except Exception:
                pass

    except Exception as e:
        log.error(f"Error Paso 3: {e}\n{traceback.format_exc()}")
        send_error_notification(f"Paso 3 fallo:\n{str(e)}")
        return

    # PASO 4 — Value bets
    log.info("\n> PASO 4 -- Deteccion de value bets")
    bets_df = pd.DataFrame()
    try:
        from src._04_value_detector import detect_all_value_bets, summarize_bets
        # FIX critico: pasar historical para usar cuotas reales de Football-Data
        bets_df = detect_all_value_bets(features_df, all_predictions, historical)
        log.info(f"Paso 4 OK: {summarize_bets(bets_df)}")
    except Exception as e:
        log.error(f"Error Paso 4: {e}\n{traceback.format_exc()}")
        send_error_notification(f"Paso 4 fallo:\n{str(e)}")

    # PASO 5 — Supabase
    log.info("\n> PASO 5 -- Guardando en Supabase")
    model_stats = {}
    try:
        from src._05_result_updater import (
            init_supabase, save_predictions_to_supabase, compute_model_stats,
        )
        sb = init_supabase()
        if not bets_df.empty and sb:
            save_predictions_to_supabase(bets_df, sb)
        model_stats = compute_model_stats(sb) or {}
        log.info(f"Paso 5 OK: {model_stats}")
    except Exception as e:
        log.error(f"Error Paso 5: {e}\n{traceback.format_exc()}")

    # PASO 6 — Telegram
    log.info("\n> PASO 6 -- Enviando a Telegram")
    try:
        from src.telegram_sender import send_daily_report
        sent = send_daily_report(bets_df, model_stats)
        log.info("Paso 6 OK: enviado" if sent else "Paso 6: fallo Telegram")
    except Exception as e:
        log.error(f"Error Paso 6: {e}\n{traceback.format_exc()}")

    log.info("\n" + "=" * 55)
    log.info(f"  Pipeline completado -- {datetime.now().strftime('%H:%M:%S')}  ")
    log.info("=" * 55)


if __name__ == "__main__":
    run_pipeline()