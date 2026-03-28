"""
scheduler.py
Orquestador principal del pipeline diario de FOOTBOT.

Correcciones respecto a versión anterior:
  - should_retrain() se evalúa UNA SOLA VEZ al inicio (no dos veces)
  - Imports uniformes: siempre src._XX_
  - Validación de credenciales al arranque
  - historical se carga una vez y se pasa a los módulos que lo necesitan
  - Se pasan cuotas reales al value detector

Cron en Oracle Cloud:
  0 8  * * * cd /home/ubuntu/footbot && /home/ubuntu/footbot/venv/bin/python scheduler.py >> logs/cron.log 2>&1
  0 23 * * * cd /home/ubuntu/footbot && /home/ubuntu/footbot/venv/bin/python src/_05_result_updater.py >> logs/cron_results.log 2>&1
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
    API_FOOTBALL_KEY, SUPABASE_URL, SUPABASE_KEY,
)

# ─── LOGGING ──────────────────────────────────────────────────────────────────
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


# ─── VALIDACIÓN DE CREDENCIALES ───────────────────────────────────────────────

def validate_credentials() -> list[str]:
    """
    Verifica que las variables de entorno críticas estén configuradas.
    Retorna lista de problemas encontrados (vacía = todo OK).
    """
    placeholders = {
        "API_FOOTBALL_KEY":   (API_FOOTBALL_KEY,   "TU_API_KEY_AQUI"),
        "TELEGRAM_TOKEN":     (TELEGRAM_TOKEN,     "TU_BOT_TOKEN_AQUI"),
        "TELEGRAM_CHAT_ID":   (TELEGRAM_CHAT_ID,   "TU_CHAT_ID_AQUI"),
        "SUPABASE_URL":       (SUPABASE_URL,       "TU_SUPABASE_URL_AQUI"),
        "SUPABASE_KEY":       (SUPABASE_KEY,       "TU_SUPABASE_ANON_KEY_AQUI"),
    }
    issues = []
    for name, (value, placeholder) in placeholders.items():
        if not value or value == placeholder:
            issues.append(f"  ✗ {name} no configurada")
    return issues


# ─── LÓGICA DE RE-ENTRENAMIENTO ──────────────────────────────────────────────

def should_retrain() -> bool:
    """
    Retorna True si hay que re-entrenar.
    CORRECCIÓN: Se llama UNA sola vez al inicio del pipeline para evitar
    inconsistencias si la ejecución cruza medianoche (ej: lunes a las 23:59).
    """
    no_model  = not os.path.exists(os.path.join(MODELS_DIR, "ensemble_latest.pkl"))
    is_monday = datetime.now().weekday() == 0
    return no_model or is_monday


# ─── PIPELINE ────────────────────────────────────────────────────────────────

def run_pipeline():
    log.info("═" * 55)
    log.info(f"  FOOTBOT · {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    log.info("═" * 55)

    # ── Validar credenciales antes de empezar ────────────────────────────
    issues = validate_credentials()
    if issues:
        log.warning("Credenciales no configuradas:")
        for issue in issues:
            log.warning(issue)
        # Continuar de todas formas (Supabase y Telegram son opcionales
        # para una primera ejecución de prueba)

    from src.telegram_sender import send_error_notification, send_telegram

    # ── Evaluar retrain UNA VEZ ──────────────────────────────────────────
    retrain = should_retrain()
    log.info(f"Re-entrenar modelo hoy: {retrain}")

    # ── PASO 1: Recolección de datos ─────────────────────────────────────
    log.info("\n▶ PASO 1 — Recolección de datos")
    try:
        from src._01_data_collector import (
            get_fixtures_today, download_football_data,
            download_statsbomb_data, download_elo_ratings,
        )

        fixtures = get_fixtures_today()

        if fixtures.empty:
            send_telegram(
                f"⚽ *FOOTBOT · {date.today()}*\n"
                "_No hay partidos en las ligas activas hoy._"
            )
            return

        if retrain:
            log.info("Actualizando datos históricos (es lunes o primer arranque)...")
            download_football_data()
            download_statsbomb_data()

        download_elo_ratings()
        log.info(f"✓ Paso 1: {len(fixtures)} partidos encontrados hoy")

    except Exception as e:
        log.error(f"Error Paso 1: {e}\n{traceback.format_exc()}")
        send_error_notification(f"Paso 1 falló:\n{str(e)}")
        return

    # ── PASO 2: Feature building ──────────────────────────────────────────
    log.info("\n▶ PASO 2 — Construyendo features")
    try:
        from src._02_feature_builder import (
            build_features_for_fixtures, build_training_dataset,
            load_historical_results,
        )

        # Cargar histórico UNA VEZ — se reutiliza en pasos 3, 4 y 5
        historical   = load_historical_results()
        features_df  = build_features_for_fixtures(fixtures)

        if features_df.empty:
            send_error_notification("Paso 2: DataFrame de features vacío")
            return

        log.info(f"✓ Paso 2: {len(features_df)} partidos con features")

    except Exception as e:
        log.error(f"Error Paso 2: {e}\n{traceback.format_exc()}")
        send_error_notification(f"Paso 2 falló:\n{str(e)}")
        return

    # ── PASO 3: Modelo ────────────────────────────────────────────────────
    log.info("\n▶ PASO 3 — Modelo predictivo")
    try:
        from src._03_model_engine import (
            train_and_save, load_models, predict_match,
        )

        if retrain:
            log.info("Re-entrenando modelo...")
            training_df = build_training_dataset(historical)
            dc, ensemble = train_and_save(training_df)
        else:
            dc, ensemble = load_models()

        if dc is None or ensemble is None:
            send_error_notification("Paso 3: modelos no disponibles")
            return

        all_predictions = [
            predict_match(
                row["home_team"], row["away_team"],
                row.to_dict(), dc, ensemble,
            )
            for _, row in features_df.iterrows()
        ]

        log.info(f"✓ Paso 3: {len(all_predictions)} predicciones generadas")

    except Exception as e:
        log.error(f"Error Paso 3: {e}\n{traceback.format_exc()}")
        send_error_notification(f"Paso 3 falló:\n{str(e)}")
        return

    # ── PASO 4: Value bets ────────────────────────────────────────────────
    log.info("\n▶ PASO 4 — Detección de value bets")
    bets_df = pd.DataFrame()
    try:
        from src._04_value_detector import detect_all_value_bets, summarize_bets

        # Pasar histórico para usar cuotas reales de Football-Data
        bets_df = detect_all_value_bets(features_df, all_predictions, historical)
        summary = summarize_bets(bets_df)
        log.info(f"✓ Paso 4: {summary}")

    except Exception as e:
        log.error(f"Error Paso 4: {e}\n{traceback.format_exc()}")
        send_error_notification(f"Paso 4 falló:\n{str(e)}")

    # ── PASO 5: Supabase ──────────────────────────────────────────────────
    log.info("\n▶ PASO 5 — Guardando en Supabase")
    model_stats = {}
    try:
        from src._05_result_updater import (
            init_supabase, save_predictions_to_supabase, compute_model_stats,
        )
        sb = init_supabase()
        if not bets_df.empty and sb:
            save_predictions_to_supabase(bets_df, sb)
        model_stats = compute_model_stats(sb) or {}
        log.info(f"✓ Paso 5: stats={model_stats}")

    except Exception as e:
        log.error(f"Error Paso 5: {e}\n{traceback.format_exc()}")

    # ── PASO 6: Telegram ──────────────────────────────────────────────────
    log.info("\n▶ PASO 6 — Enviando a Telegram")
    try:
        from src.telegram_sender import send_daily_report
        sent = send_daily_report(bets_df, model_stats)
        log.info("✓ Paso 6: mensaje enviado" if sent else "⚠ Paso 6: fallo Telegram")
    except Exception as e:
        log.error(f"Error Paso 6: {e}\n{traceback.format_exc()}")

    log.info("\n" + "═" * 55)
    log.info(f"  Pipeline completado · {datetime.now().strftime('%H:%M:%S')}  ")
    log.info("═" * 55)


if __name__ == "__main__":
    run_pipeline()