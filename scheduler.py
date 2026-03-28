"""
scheduler.py
Orquestador principal del pipeline diario de FOOTBOT.
Corre cada mañana via cron y coordina todos los módulos.

Configuración cron en Oracle Cloud:
  0 8  * * * /home/ubuntu/footbot/venv/bin/python /home/ubuntu/footbot/scheduler.py
  0 23 * * * /home/ubuntu/footbot/venv/bin/python /home/ubuntu/footbot/src/05_result_updater.py
"""

import os
import sys
import logging
import traceback
import pandas as pd
from datetime import date, datetime
from pathlib import Path

# Asegurar que el root del proyecto esté en el path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_RAW, DATA_PROCESSED, MODELS_DIR, LOGS_DIR
)

# ─── LOGGING ──────────────────────────────────────────────────────────────────
log_file = os.path.join(LOGS_DIR, f"footbot_{date.today()}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("footbot")


def should_retrain() -> bool:
    """Re-entrenar si es lunes o si no hay modelo guardado."""
    no_model = not os.path.exists(
        os.path.join(MODELS_DIR, "ensemble_latest.pkl")
    )
    is_monday = datetime.now().weekday() == 0
    return no_model or is_monday


def run_pipeline():
    """Pipeline completo del día."""
    log.info("═" * 55)
    log.info(f"  FOOTBOT · {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    log.info("═" * 55)
    
    from src.telegram_sender import send_error_notification
    
    # ── PASO 1: Recolección de datos ─────────────────────────────────
    log.info("\n▶ PASO 1 — Recolección de datos")
    try:
        from src._01_data_collector import (
            get_fixtures_today, download_football_data,
            download_statsbomb_data, download_elo_ratings
        )
        
        fixtures = get_fixtures_today()
        
        if fixtures.empty:
            log.info("No hay partidos hoy. Enviando notificación...")
            from src.telegram_sender import send_telegram
            send_telegram(
                f"⚽ *FOOTBOT · {date.today()}*\n"
                "_No hay partidos en las ligas activas hoy._"
            )
            return
        
        # Descargar datos históricos (solo si es lunes o primer arranque)
        if should_retrain():
            log.info("Actualizando datos históricos...")
            download_football_data()
            download_statsbomb_data()
        
        download_elo_ratings()  # ELO siempre actualizado
        log.info(f"Paso 1 completado: {len(fixtures)} partidos hoy")
        
    except Exception as e:
        log.error(f"Error en recolección de datos: {e}")
        log.error(traceback.format_exc())
        send_error_notification(f"Paso 1 falló:\n{str(e)}")
        return
    
    # ── PASO 2: Feature building ──────────────────────────────────────
    log.info("\n▶ PASO 2 — Construyendo features")
    try:
        from src._02_feature_builder import (
            build_features_for_fixtures,
            build_training_dataset,
            load_historical_results
        )
        
        features_df = build_features_for_fixtures(fixtures)
        
        if features_df.empty:
            log.error("No se pudieron calcular features. Abortando.")
            send_error_notification("Paso 2 falló: DataFrame de features vacío")
            return
        
        log.info(f"Paso 2 completado: {len(features_df)} partidos con features")
        
    except Exception as e:
        log.error(f"Error en feature building: {e}")
        log.error(traceback.format_exc())
        send_error_notification(f"Paso 2 falló:\n{str(e)}")
        return
    
    # ── PASO 3: Entrenamiento (si corresponde) + Predicción ──────────
    log.info("\n▶ PASO 3 — Modelo predictivo")
    try:
        from src._03_model_engine import (
            train_and_save, load_models, predict_match
        )
        
        if should_retrain():
            log.info("Re-entrenando modelo (lunes o primer arranque)...")
            historical = load_historical_results()
            training_df = build_training_dataset(historical)
            dc, ensemble = train_and_save(training_df)
        else:
            log.info("Cargando modelo existente...")
            dc, ensemble = load_models()
        
        if dc is None or ensemble is None:
            log.error("Modelos no disponibles. Abortando.")
            send_error_notification("Paso 3: modelos no disponibles")
            return
        
        # Generar predicciones para cada partido
        all_predictions = []
        for _, fixture in features_df.iterrows():
            pred = predict_match(
                fixture["home_team"],
                fixture["away_team"],
                fixture.to_dict(),
                dc, ensemble
            )
            all_predictions.append(pred)
        
        log.info(f"Paso 3 completado: {len(all_predictions)} predicciones generadas")
        
    except Exception as e:
        log.error(f"Error en modelo: {e}")
        log.error(traceback.format_exc())
        send_error_notification(f"Paso 3 falló:\n{str(e)}")
        return
    
    # ── PASO 4: Detección de value bets ──────────────────────────────
    log.info("\n▶ PASO 4 — Detección de value bets")
    try:
        from src._04_value_detector import (
            detect_all_value_bets, summarize_bets
        )
        
        bets_df = detect_all_value_bets(features_df, all_predictions)
        summary = summarize_bets(bets_df)
        
        log.info(f"Value bets encontradas: {summary}")
        
    except Exception as e:
        log.error(f"Error en detector: {e}")
        log.error(traceback.format_exc())
        bets_df = pd.DataFrame()
        send_error_notification(f"Paso 4 falló:\n{str(e)}")
    
    # ── PASO 5: Guardar en Supabase ───────────────────────────────────
    log.info("\n▶ PASO 5 — Guardando en Supabase")
    try:
        from src._05_result_updater import (
            init_supabase, save_predictions_to_supabase,
            compute_model_stats
        )
        
        sb = init_supabase()
        
        if not bets_df.empty and sb:
            save_predictions_to_supabase(bets_df, sb)
        
        model_stats = compute_model_stats(sb)
        log.info(f"Stats del modelo: {model_stats}")
        
    except Exception as e:
        log.error(f"Error Supabase: {e}")
        log.error(traceback.format_exc())
        model_stats = {}
    
    # ── PASO 6: Enviar a Telegram ─────────────────────────────────────
    log.info("\n▶ PASO 6 — Enviando a Telegram")
    try:
        from src.telegram_sender import send_daily_report
        
        sent = send_daily_report(bets_df, model_stats)
        if sent:
            log.info("Mensaje enviado a Telegram exitosamente")
        else:
            log.warning("Fallo al enviar a Telegram")
            
    except Exception as e:
        log.error(f"Error Telegram: {e}")
        log.error(traceback.format_exc())
    
    log.info("\n" + "═" * 55)
    log.info(f"  Pipeline completado · {datetime.now().strftime('%H:%M:%S')}  ")
    log.info("═" * 55)


if __name__ == "__main__":
    run_pipeline()
