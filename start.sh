#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTBOT — start.sh
#  Orquestador unificado: clubes + selecciones en un solo punto de entrada.
#
#  Uso:
#    ./start.sh predict   # 8:00 AM  — predicciones del día
#    ./start.sh results   # 11:00 PM — cierre de resultados
#    ./start.sh live      # 2:00 PM  — live polling selecciones
#    ./start.sh all       # manual   — predict + live + results completo
#
#  Crontab:
#    0  8 * * * /home/ubuntu/footbot/start.sh predict >> /home/ubuntu/footbot/logs/cron.log 2>&1
#    0 23 * * * /home/ubuntu/footbot/start.sh results >> /home/ubuntu/footbot/logs/cron.log 2>&1
#    0 14 * * * /home/ubuntu/footbot/start.sh live    >> /home/ubuntu/footbot/logs/cron.log 2>&1
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── CONFIG ───────────────────────────────────────────────────────────────────
FOOTBOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${FOOTBOT_DIR}/venv/bin/python"
LOGS_DIR="${FOOTBOT_DIR}/logs"
MODE="${1:-predict}"

RETRY_WAIT=60        # segundos entre reintento
ESPN_COOLDOWN=30     # segundos entre scheduler.py y scheduler_nacional.py
                     # para no saturar la ESPN API

# ─── SETUP ────────────────────────────────────────────────────────────────────
mkdir -p "${LOGS_DIR}"
cd "${FOOTBOT_DIR}"

# Activar entorno virtual
# shellcheck disable=SC1091
source "${FOOTBOT_DIR}/venv/bin/activate"

# ─── LOGGING ──────────────────────────────────────────────────────────────────
LOG_DATE="$(date '+%Y-%m-%d')"
LOG_FILE="${LOGS_DIR}/footbot_${LOG_DATE}.log"

log() {
    local level="$1"; shift
    local msg="$*"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "${ts} [${level}] ${msg}" | tee -a "${LOG_FILE}"
}

log INFO "════════════════════════════════════════"
log INFO "  FOOTBOT start.sh — modo: ${MODE}"
log INFO "  $(date '+%Y-%m-%d %H:%M:%S')"
log INFO "════════════════════════════════════════"

# ─── NOTIFICACIÓN TELEGRAM (bash puro, sin depender de Python) ────────────────
telegram_notify() {
    local msg="$1"
    # Lee token y chat_id del .env si existe
    if [ -f "${FOOTBOT_DIR}/.env" ]; then
        # shellcheck disable=SC1091
        set -a; source "${FOOTBOT_DIR}/.env"; set +a
    fi
    if [ -z "${TELEGRAM_TOKEN:-}" ] || [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
        log WARN "Telegram no configurado — skipping notificación bash"
        return 0
    fi
    curl -s -X POST \
        "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        -d "text=${msg}" \
        -d "parse_mode=Markdown" \
        > /dev/null 2>&1 || true   # nunca rompe el flujo
}

# ─── RUNNER CON REINTENTO ─────────────────────────────────────────────────────
# run_step <nombre> <script> [args...]
#   1. Ejecuta el script.
#   2. Si falla, espera RETRY_WAIT segundos y reintenta una vez.
#   3. Si vuelve a fallar, notifica por Telegram y retorna código de error
#      (el llamador decide si continuar o no).
run_step() {
    local name="$1"; shift
    local cmd=("$@")

    log INFO "▶ Iniciando: ${name}"

    if "${PYTHON}" "${cmd[@]}" >> "${LOG_FILE}" 2>&1; then
        log INFO "✓ OK: ${name}"
        return 0
    fi

    log WARN "✗ Falló: ${name} — reintentando en ${RETRY_WAIT}s..."
    sleep "${RETRY_WAIT}"

    if "${PYTHON}" "${cmd[@]}" >> "${LOG_FILE}" 2>&1; then
        log INFO "✓ OK (reintento): ${name}"
        return 0
    fi

    log ERROR "✗ FALLO DEFINITIVO: ${name}"
    telegram_notify "⚠️ *FOOTBOT ERROR*%0A\`${name}\` falló después de 2 intentos.%0AVer: \`${LOG_FILE}\`"
    return 1
}

# ─── LIVE GUARD (evita instancias duplicadas) ─────────────────────────────────
live_guard() {
    if pgrep -f "scheduler_nacional.*live" > /dev/null 2>&1; then
        log WARN "Ya hay un proceso live corriendo — abortando este."
        exit 0
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
#  MODOS
# ══════════════════════════════════════════════════════════════════════════════

mode_predict() {
    log INFO "━━━ MODO: PREDICT ━━━"

    # 1. Clubes — pipeline principal
    #    Si falla definitivamente, notifica pero igual intenta selecciones
    #    porque son pipelines independientes.
    run_step "Clubes — predicciones" scheduler.py || {
        log WARN "Pipeline de clubes falló — continuando con selecciones."
    }

    # Cooldown para no saturar ESPN API (comparte endpoints con scheduler.py)
    log INFO "Cooldown ESPN: ${ESPN_COOLDOWN}s..."
    sleep "${ESPN_COOLDOWN}"

    # 2. Selecciones — pipeline nacional
    run_step "Selecciones — predicciones" scheduler_nacional.py --mode predict || {
        log WARN "Pipeline de selecciones falló."
    }

    log INFO "━━━ PREDICT completado ━━━"
}

mode_results() {
    log INFO "━━━ MODO: RESULTS ━━━"

    # 1. Cerrar resultados de clubes (football-data.org → API-Football fallback)
    run_step "Clubes — resultados" src/_05_result_updater.py || {
        log WARN "Cierre de clubes falló — continuando con selecciones."
    }

    sleep 30

    # 2. Cerrar resultados de selecciones (ESPN)
    run_step "Selecciones — resultados" scheduler_nacional.py --mode results || {
        log WARN "Cierre de selecciones falló."
    }

    log INFO "━━━ RESULTS completado ━━━"
}

mode_live() {
    log INFO "━━━ MODO: LIVE ━━━"

    # Guard: una sola instancia activa a la vez
    live_guard

    # El live polling es bloqueante — corre hasta que terminan los partidos.
    # No usamos run_step porque no tiene sentido reintentar un proceso largo;
    # si falla, simplemente se notifica.
    log INFO "▶ Iniciando live polling de selecciones..."
    if ! "${PYTHON}" scheduler_nacional.py --mode live >> "${LOG_FILE}" 2>&1; then
        log ERROR "Live polling terminó con error."
        telegram_notify "⚠️ *FOOTBOT LIVE ERROR*%0ALive polling falló o fue interrumpido.%0AVer: \`${LOG_FILE}\`"
    else
        log INFO "✓ Live polling finalizado correctamente."
    fi

    log INFO "━━━ LIVE completado ━━━"
}

mode_all() {
    log INFO "━━━ MODO: ALL (predict + live + results) ━━━"
    mode_predict
    mode_live
    mode_results
}

# ══════════════════════════════════════════════════════════════════════════════
#  DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

case "${MODE}" in
    predict) mode_predict ;;
    results) mode_results ;;
    live)    mode_live    ;;
    all)     mode_all     ;;
    *)
        log ERROR "Modo desconocido: '${MODE}'. Usa: predict | results | live | all"
        exit 1
        ;;
esac

log INFO "════════════════════════════════════════"
log INFO "  start.sh finalizado — $(date '+%H:%M:%S')"
log INFO "════════════════════════════════════════"