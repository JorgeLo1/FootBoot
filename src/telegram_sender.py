"""
telegram_sender.py
Formatea y envía el reporte diario a Telegram.
Corrección: valida el token antes de intentar enviar.
"""

import sys
import logging
import requests
import pandas as pd
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger(__name__)

DIAS_ES = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes",
    "Saturday": "Sábado", "Sunday": "Domingo",
}
MESES_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}

PLACEHOLDER_TOKENS = {"TU_BOT_TOKEN_AQUI", "TU_CHAT_ID_AQUI", "", None}


def _credentials_ok() -> bool:
    """Valida que el token y chat_id sean reales antes de hacer requests."""
    if TELEGRAM_TOKEN in PLACEHOLDER_TOKENS:
        log.warning("TELEGRAM_TOKEN no configurado. Skipping notificaciones.")
        return False
    if TELEGRAM_CHAT_ID in PLACEHOLDER_TOKENS:
        log.warning("TELEGRAM_CHAT_ID no configurado. Skipping notificaciones.")
        return False
    return True


def format_date_es(d: date) -> str:
    dia_es = DIAS_ES.get(d.strftime("%A"), d.strftime("%A"))
    mes_es = MESES_ES.get(d.month, "")
    return f"{dia_es} {d.day} {mes_es}"


def format_message(bets_df: pd.DataFrame, model_stats: dict = None) -> str:
    today = date.today()
    fecha = format_date_es(today)
    total = len(bets_df) if not bets_df.empty else 0

    alta  = bets_df[bets_df["confidence"] == "alta"]  if not bets_df.empty else pd.DataFrame()
    media = bets_df[bets_df["confidence"] == "media"] if not bets_df.empty else pd.DataFrame()

    lines = [
        f"⚽ *FOOTBOT · {fecha}*",
        f"_{total} apuesta{'s' if total != 1 else ''} encontrada{'s' if total != 1 else ''}_",
        "",
    ]

    def render_bet(bet: pd.Series, show_goals: bool = True) -> list:
        odds_tag = "" if bet.get("odds_source") == "real" else " _(cuota estimada)_"
        result = [
            f"*{bet['home_team']} vs {bet['away_team']}*",
            f"🏆 {bet.get('league', '')}",
            f"📌 {bet['market_display']}",
            (f"📊 Prob: `{bet['model_prob_pct']}`  "
             f"Cuota: `{bet['reference_odds']}`{odds_tag}  "
             f"Edge: `+{bet['edge_pct']}%`"),
            f"💰 Kelly: `{bet['kelly_pct']}%` bankroll",
        ]
        if show_goals and bet.get("exp_home_goals") and bet.get("exp_away_goals"):
            result.append(
                f"📈 Goles esperados: "
                f"{bet['exp_home_goals']:.1f} – {bet['exp_away_goals']:.1f}"
            )
        if bet.get("explanation"):
            result.append(f"💡 _{bet['explanation']}_")
        result.append("")
        return result

    if not alta.empty:
        lines += ["🟢 *ALTA CONFIANZA*", "─" * 28]
        for _, bet in alta.iterrows():
            lines += render_bet(bet, show_goals=True)

    if not media.empty:
        lines += ["🟡 *MEDIA CONFIANZA*", "─" * 28]
        for _, bet in media.iterrows():
            lines += render_bet(bet, show_goals=False)

    if total == 0:
        lines += [
            "❌ *Sin value bets hoy*",
            "_No se encontró edge suficiente en los partidos disponibles._",
            "",
        ]

    lines.append("─" * 28)
    if model_stats and model_stats.get("total", 0) > 10:
        tasa = model_stats.get("tasa_pct", 0)
        roi  = model_stats.get("roi_pct", 0)
        tot  = model_stats.get("total", 0)
        lines.append(
            f"📉 *Rendimiento* ({tot} apuestas)  "
            f"Acierto: `{tasa}%`  ROI: `{roi:+.1f}%`"
        )
        if "tasa_alta_pct" in model_stats:
            lines.append(
                f"🟢 Alta: `{model_stats['tasa_alta_pct']}%`  "
                f"ROI `{model_stats.get('roi_alta_pct', 0):+.1f}%`"
            )
        if "tasa_media_pct" in model_stats:
            lines.append(
                f"🟡 Media: `{model_stats['tasa_media_pct']}%`  "
                f"ROI `{model_stats.get('roi_media_pct', 0):+.1f}%`"
            )
    else:
        lines.append("_Estadísticas disponibles tras 10+ apuestas cerradas_")

    lines += [
        "",
        "⚠️ _Modelo estadístico experimental. Apuesta con responsabilidad._",
    ]
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    """Envía un mensaje a Telegram. Retorna False si las credenciales no están listas."""
    if not _credentials_ok():
        log.info(f"[TELEGRAM SIMULADO]\n{message[:300]}...")
        return False

    url    = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]

    success = True
    for i, chunk in enumerate(chunks):
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id":                  TELEGRAM_CHAT_ID,
                    "text":                     chunk,
                    "parse_mode":               "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                log.error(f"Telegram error: {data.get('description', data)}")
                success = False
        except Exception as e:
            log.error(f"Error enviando a Telegram: {e}")
            success = False

    return success


def send_daily_report(bets_df: pd.DataFrame, model_stats: dict = None) -> bool:
    message = format_message(bets_df, model_stats)
    log.info(f"Reporte Telegram: {len(message)} chars")
    return send_telegram(message)


def send_error_notification(error_msg: str):
    send_telegram(f"⚠️ *FOOTBOT ERROR*\n```{error_msg[:500]}```")