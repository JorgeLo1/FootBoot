"""
telegram_sender.py
Formatea y envía el mensaje diario de apuestas a Telegram.
Incluye Alta y Media confianza con toda la información relevante.
"""

import sys
import logging
import requests
import pandas as pd
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DIAS_ES = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes",
    "Saturday": "Sábado", "Sunday": "Domingo"
}

MESES_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}


def format_date_es(d: date) -> str:
    dia_en = d.strftime("%A")
    dia_es = DIAS_ES.get(dia_en, dia_en)
    mes_es = MESES_ES.get(d.month, "")
    return f"{dia_es} {d.day} {mes_es}"


def format_message(bets_df: pd.DataFrame, model_stats: dict = None) -> str:
    """
    Construye el mensaje completo de Telegram con formato claro y legible.
    """
    today = date.today()
    fecha = format_date_es(today)
    
    alta  = bets_df[bets_df["confidence"] == "alta"]  if not bets_df.empty else pd.DataFrame()
    media = bets_df[bets_df["confidence"] == "media"] if not bets_df.empty else pd.DataFrame()
    
    total = len(bets_df) if not bets_df.empty else 0
    
    lines = []
    lines.append(f"⚽ *FOOTBOT · {fecha}*")
    lines.append(f"_{total} apuesta{'s' if total != 1 else ''} encontrada{'s' if total != 1 else ''} hoy_")
    lines.append("")
    
    # ── ALTA CONFIANZA ──────────────────────────────────────────────
    if not alta.empty:
        lines.append("🟢 *ALTA CONFIANZA*")
        lines.append("─" * 30)
        
        for _, bet in alta.iterrows():
            lines.append(f"*{bet['home_team']} vs {bet['away_team']}*")
            lines.append(f"🏆 {bet.get('league','')}")
            lines.append(f"📌 {bet['market_display']}")
            lines.append(
                f"📊 Prob: `{bet['model_prob_pct']}`  |  "
                f"Cuota: `{bet['reference_odds']}`  |  "
                f"Edge: `+{bet['edge_pct']}%`"
            )
            lines.append(f"💰 Kelly: `{bet['kelly_pct']}%` del bankroll")
            
            if bet.get("exp_home_goals") and bet.get("exp_away_goals"):
                lines.append(
                    f"📈 Goles esperados: "
                    f"{bet['exp_home_goals']:.1f} - {bet['exp_away_goals']:.1f}"
                )
            
            if bet.get("explanation"):
                lines.append(f"💡 _{bet['explanation']}_")
            
            lines.append("")
    
    # ── MEDIA CONFIANZA ─────────────────────────────────────────────
    if not media.empty:
        lines.append("🟡 *MEDIA CONFIANZA*")
        lines.append("─" * 30)
        
        for _, bet in media.iterrows():
            lines.append(f"*{bet['home_team']} vs {bet['away_team']}*")
            lines.append(f"🏆 {bet.get('league','')}")
            lines.append(f"📌 {bet['market_display']}")
            lines.append(
                f"📊 Prob: `{bet['model_prob_pct']}`  |  "
                f"Cuota: `{bet['reference_odds']}`  |  "
                f"Edge: `+{bet['edge_pct']}%`"
            )
            lines.append(f"💰 Kelly: `{bet['kelly_pct']}%` del bankroll")
            
            if bet.get("explanation"):
                lines.append(f"💡 _{bet['explanation']}_")
            
            lines.append("")
    
    # ── SIN APUESTAS ────────────────────────────────────────────────
    if bets_df.empty or total == 0:
        lines.append("❌ *Sin value bets hoy*")
        lines.append("_El modelo no encontró suficiente edge en los partidos disponibles._")
        lines.append("")
    
    # ── ESTADÍSTICAS DEL MODELO ─────────────────────────────────────
    lines.append("─" * 30)
    if model_stats and model_stats.get("total", 0) > 10:
        tasa = model_stats.get("tasa_pct", 0)
        roi  = model_stats.get("roi_pct", 0)
        tot  = model_stats.get("total", 0)
        lines.append(
            f"📉 *Rendimiento acumulado* ({tot} apuestas)\n"
            f"Acierto: `{tasa}%`  |  ROI: `{roi:+.1f}%`"
        )
        
        if "tasa_alta_pct" in model_stats:
            lines.append(
                f"🟢 Alta: `{model_stats['tasa_alta_pct']}%` acierto  |  "
                f"ROI `{model_stats.get('roi_alta_pct',0):+.1f}%`"
            )
        if "tasa_media_pct" in model_stats:
            lines.append(
                f"🟡 Media: `{model_stats['tasa_media_pct']}%` acierto  |  "
                f"ROI `{model_stats.get('roi_media_pct',0):+.1f}%`"
            )
    else:
        lines.append("_Estadísticas disponibles tras 10+ apuestas cerradas_")
    
    lines.append("")
    lines.append("⚠️ _Apuesta con responsabilidad. Esto es un modelo estadístico, no una garantía._")
    
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    """Envía el mensaje formateado al chat de Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    
    # Telegram tiene límite de 4096 caracteres por mensaje
    chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
    
    success = True
    for i, chunk in enumerate(chunks):
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id":    TELEGRAM_CHAT_ID,
                    "text":       chunk,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                log.error(f"Error Telegram: {data}")
                success = False
            else:
                log.info(f"Mensaje {i+1}/{len(chunks)} enviado a Telegram")
        except Exception as e:
            log.error(f"Error enviando a Telegram: {e}")
            success = False
    
    return success


def send_daily_report(bets_df: pd.DataFrame, model_stats: dict = None) -> bool:
    """Función principal: formatea y envía el reporte diario."""
    message = format_message(bets_df, model_stats)
    log.info(f"Mensaje Telegram preparado ({len(message)} chars)")
    return send_telegram(message)


def send_error_notification(error_msg: str):
    """Envía notificación de error al canal."""
    msg = f"⚠️ *FOOTBOT ERROR*\n```{error_msg[:500]}```"
    send_telegram(msg)


if __name__ == "__main__":
    # Test con datos simulados
    import pandas as pd
    
    test_bets = pd.DataFrame([
        {
            "home_team": "Arsenal", "away_team": "Chelsea",
            "league": "Premier League", "market_display": "Ambos marcan — SÍ",
            "model_prob_pct": "68.4%", "reference_odds": 1.82,
            "edge_pct": 24.3, "kelly_pct": 2.8, "confidence": "alta",
            "exp_home_goals": 1.9, "exp_away_goals": 1.3,
            "explanation": "xG ofensivo Arsenal 1.9 | BTTS en 72% H2H | xG total esperado 3.2",
        },
        {
            "home_team": "Bayern Munich", "away_team": "Dortmund",
            "league": "Bundesliga", "market_display": "Doble oportunidad Bayern/Empate",
            "model_prob_pct": "79.1%", "reference_odds": 1.38,
            "edge_pct": 9.2, "kelly_pct": 1.1, "confidence": "media",
            "exp_home_goals": 2.1, "exp_away_goals": 1.2,
            "explanation": "ELO ventaja 210 pts para local | forma Bayern 2.4 pts/partido",
        },
    ])
    
    test_stats = {
        "total": 47, "tasa_pct": 61.7, "roi_pct": 8.3,
        "tasa_alta_pct": 64.5, "roi_alta_pct": 12.1,
        "tasa_media_pct": 58.3, "roi_media_pct": 4.8,
    }
    
    msg = format_message(test_bets, test_stats)
    print(msg)
    print(f"\n[{len(msg)} caracteres]")
