"""
telegram_sender.py — v5
Formatea y envía el reporte diario a Telegram.
Soporte completo para mercados expandidos — agrupados por partido y categoría.
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

# Grupos de mercados para el mensaje — orden de aparición
MARKET_GROUPS = {
    "resultado": ["home_win", "draw", "away_win",
                  "double_1x", "double_x2", "double_12"],
    "goles":     ["over15", "under15", "over25", "under25",
                  "over35", "under35", "over45"],
    "btts":      ["btts_si", "btts_no"],
    "por_equipo":["home_over05", "home_under05", "home_over15", "home_under15",
                  "away_over05", "away_under05", "away_over15", "away_under15"],
    "exactos":   ["exact_0", "exact_1", "exact_2", "exact_3", "exact_4plus"],
    "combinadas":["home_and_btts", "draw_and_btts", "away_and_btts"],
    "handicap":  ["ah_home_minus05", "ah_away_minus05",
                  "ah_home_plus05",  "ah_away_plus05",
                  "ah_home_minus1",  "ah_away_minus1"],
}

GROUP_EMOJI = {
    "resultado":  "🏆",
    "goles":      "⚽",
    "btts":       "🔵",
    "por_equipo": "🎯",
    "exactos":    "🔢",
    "combinadas": "🔗",
    "handicap":   "⚖️",
}

GROUP_LABELS = {
    "resultado":  "Resultado",
    "goles":      "Totales de goles",
    "btts":       "Ambos marcan",
    "por_equipo": "Goles por equipo",
    "exactos":    "Goles exactos",
    "combinadas": "Combinadas",
    "handicap":   "Asian Handicap",
}

# Etiqueta de fuente de cuota para el usuario
_ODDS_SOURCE_LABEL = {
    "espn_live":      "",                      # cuota real — no necesita nota
    "exact_match":    "",                      # cuota real — no necesita nota
    "contextual_avg": " _(hist.)_",            # real pero no del partido exacto
    "fd_historical":  " _(hist.)_",
    "model_implied":  " _(est.)_",             # derivada del modelo
}


def _credentials_ok() -> bool:
    if TELEGRAM_TOKEN in PLACEHOLDER_TOKENS:
        log.warning("TELEGRAM_TOKEN no configurado.")
        return False
    if TELEGRAM_CHAT_ID in PLACEHOLDER_TOKENS:
        log.warning("TELEGRAM_CHAT_ID no configurado.")
        return False
    return True


def format_date_es(d: date) -> str:
    dia_es = DIAS_ES.get(d.strftime("%A"), d.strftime("%A"))
    mes_es = MESES_ES.get(d.month, "")
    return f"{dia_es} {d.day} {mes_es}"


def _render_bet_line(bet: pd.Series) -> str:
    """
    Renderiza una apuesta en una sola línea compacta + línea de detalle.
    Formato:
      📌 *Mercado*  Prob: 63.2%  Cuota: 1.85 _(hist.)_  Edge: +8.4%
         Kelly: 1.2% bankroll | explicación
    """
    source_tag = _ODDS_SOURCE_LABEL.get(bet.get("odds_source", ""), " _(est.)_")

    line1 = (
        f"  📌 *{bet['market_display']}*\n"
        f"     Prob `{bet['model_prob_pct']}` · "
        f"Cuota `{bet['reference_odds']}`{source_tag} · "
        f"Edge `+{bet['edge_pct']}%`"
    )
    line2_parts = [f"Kelly `{bet['kelly_pct']}%`"]
    if bet.get("explanation"):
        line2_parts.append(bet["explanation"])
    line2 = "     " + " | ".join(line2_parts)

    return f"{line1}\n{line2}"


def _group_bets_by_match(bets_df: pd.DataFrame) -> list[dict]:
    """Agrupa apuestas por partido manteniendo el orden alta → media."""
    seen   = {}
    result = []
    for _, bet in bets_df.iterrows():
        key = (bet["home_team"], bet["away_team"])
        if key not in seen:
            seen[key] = len(result)
            result.append({
                "home":     bet["home_team"],
                "away":     bet["away_team"],
                "league":   bet.get("league", ""),
                "mu":       bet.get("exp_home_goals", 0),
                "lam":      bet.get("exp_away_goals", 0),
                "bets":     [],
            })
        result[seen[key]]["bets"].append(bet)
    return result


def _render_match_block(match: dict) -> list[str]:
    """
    Bloque de un partido con encabezado + apuestas agrupadas por categoría.
    """
    home  = match["home"]
    away  = match["away"]
    mu    = match["mu"]
    lam   = match["lam"]

    lines = [
        f"*{home} vs {away}*",
        f"🏆 {match['league']}",
        f"📈 Esperados: `{mu:.1f}` – `{lam:.1f}` (total `{mu+lam:.1f}`)",
        "",
    ]

    # Agrupar apuestas por categoría
    by_group: dict[str, list] = {}
    for bet in match["bets"]:
        placed = False
        for group, markets in MARKET_GROUPS.items():
            if bet["market"] in markets:
                by_group.setdefault(group, []).append(bet)
                placed = True
                break
        if not placed:
            by_group.setdefault("otros", []).append(bet)

    for group in list(MARKET_GROUPS.keys()) + ["otros"]:
        group_bets = by_group.get(group)
        if not group_bets:
            continue
        emoji = GROUP_EMOJI.get(group, "•")
        label = GROUP_LABELS.get(group, group.title())
        lines.append(f"_{emoji} {label}_")
        for bet in group_bets:
            lines.append(_render_bet_line(pd.Series(bet)))
        lines.append("")

    return lines


def format_message(bets_df: pd.DataFrame, model_stats: dict = None) -> str:
    today = date.today()
    fecha = format_date_es(today)
    total = len(bets_df) if not bets_df.empty else 0

    alta  = bets_df[bets_df["confidence"] == "alta"]  if not bets_df.empty else pd.DataFrame()
    media = bets_df[bets_df["confidence"] == "media"] if not bets_df.empty else pd.DataFrame()

    # Contar cuántas son con cuotas reales
    n_real = 0
    if not bets_df.empty and "odds_are_real" in bets_df.columns:
        n_real = int(bets_df["odds_are_real"].sum())

    subtitle_parts = [f"_{total} apuesta{'s' if total != 1 else ''}_"]
    if total > 0 and n_real < total:
        n_est = total - n_real
        subtitle_parts.append(f"_({n_est} con cuota estimada)_")

    lines = [
        f"⚽ *FOOTBOT · {fecha}*",
        " ".join(subtitle_parts),
        "",
    ]

    if not alta.empty:
        lines += ["🟢 *ALTA CONFIANZA*", "─" * 28, ""]
        for match in _group_bets_by_match(alta):
            lines += _render_match_block(match)
            lines.append("· · ·")
        lines.append("")

    if not media.empty:
        lines += ["🟡 *MEDIA CONFIANZA*", "─" * 28, ""]
        for match in _group_bets_by_match(media):
            lines += _render_match_block(match)
            lines.append("· · ·")
        lines.append("")

    if total == 0:
        lines += [
            "❌ *Sin value bets hoy*",
            "_No se encontró edge suficiente en los partidos disponibles._",
            "",
        ]

    lines.append("─" * 28)

    if model_stats and model_stats.get("total", 0) > 10:
        tasa = model_stats.get("tasa_pct", 0)
        roi  = model_stats.get("roi_pct",  0)
        tot  = model_stats.get("total",    0)
        lines.append(
            f"📊 *Rendimiento* ({tot} apuestas): "
            f"Acierto `{tasa}%` · ROI `{roi:+.1f}%`"
        )
        if "tasa_alta_pct" in model_stats:
            lines.append(
                f"  🟢 Alta: `{model_stats['tasa_alta_pct']}%` · "
                f"ROI `{model_stats.get('roi_alta_pct', 0):+.1f}%`"
            )
        if "tasa_media_pct" in model_stats:
            lines.append(
                f"  🟡 Media: `{model_stats['tasa_media_pct']}%` · "
                f"ROI `{model_stats.get('roi_media_pct', 0):+.1f}%`"
            )
        # Top mercados si están disponibles
        top_mkts = model_stats.get("top_mercados")
        if top_mkts:
            mkt_str = " · ".join(f"{m}(`{v}`)" for m, v in list(top_mkts.items())[:3])
            lines.append(f"  📌 Top mercados: {mkt_str}")
    else:
        lines.append("_Stats disponibles tras 10+ apuestas cerradas_")

    lines += [
        "",
        "⚠️ _Modelo estadístico experimental. Apuesta con responsabilidad._",
    ]
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    if not _credentials_ok():
        log.info(f"[TELEGRAM SIMULADO]\n{message[:600]}...")
        return False

    url    = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # Telegram límite: 4096 chars por mensaje
    chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
    success = True
    for chunk in chunks:
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
            if not resp.json().get("ok"):
                log.error(f"Telegram error: {resp.json().get('description')}")
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