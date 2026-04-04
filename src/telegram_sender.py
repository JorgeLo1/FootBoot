"""
telegram_sender.py — v7
Formatea y envía el reporte diario a Telegram.

CAMBIOS v7:
  - Header con resumen numérico compacto (alta/media/baja de un vistazo).
  - Separador de partido más limpio — sin "· · ·", usa línea en blanco + divisor fino.
  - Bloque de apuesta rediseñado: una línea por dato, mejor alineación visual.
  - Badge de nivel inline en cada apuesta (para chunks largos que pierden el header).
  - Cuotas reales marcadas con ✅, estimadas con 〰️ — sin nota parentética.
  - Sección de stats más densa: tabla compacta con pipes.
  - Footer de advertencia reducido a una sola línea.
  - Función send_simple_alert() para notificaciones rápidas sin DataFrame.
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

MARKET_GROUPS = {
    "resultado":  ["home_win", "draw", "away_win",
                   "double_1x", "double_x2", "double_12"],
    "goles":      ["over15", "under15", "over25", "under25",
                   "over35", "under35", "over45"],
    "btts":       ["btts_si", "btts_no"],
    "por_equipo": ["home_over05", "home_under05", "home_over15", "home_under15",
                   "away_over05", "away_under05", "away_over15", "away_under15"],
    "exactos":    ["exact_0", "exact_1", "exact_2", "exact_3", "exact_4plus"],
    "combinadas": ["home_and_btts", "draw_and_btts", "away_and_btts"],
    "handicap":   ["ah_home_minus05", "ah_away_minus05",
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
    "goles":      "Totales",
    "btts":       "Ambos marcan",
    "por_equipo": "Por equipo",
    "exactos":    "Exactos",
    "combinadas": "Combinadas",
    "handicap":   "Handicap",
}

# ✅ = cuota real del mercado   〰️ = cuota estimada/histórica
_ODDS_SOURCE_ICON = {
    "espn_live":      "✅",
    "exact_match":    "✅",
    "contextual_avg": "〰️",
    "fd_historical":  "〰️",
    "model_implied":  "〰️",
}

_CONFIDENCE_BADGE = {
    "alta":  "🟢",
    "media": "🟡",
    "baja":  "🔵",
}

_SEP_THICK = "━" * 28
_SEP_THIN  = "┄" * 24


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


def _render_bet_line(bet: pd.Series, is_baja: bool = False) -> str:
    """
    Bloque de una apuesta — diseño vertical compacto:

    📌 *Victoria Local*
    ├ Prob 81.3%  ·  Cuota ✅ 1.34  ·  Edge +8.9%
    └ Kelly 1.2%  |  forma local 2.2pts/PJ
    """
    source_icon = _ODDS_SOURCE_ICON.get(bet.get("odds_source", ""), "〰️")
    prob_str  = bet.get("model_prob_pct", "—")
    odds_str  = bet.get("reference_odds", "—")
    edge_str  = bet.get("edge_pct", "—")
    kelly_str = bet.get("kelly_pct", "—")

    line_title = f"  📌 *{bet['market_display']}*"

    line_stats = (
        f"  ├ Prob `{prob_str}%`  ·  "
        f"Cuota {source_icon} `{odds_str}`  ·  "
        f"Edge `+{edge_str}%`"
    )

    detail_parts = [f"Kelly `{kelly_str}%`"]
    if bet.get("explanation"):
        detail_parts.append(bet["explanation"])
    if is_baja:
        detail_parts.append("⚠️ stake reducido")

    line_detail = "  └ " + "  |  ".join(detail_parts)

    return f"{line_title}\n{line_stats}\n{line_detail}"


def _group_bets_by_match(bets_df: pd.DataFrame) -> list[dict]:
    seen   = {}
    result = []
    for _, bet in bets_df.iterrows():
        key = (bet["home_team"], bet["away_team"])
        if key not in seen:
            seen[key] = len(result)
            result.append({
                "home":   bet["home_team"],
                "away":   bet["away_team"],
                "league": bet.get("league", ""),
                "mu":     bet.get("exp_home_goals", 0),
                "lam":    bet.get("exp_away_goals", 0),
                "bets":   [],
            })
        result[seen[key]]["bets"].append(bet)
    return result


def _render_match_block(match: dict, is_baja: bool = False) -> list[str]:
    """
    Bloque de un partido.

    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
    🏟 *Millonarios vs Nacional*
    🏆 Liga BetPlay  ·  📈 2.5 – 0.8 (3.3 goles esp.)

    _🏆 Resultado_
      📌 *Victoria Local*
      ├ Prob 81.3%  ·  Cuota ✅ 1.34  ·  Edge +8.9%
      └ Kelly 1.2%  |  forma local 2.2pts/PJ
    """
    home = match["home"]
    away = match["away"]
    mu   = match["mu"]
    lam  = match["lam"]

    lines = [
        _SEP_THIN,
        f"🏟 *{home} vs {away}*",
        f"🏆 {match['league']}  ·  📈 `{mu:.1f}` – `{lam:.1f}` _(esp. {mu+lam:.1f} goles)_",
        "",
    ]

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
            lines.append(_render_bet_line(pd.Series(bet), is_baja=is_baja))
            lines.append("")  # espacio entre apuestas del mismo grupo

    return lines


def _build_header_summary(alta, media, baja, total) -> list[str]:
    """
    Resumen compacto en el encabezado para ver el día de un vistazo.

    📊 Hoy: 8 apuestas  |  🟢×3  🟡×3  🔵×2
    """
    parts = []
    if not alta.empty:
        parts.append(f"🟢×{len(alta)}")
    if not media.empty:
        parts.append(f"🟡×{len(media)}")
    if not baja.empty:
        parts.append(f"🔵×{len(baja)}")

    summary = "  ".join(parts) if parts else "sin señales"
    return [
        f"📊 *{total} apuesta{'s' if total != 1 else ''}*  |  {summary}",
    ]


def format_message(bets_df: pd.DataFrame, model_stats: dict = None) -> str:
    today = date.today()
    fecha = format_date_es(today)
    total = len(bets_df) if not bets_df.empty else 0

    alta  = bets_df[bets_df["confidence"] == "alta"]  if not bets_df.empty else pd.DataFrame()
    media = bets_df[bets_df["confidence"] == "media"] if not bets_df.empty else pd.DataFrame()
    baja  = bets_df[bets_df["confidence"] == "baja"]  if not bets_df.empty else pd.DataFrame()

    lines = [
        f"⚽ *FOOTBOT  ·  {fecha}*",
        _SEP_THICK,
    ]
    lines += _build_header_summary(alta, media, baja, total)
    lines.append("")

    # ── ALTA ─────────────────────────────────────────────────────────────
    if not alta.empty:
        lines += [f"🟢 *ALTA CONFIANZA*", ""]
        for match in _group_bets_by_match(alta):
            lines += _render_match_block(match)
        lines.append("")

    # ── MEDIA ─────────────────────────────────────────────────────────────
    if not media.empty:
        lines += [f"🟡 *MEDIA CONFIANZA*", ""]
        for match in _group_bets_by_match(media):
            lines += _render_match_block(match)
        lines.append("")

    # ── BAJA ──────────────────────────────────────────────────────────────
    if not baja.empty:
        lines += [
            "🔵 *BAJA CONFIANZA*",
            "_Señales exploratorias — gestiona el stake con cuidado_",
            "",
        ]
        for match in _group_bets_by_match(baja):
            lines += _render_match_block(match, is_baja=True)
        lines.append("")

    if total == 0:
        lines += [
            "❌ *Sin value bets hoy*",
            "_No se encontró edge suficiente en los partidos disponibles._",
            "",
        ]

    lines.append(_SEP_THICK)

    # ── Estadísticas ──────────────────────────────────────────────────────
    if model_stats and model_stats.get("total", 0) > 10:
        tot  = model_stats.get("total", 0)
        tasa = model_stats.get("tasa_pct", 0)
        roi  = model_stats.get("roi_pct", 0)

        lines.append(f"📈 *Rendimiento acumulado* _{tot} apuestas_")
        lines.append("")

        # Tabla compacta con pipes
        header = "`Nivel  | Acierto |   ROI  `"
        sep    = "`───────┼─────────┼────────`"
        lines += [header, sep]

        def _stat_row(badge, label, tasa_key, roi_key):
            t = model_stats.get(tasa_key)
            r = model_stats.get(roi_key)
            if t is None:
                return None
            roi_fmt = f"{r:+.1f}%" if r is not None else " —"
            return f"`{label:<6} | {str(t)+'%':>7} | {roi_fmt:>6} ` {badge}"

        for badge, label, tk, rk in [
            ("🟢", "Alta",  "tasa_alta_pct",  "roi_alta_pct"),
            ("🟡", "Media", "tasa_media_pct", "roi_media_pct"),
            ("🔵", "Baja",  "tasa_baja_pct",  "roi_baja_pct"),
        ]:
            row = _stat_row(badge, label, tk, rk)
            if row:
                lines.append(row)

        # Total al pie
        lines.append(sep)
        lines.append(f"`{'Total':<6} | {str(tasa)+'%':>7} | {roi:+.1f}% ` 📊")

        # Top mercados si disponible
        top_mkts = model_stats.get("top_mercados")
        if top_mkts:
            mkt_str = "  ·  ".join(
                f"{m} `{v}`" for m, v in list(top_mkts.items())[:3]
            )
            lines += ["", f"📌 _Mejores mercados:_ {mkt_str}"]
    else:
        lines.append("_Stats disponibles tras 10+ apuestas cerradas_")

    lines += [
        "",
        "⚠️ _Modelo experimental · apuesta con responsabilidad_",
    ]
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    if not _credentials_ok():
        log.info(f"[TELEGRAM SIMULADO]\n{message[:600]}...")
        return False

    url    = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
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


def send_simple_alert(title: str, body: str, emoji: str = "ℹ️") -> bool:
    """Notificación rápida sin DataFrame — para errores, avisos del scheduler, etc."""
    msg = f"{emoji} *{title}*\n{_SEP_THIN}\n{body}"
    return send_telegram(msg)


def send_daily_report(bets_df: pd.DataFrame, model_stats: dict = None) -> bool:
    message = format_message(bets_df, model_stats)
    log.info(f"Reporte Telegram: {len(message)} chars")
    return send_telegram(message)


def send_error_notification(error_msg: str):
    send_telegram(f"🚨 *FOOTBOT ERROR*\n```{error_msg[:500]}```")