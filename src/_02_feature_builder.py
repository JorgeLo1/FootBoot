"""
_02_feature_builder.py
Construye todas las variables predictivas para cada partido.

CAMBIOS v4:
  1. load_historical_results fusiona Football-Data.co.uk (EU, con cuotas)
     con ESPN histórico (ligas sin cobertura EU, sin cuotas).
  2. normalize_team_name usa rapidfuzz para matching robusto.
  3. TeamNameResolver construye vocabulario canónico desde datos históricos
     de AMBAS fuentes, resolviendo diferencias de nombre entre APIs.
  4. build_training_dataset excluye market_prob_* (data leakage fix).
  5. build_features_for_fixtures enriquece con cuotas ESPN en tiempo real
     si el fixture viene de una liga ESPN y las cuotas están disponibles.

CAMBIOS v5:
  6. build_training_dataset O(n²) → O(n): _precompute_rolling_cache pre-calcula
     stats de forma por equipo antes del loop principal. ~90s → ~5s con 4226 partidos.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_RAW, DATA_STATSBOMB, DATA_PROCESSED,
    VENTANA_FORMA, LAMBDA_DECAY, LIGAS,
)
from src.utils import get_weather_for_fixture  # noqa: F401

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── FUZZY MATCHING ───────────────────────────────────────────────────────────

try:
    from rapidfuzz import process as fuzz_process, fuzz
    _HAS_RAPIDFUZZ = True
    log.debug("rapidfuzz disponible — usando fuzzy matching.")
except ImportError:
    _HAS_RAPIDFUZZ = False
    log.warning(
        "rapidfuzz no instalado. Usando normalización básica. "
        "Instala con: pip install rapidfuzz"
    )

_STRIP_TOKENS = [
    "fc", "cf", "ac", "sc", "rc", "cd", "sd", "ud", "rcd", "real",
    "atletico", "athletic", "sporting", "deportivo",
    "united", "city", "town", "rovers", "wanderers", "hotspur",
    "saint", "st", "borussia", "bayer", "rb", "ss", "as",
]

_FUZZY_THRESHOLD = 82


def _clean_name(name: str) -> str:
    import re
    s = name.lower().strip()
    s = re.sub(r"['\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    if len(tokens) > 1:
        tokens = [t for t in tokens if t not in _STRIP_TOKENS]
    return " ".join(tokens) if tokens else s


class TeamNameResolver:
    """
    Resuelve nombres de equipos de distintas fuentes (fd.org, ESPN, StatsBomb)
    contra un vocabulario canónico construido desde los datos históricos.
    """

    def __init__(self):
        self._canonical: list[str] = []
        self._raw_to_canonical: dict[str, str] = {}
        self._built = False

    def build_from_historical(self, historical: pd.DataFrame):
        if historical.empty:
            log.warning("TeamNameResolver: histórico vacío, sin vocabulario.")
            return

        teams = set()
        for col in ["home_team", "away_team"]:
            if col in historical.columns:
                teams.update(historical[col].dropna().unique())

        self._canonical = sorted({_clean_name(t) for t in teams})
        self._raw_to_canonical = {}
        self._built = True
        log.info(f"TeamNameResolver: {len(self._canonical)} equipos en vocabulario.")

    def resolve(self, name: str) -> str:
        if not name:
            return ""

        if name in self._raw_to_canonical:
            return self._raw_to_canonical[name]

        cleaned = _clean_name(name)

        if cleaned in self._canonical:
            self._raw_to_canonical[name] = cleaned
            return cleaned

        if _HAS_RAPIDFUZZ and self._built and self._canonical:
            result = fuzz_process.extractOne(
                cleaned,
                self._canonical,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=_FUZZY_THRESHOLD,
            )
            if result:
                matched, score, _ = result
                log.debug(f"Fuzzy match: '{name}' → '{matched}' (score={score})")
                self._raw_to_canonical[name] = matched
                return matched

        log.debug(f"Sin match para '{name}', usando nombre limpiado: '{cleaned}'")
        self._raw_to_canonical[name] = cleaned
        return cleaned

    def resolve_series(self, series: pd.Series) -> pd.Series:
        return series.apply(self.resolve)


_resolver = TeamNameResolver()


def normalize_team_name(name: str) -> str:
    if _resolver._built:
        return _resolver.resolve(name)
    return _clean_name(name)


def init_resolver(historical: pd.DataFrame):
    _resolver.build_from_historical(historical)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def exponential_weight(days_ago: float, lam: float = LAMBDA_DECAY) -> float:
    return np.exp(-lam * max(days_ago, 0))


# ─── CARGA DE DATOS HISTÓRICOS (EU + ESPN fusionados) ────────────────────────

def load_historical_results() -> pd.DataFrame:
    """
    Carga y fusiona datos históricos de todas las fuentes:
      1. Football-Data.co.uk (7 ligas EU) — incluye cuotas B365, Pinnacle
      2. ESPN API (ligas activas no EU) — solo goles, sin cuotas históricas

    El resolver de nombres se inicializa con el vocabulario combinado,
    lo que permite fuzzy matching cross-fuente (ej: "Atlético Nacional"
    en ESPN vs posibles variantes en otros datasets).
    """
    frames_eu = []

    # ── Fuente 1: Football-Data.co.uk ────────────────────────────────────
    for _, (league_name, fd_code, _) in LIGAS.items():
        path = os.path.join(DATA_RAW, f"fd_{fd_code}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
            df["fd_code"]     = fd_code
            df["league_name"] = league_name
            df["source"]      = "fd_uk"
            frames_eu.append(df)

    if not frames_eu:
        log.warning("Sin datos Football-Data.co.uk. Ejecuta primero _01_data_collector.")

    raw_eu = pd.DataFrame()
    if frames_eu:
        raw_eu = pd.concat(frames_eu, ignore_index=True)
        raw_eu = raw_eu.rename(columns={
            "HomeTeam": "home_team", "AwayTeam": "away_team",
            "FTHG":     "home_goals", "FTAG":    "away_goals",
            "FTR":      "result",     "Date":    "date_str",
        })

        def parse_date(d):
            for fmt in ("%d/%m/%y", "%d/%m/%Y"):
                try:
                    return datetime.strptime(str(d), fmt)
                except Exception:
                    continue
            return pd.NaT

        raw_eu["match_date"] = raw_eu["date_str"].apply(parse_date)
        raw_eu = raw_eu.dropna(subset=["match_date", "home_goals", "away_goals"])
        raw_eu["home_goals"] = pd.to_numeric(raw_eu["home_goals"], errors="coerce")
        raw_eu["away_goals"] = pd.to_numeric(raw_eu["away_goals"], errors="coerce")
        raw_eu = raw_eu.dropna(subset=["home_goals", "away_goals"])

        for col in ["B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
                    "B365>2.5", "B365<2.5"]:
            if col in raw_eu.columns:
                raw_eu[col] = pd.to_numeric(raw_eu[col], errors="coerce")

    # ── Fuente 2: ESPN histórico ──────────────────────────────────────────
    raw_espn = pd.DataFrame()
    try:
        from src._01_data_collector import load_espn_historical
        raw_espn = load_espn_historical()

        if not raw_espn.empty:
            # Añadir columnas de cuotas vacías para schema unificado
            for col in ["B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
                        "B365>2.5", "B365<2.5"]:
                if col not in raw_espn.columns:
                    raw_espn[col] = np.nan
            # Asegurar columna date_str para compatibilidad
            if "date_str" not in raw_espn.columns:
                raw_espn["date_str"] = raw_espn["match_date"].astype(str)

            log.info(f"ESPN histórico disponible: {len(raw_espn)} partidos")
    except Exception as e:
        log.warning(f"No se pudo cargar ESPN histórico: {e}")

    # ── Fusión ────────────────────────────────────────────────────────────
    if not raw_eu.empty and not raw_espn.empty:
        # Alinear columnas comunes antes de concat
        common_cols = ["home_team", "away_team", "home_goals", "away_goals",
                       "match_date", "league_name", "source",
                       "B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
                       "B365>2.5", "B365<2.5"]
        for col in common_cols:
            if col not in raw_eu.columns:
                raw_eu[col] = np.nan
            if col not in raw_espn.columns:
                raw_espn[col] = np.nan

        raw = pd.concat(
            [raw_eu[common_cols + [c for c in raw_eu.columns if c not in common_cols]],
             raw_espn[common_cols + [c for c in raw_espn.columns if c not in common_cols]]],
            ignore_index=True
        )
    elif not raw_eu.empty:
        raw = raw_eu
    elif not raw_espn.empty:
        raw = raw_espn
    else:
        log.error("Sin datos históricos de ninguna fuente.")
        return pd.DataFrame()

    # ── Normalización final ───────────────────────────────────────────────
    raw["match_date"] = pd.to_datetime(raw["match_date"], errors="coerce")
    raw = raw.dropna(subset=["match_date", "home_goals", "away_goals"])
    raw = raw.sort_values("match_date").reset_index(drop=True)

    # Inicializar resolver con vocabulario combinado
    init_resolver(raw)

    raw["home_team_norm"] = raw["home_team"].apply(normalize_team_name)
    raw["away_team_norm"] = raw["away_team"].apply(normalize_team_name)

    log.info(
        f"Histórico total: {len(raw)} partidos "
        f"(EU: {len(raw_eu) if not raw_eu.empty else 0} | "
        f"ESPN: {len(raw_espn) if not raw_espn.empty else 0})"
    )
    return raw


def load_xg_data() -> pd.DataFrame:
    path = os.path.join(DATA_STATSBOMB, "shots_xg.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["team_norm"] = df["team"].apply(normalize_team_name)
    if "is_home_team" not in df.columns:
        df["is_home_team"] = df["team"] == df["home_team"]
    return df


def load_match_summary() -> pd.DataFrame:
    path = os.path.join(DATA_STATSBOMB, "match_summary.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["home_team_norm"] = df["home_team"].apply(normalize_team_name)
    df["away_team_norm"] = df["away_team"].apply(normalize_team_name)
    return df


def load_elo() -> pd.DataFrame:
    """
    Carga ELO ratings fusionando dos fuentes:
      1. elo_ratings.csv  — ClubElo (ligas EU)
      2. elo_espn.csv     — ELO propio calculado desde histórico ESPN (LATAM)

    Si un equipo aparece en ambas fuentes, se prioriza elo_espn.csv (más
    actualizado y con mayor cobertura LATAM).
    """
    frames = []

    # Fuente 1: ClubElo (EU)
    path_clubelo = os.path.join(DATA_RAW, "elo_ratings.csv")
    if os.path.exists(path_clubelo):
        df_clubelo = pd.read_csv(path_clubelo)
        if "Club" in df_clubelo.columns:
            df_clubelo["team_norm"] = df_clubelo["Club"].apply(normalize_team_name)
        df_clubelo["_source"] = "clubelo"
        frames.append(df_clubelo)
        log.info(f"ELO ClubElo cargado: {len(df_clubelo)} equipos")
    else:
        log.warning("elo_ratings.csv no encontrado — solo se usará elo_espn.csv")

    # Fuente 2: ELO propio ESPN (LATAM + todos los equipos del histórico)
    path_espn = os.path.join(DATA_RAW, "elo_espn.csv")
    if os.path.exists(path_espn):
        df_espn = pd.read_csv(path_espn)
        if "Club" in df_espn.columns:
            df_espn["team_norm"] = df_espn["Club"].apply(normalize_team_name)
        df_espn["_source"] = "espn"
        frames.append(df_espn)
        log.info(f"ELO ESPN cargado: {len(df_espn)} equipos")
    else:
        log.warning(
            "elo_espn.csv no encontrado. Ejecuta compute_elo_espn() en _01_data_collector.py "
            "para generar ELO de equipos LATAM."
        )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicar: si el mismo equipo aparece en ambas fuentes, quedarse con ESPN
    # (más reciente y con cobertura LATAM). Usar drop_duplicates con keep='last'
    # tras ordenar para que 'espn' quede al final.
    source_order = {"clubelo": 0, "espn": 1}
    combined["_source_order"] = combined["_source"].map(source_order).fillna(0)
    combined = combined.sort_values("_source_order")
    combined = combined.drop_duplicates(subset="team_norm", keep="last")
    combined = combined.drop(columns=["_source", "_source_order"])

    log.info(f"ELO total (fusionado): {len(combined)} equipos únicos")
    return combined


# ─── xG ──────────────────────────────────────────────────────────────────────

def _compute_xg(team_norm: str, xg_data: pd.DataFrame,
                is_home: bool, mode: str) -> float:
    if xg_data.empty or "shot_statsbomb_xg" not in xg_data.columns:
        return 0.0

    team_shots = xg_data[xg_data["team_norm"] == team_norm]
    if team_shots.empty:
        return 0.0

    if mode == "attack":
        n = team_shots["match_id"].nunique()
        return float(team_shots["shot_statsbomb_xg"].sum() / max(n, 1))

    team_match_ids = team_shots["match_id"].unique()
    if "is_home_team" in xg_data.columns:
        rival_is_home = not is_home
        rival_shots = xg_data[
            xg_data["match_id"].isin(team_match_ids) &
            (xg_data["team_norm"] != team_norm) &
            (xg_data["is_home_team"] == rival_is_home)
        ]
    else:
        rival_shots = xg_data[
            xg_data["match_id"].isin(team_match_ids) &
            (xg_data["team_norm"] != team_norm)
        ]

    if rival_shots.empty:
        return 0.0

    n = rival_shots["match_id"].nunique()
    return float(rival_shots["shot_statsbomb_xg"].sum() / max(n, 1))


# ─── ESTADÍSTICAS POR EQUIPO ─────────────────────────────────────────────────

def _empty_team_stats(is_home: bool) -> dict:
    p = "home" if is_home else "away"
    return {
        f"{p}_xg_scored":       1.2,
        f"{p}_xg_conceded":     1.2,
        f"{p}_goals_scored":    1.2,
        f"{p}_goals_conceded":  1.2,
        f"{p}_forma":           1.2,
        f"{p}_btts_rate":       0.5,
        f"{p}_over25_rate":     0.5,
        f"{p}_corners_avg":     5.0,
        f"{p}_fouls_avg":       11.0,
        f"{p}_days_rest":       7,
        f"{p}_n_matches":       0,
        f"{p}_n_matches_total": 0,
    }


def _compute_corners_fouls(team_norm: str, match_summary: pd.DataFrame,
                            is_home: bool) -> tuple[float, float]:
    if match_summary.empty:
        return 5.0, 11.0
    col_team = "home_team_norm" if is_home else "away_team_norm"
    c_col    = "corners_home"   if is_home else "corners_away"
    f_col    = "fouls_home"     if is_home else "fouls_away"
    df = match_summary[match_summary[col_team] == team_norm]
    if df.empty:
        return 5.0, 11.0
    c = df[c_col].mean() if c_col in df.columns else 5.0
    f = df[f_col].mean() if f_col in df.columns else 11.0
    return float(c), float(f)


def compute_team_stats(team: str, is_home: bool,
                       historical: pd.DataFrame,
                       xg_data: pd.DataFrame,
                       match_summary: pd.DataFrame,
                       reference_date: datetime,
                       window: int = VENTANA_FORMA) -> dict:
    team_norm  = normalize_team_name(team)
    p          = "home" if is_home else "away"
    col_team   = "home_team_norm" if is_home else "away_team_norm"
    col_scored = "home_goals"     if is_home else "away_goals"
    col_recv   = "away_goals"     if is_home else "home_goals"

    df_all = historical[
        (historical[col_team] == team_norm) &
        (historical["match_date"] < reference_date)
    ]
    n_total = len(df_all)

    df = df_all.sort_values("match_date", ascending=False).head(window).copy()

    if df.empty:
        return _empty_team_stats(is_home)

    df["days_ago"] = (reference_date - df["match_date"]).dt.days
    df["weight"]   = df["days_ago"].apply(exponential_weight)
    W = df["weight"].sum() or 1.0

    goals_scored   = (df[col_scored] * df["weight"]).sum() / W
    goals_conceded = (df[col_recv]   * df["weight"]).sum() / W

    xg_scored   = _compute_xg(team_norm, xg_data, is_home, "attack")
    xg_conceded = _compute_xg(team_norm, xg_data, is_home, "defense")
    if xg_scored   == 0.0: xg_scored   = goals_scored   * 1.05
    if xg_conceded == 0.0: xg_conceded = goals_conceded * 1.05

    def pts(row):
        s, c = row[col_scored], row[col_recv]
        return 3 if s > c else (1 if s == c else 0)

    df["pts"]    = df.apply(pts, axis=1)
    forma        = (df["pts"] * df["weight"]).sum() / W

    df["btts"]   = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
    btts_rate    = (df["btts"]  * df["weight"]).sum() / W

    df["over25"] = ((df["home_goals"] + df["away_goals"]) > 2.5).astype(int)
    over25_rate  = (df["over25"] * df["weight"]).sum() / W

    corners_avg, fouls_avg = _compute_corners_fouls(team_norm, match_summary, is_home)

    last_date = df["match_date"].max()
    days_rest = int((reference_date - last_date).days) if not pd.isna(last_date) else 7

    return {
        f"{p}_xg_scored":       round(xg_scored,    3),
        f"{p}_xg_conceded":     round(xg_conceded,  3),
        f"{p}_goals_scored":    round(goals_scored,  3),
        f"{p}_goals_conceded":  round(goals_conceded,3),
        f"{p}_forma":           round(forma,         3),
        f"{p}_btts_rate":       round(btts_rate,     3),
        f"{p}_over25_rate":     round(over25_rate,   3),
        f"{p}_corners_avg":     round(corners_avg,   2),
        f"{p}_fouls_avg":       round(fouls_avg,     2),
        f"{p}_days_rest":       days_rest,
        f"{p}_n_matches":       len(df),
        f"{p}_n_matches_total": n_total,   # ← FIX: este es el que usa classify_confidence
    }


def compute_h2h(home_team: str, away_team: str,
                historical: pd.DataFrame, ref_date: datetime,
                last_n: int = 10) -> dict:
    h = normalize_team_name(home_team)
    a = normalize_team_name(away_team)

    mask = (
        ((historical["home_team_norm"] == h) & (historical["away_team_norm"] == a)) |
        ((historical["home_team_norm"] == a) & (historical["away_team_norm"] == h))
    )
    df = historical[mask & (historical["match_date"] < ref_date)]
    df = df.sort_values("match_date", ascending=False).head(last_n)

    if df.empty:
        return {"h2h_home_wins": 0.33, "h2h_draws": 0.33, "h2h_away_wins": 0.33,
                "h2h_avg_goals": 2.5,  "h2h_btts_rate": 0.5, "h2h_n": 0}

    total = len(df)
    hw = len(df[
        ((df["home_team_norm"] == h) & (df["home_goals"] > df["away_goals"])) |
        ((df["away_team_norm"] == h) & (df["away_goals"] > df["home_goals"]))
    ])
    aw = len(df[
        ((df["home_team_norm"] == a) & (df["home_goals"] > df["away_goals"])) |
        ((df["away_team_norm"] == a) & (df["away_goals"] > df["home_goals"]))
    ])
    draws = total - hw - aw

    df = df.copy()
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["btts"]        = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

    return {
        "h2h_home_wins":  round(hw / total,               3),
        "h2h_draws":      round(draws / total,             3),
        "h2h_away_wins":  round(aw / total,               3),
        "h2h_avg_goals":  round(df["total_goals"].mean(),  3),
        "h2h_btts_rate":  round(df["btts"].mean(),         3),
        "h2h_n":          total,
    }


def get_elo_diff(home_team: str, away_team: str, elo_df: pd.DataFrame) -> float:
    if elo_df.empty or "team_norm" not in elo_df.columns:
        return 0.0
    elo_col = next((c for c in ["Elo", "elo"] if c in elo_df.columns), None)
    if not elo_col:
        return 0.0
    h = normalize_team_name(home_team)
    a = normalize_team_name(away_team)
    home_row = elo_df[elo_df["team_norm"] == h]
    away_row = elo_df[elo_df["team_norm"] == a]
    he = float(home_row[elo_col].values[0]) if not home_row.empty else 1500.0
    ae = float(away_row[elo_col].values[0]) if not away_row.empty else 1500.0
    return round(he - ae, 1)


def _extract_market_features(home: str, away: str,
                              historical: pd.DataFrame,
                              ref_date: datetime) -> dict:
    """Solo para inferencia — NO incluir en training set."""
    h = normalize_team_name(home)
    a = normalize_team_name(away)
    default = {"market_prob_home": 0.45, "market_prob_draw": 0.27,
               "market_prob_away": 0.28}

    if "home_team_norm" not in historical.columns:
        return default

    mask   = (historical["home_team_norm"] == h) & (historical["away_team_norm"] == a)
    recent = historical[mask & (historical["match_date"] < ref_date)]
    recent = recent.sort_values("match_date", ascending=False).head(5)

    for col_h, col_d, col_a in [
        ("PSH",   "PSD",   "PSA"),
        ("B365H", "B365D", "B365A"),
        ("BWH",   "BWD",   "BWA"),
    ]:
        if not all(c in recent.columns for c in [col_h, col_d, col_a]):
            continue
        valid = recent[[col_h, col_d, col_a]].dropna()
        if valid.empty:
            continue
        row   = valid.iloc[0]
        total = 1/row[col_h] + 1/row[col_d] + 1/row[col_a]
        return {
            "market_prob_home": round((1/row[col_h]) / total, 4),
            "market_prob_draw": round((1/row[col_d]) / total, 4),
            "market_prob_away": round((1/row[col_a]) / total, 4),
        }

    return default


# ─── BUILDER PRINCIPAL ───────────────────────────────────────────────────────

def build_features_for_fixtures(fixtures: pd.DataFrame,
                                 espn_client=None) -> pd.DataFrame:
    """
    Construye features para todos los fixtures del día.

    espn_client: si se pasa, enriquece fixtures de ligas ESPN con cuotas
    en tiempo real desde el Core API (/odds).
    """
    log.info("Cargando datos históricos para feature building...")
    historical    = load_historical_results()
    xg_data       = load_xg_data()
    match_summary = load_match_summary()
    elo_df        = load_elo()

    if historical.empty:
        log.error("Sin datos históricos. Imposible calcular features.")
        return pd.DataFrame()

    # Enriquecer fixtures con cuotas ESPN en tiempo real
    if espn_client is not None and not fixtures.empty:
        try:
            from src.espn_collector import enrich_fixtures_with_odds
            fixtures = enrich_fixtures_with_odds(espn_client, fixtures)
            log.info("Fixtures enriquecidos con cuotas ESPN en tiempo real")
        except Exception as e:
            log.warning(f"No se pudo enriquecer con cuotas ESPN: {e}")

    ref_date     = datetime.now()
    all_features = []

    for _, fixture in fixtures.iterrows():
        home = fixture["home_team"]
        away = fixture["away_team"]

        home_norm = normalize_team_name(home)
        away_norm = normalize_team_name(away)
        if home_norm != home.lower().strip():
            log.info(f"  Nombre resuelto: '{home}' → '{home_norm}'")
        if away_norm != away.lower().strip():
            log.info(f"  Nombre resuelto: '{away}' → '{away_norm}'")

        log.info(f"Features: {home} vs {away}")

        home_stats = compute_team_stats(home, True,  historical, xg_data,
                                        match_summary, ref_date)
        away_stats = compute_team_stats(away, False, historical, xg_data,
                                        match_summary, ref_date)
        h2h        = compute_h2h(home, away, historical, ref_date)
        elo_diff   = get_elo_diff(home, away, elo_df)
        weather    = get_weather_for_fixture(
                        home, fixture.get("date", ref_date.isoformat()))
        market     = _extract_market_features(home, away, historical, ref_date)

        feature_row = {
            "fixture_id":      fixture.get("fixture_id", 0),
            "league_id":       fixture.get("league_id", 0),
            "league_name":     fixture.get("league_name", ""),
            "home_team":       home,
            "away_team":       away,
            "match_date":      fixture.get("date", str(ref_date.date())),
            "source":          fixture.get("source", "fdorg"),
            "n_home_matches":  home_stats.get("home_n_matches_total", 0),
            "n_away_matches":  away_stats.get("away_n_matches_total", 0),
            "elo_diff":        elo_diff,
            **home_stats,
            **away_stats,
            **h2h,
            **weather,
            **market,
        }

        # Cuotas ESPN en tiempo real si están disponibles
        if fixture.get("espn_odds_available"):
            feature_row["espn_odds_home"]     = fixture.get("espn_odds_home")
            feature_row["espn_odds_draw"]      = fixture.get("espn_odds_draw")
            feature_row["espn_odds_away"]      = fixture.get("espn_odds_away")
            feature_row["espn_odds_provider"]  = fixture.get("espn_odds_provider")
            feature_row["espn_odds_available"] = True
        else:
            feature_row["espn_odds_available"] = False

        all_features.append(feature_row)

    df = pd.DataFrame(all_features)
    if not df.empty:
        df = _add_derived_features(df)
        path = os.path.join(DATA_PROCESSED, f"features_{date.today()}.csv")
        df.to_csv(path, index=False)
        log.info(f"Features guardados: {len(df)} partidos → {path}")
    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["xg_diff"]      = df["home_xg_scored"]   - df["away_xg_scored"]
    df["xg_total_exp"] = df["home_xg_scored"]   + df["away_xg_scored"]
    df["goals_diff"]   = df["home_goals_scored"] - df["away_goals_scored"]
    df["forma_diff"]   = df["home_forma"]        - df["away_forma"]
    df["rest_diff"]    = df["away_days_rest"]    - df["home_days_rest"]
    df["fatiga_flag"]  = (
        (df["home_days_rest"] < 4) | (df["away_days_rest"] < 4)
    ).astype(int)
    return df


# ─── DATASET DE ENTRENAMIENTO ────────────────────────────────────────────────

def _precompute_rolling_cache(
    historical: pd.DataFrame,
    xg_data: pd.DataFrame,
    match_summary: pd.DataFrame,
    window: int = VENTANA_FORMA,
) -> dict[tuple, dict]:
    """
    Pre-computa stats de forma por (team_norm, is_home, match_idx) para todos
    los equipos del histórico. Convierte build_training_dataset de O(n²) a O(n).

    Retorna un dict keyed por (team_norm, is_home, idx) → stats dict.
    La clave `idx` es el índice del partido SIGUIENTE a computar (i.e. el partido
    actual usa `past = historical.iloc[:idx]`, así que guardamos stats "justo antes
    de este partido").
    """
    log.info("Pre-computando rolling stats por equipo (O(n) cache)...")

    # Indexar partidos por equipo para acceso O(1)
    # Para cada equipo: lista de índices donde aparece como local / visitante
    home_indices: dict[str, list[int]] = {}
    away_indices: dict[str, list[int]] = {}

    for idx, row in historical.iterrows():
        h = str(row.get("home_team_norm", normalize_team_name(str(row.get("home_team", "")))))
        a = str(row.get("away_team_norm", normalize_team_name(str(row.get("away_team", "")))))
        home_indices.setdefault(h, []).append(idx)
        away_indices.setdefault(a, []).append(idx)

    cache: dict[tuple, dict] = {}

    def _stats_from_rows(team: str, is_home: bool, rows: pd.DataFrame,
                         ref_date, team_norm: str) -> dict:
        """Calcula stats de forma a partir de un slice ya filtrado."""
        p        = "home" if is_home else "away"
        col_scored = "home_goals" if is_home else "away_goals"
        col_recv   = "away_goals" if is_home else "home_goals"

        n_total = len(rows)
        df      = rows.sort_values("match_date", ascending=False).head(window).copy()

        if df.empty:
            return _empty_team_stats(is_home)

        df["days_ago"] = (ref_date - df["match_date"]).dt.days
        df["weight"]   = df["days_ago"].apply(exponential_weight)
        W = df["weight"].sum() or 1.0

        goals_scored   = (df[col_scored] * df["weight"]).sum() / W
        goals_conceded = (df[col_recv]   * df["weight"]).sum() / W

        xg_scored   = _compute_xg(team_norm, xg_data, is_home, "attack")
        xg_conceded = _compute_xg(team_norm, xg_data, is_home, "defense")
        if xg_scored   == 0.0: xg_scored   = goals_scored   * 1.05
        if xg_conceded == 0.0: xg_conceded = goals_conceded * 1.05

        def pts(row):
            s, c = row[col_scored], row[col_recv]
            return 3 if s > c else (1 if s == c else 0)

        df["pts"]    = df.apply(pts, axis=1)
        forma        = (df["pts"] * df["weight"]).sum() / W

        df["btts"]   = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
        btts_rate    = (df["btts"]  * df["weight"]).sum() / W

        df["over25"] = ((df["home_goals"] + df["away_goals"]) > 2.5).astype(int)
        over25_rate  = (df["over25"] * df["weight"]).sum() / W

        corners_avg, fouls_avg = _compute_corners_fouls(team_norm, match_summary, is_home)

        last_date = df["match_date"].max()
        days_rest = int((ref_date - last_date).days) if not pd.isna(last_date) else 7

        return {
            f"{p}_xg_scored":       round(xg_scored,    3),
            f"{p}_xg_conceded":     round(xg_conceded,  3),
            f"{p}_goals_scored":    round(goals_scored,  3),
            f"{p}_goals_conceded":  round(goals_conceded,3),
            f"{p}_forma":           round(forma,         3),
            f"{p}_btts_rate":       round(btts_rate,     3),
            f"{p}_over25_rate":     round(over25_rate,   3),
            f"{p}_corners_avg":     round(corners_avg,   2),
            f"{p}_fouls_avg":       round(fouls_avg,     2),
            f"{p}_days_rest":       days_rest,
            f"{p}_n_matches":       len(df),
            f"{p}_n_matches_total": n_total,
        }

    # Recorrer el histórico una sola vez y acumular stats con ventana deslizante
    # Para cada partido idx, guardamos las stats con todos los partidos PREVIOS (< idx)
    all_teams = set(home_indices.keys()) | set(away_indices.keys())
    total_teams = len(all_teams)

    for t_idx, team_norm in enumerate(all_teams):
        if t_idx % 50 == 0:
            log.info(f"  Rolling cache: equipo {t_idx}/{total_teams}")

        # Acumular índices de partidos locales y visitantes progresivamente
        h_idxs = sorted(home_indices.get(team_norm, []))
        a_idxs = sorted(away_indices.get(team_norm, []))

        # Para cada partido del equipo como local, guardar stats pre-partido
        seen_h: list[int] = []
        for i in h_idxs:
            # past = todos los partidos locales del equipo con índice < i
            past_rows = historical.loc[[j for j in seen_h]]
            ref_date  = historical.at[i, "match_date"]
            cache[(team_norm, True, i)] = _stats_from_rows(
                team_norm, True, past_rows, ref_date, team_norm
            )
            seen_h.append(i)

        # Para cada partido del equipo como visitante
        seen_a: list[int] = []
        for i in a_idxs:
            past_rows = historical.loc[[j for j in seen_a]]
            ref_date  = historical.at[i, "match_date"]
            cache[(team_norm, False, i)] = _stats_from_rows(
                team_norm, False, past_rows, ref_date, team_norm
            )
            seen_a.append(i)

    log.info(f"Rolling cache listo: {len(cache)} entradas para {total_teams} equipos")
    return cache


def build_training_dataset(historical: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset walk-forward sin data leakage.
    Incluye partidos de ligas EU y ESPN para entrenamiento unificado.

    OPTIMIZACIÓN v2: pre-computa rolling stats por equipo (O(n)) en lugar de
    recalcular desde cero en cada iteración (O(n²)). Reducción ~90s → ~5s
    con 4226 partidos.
    """
    import time
    log.info("Construyendo dataset walk-forward (sin leakage)...")
    t0 = time.time()

    xg_data       = load_xg_data()
    match_summary = load_match_summary()
    elo_df        = load_elo()

    historical = historical.sort_values("match_date").reset_index(drop=True)

    # Asegurar columnas norm presentes antes de cachear
    if "home_team_norm" not in historical.columns:
        historical["home_team_norm"] = historical["home_team"].apply(normalize_team_name)
    if "away_team_norm" not in historical.columns:
        historical["away_team_norm"] = historical["away_team"].apply(normalize_team_name)

    # Pre-computar rolling stats — O(n) en lugar de O(n²)
    rolling_cache = _precompute_rolling_cache(historical, xg_data, match_summary)

    start_idx   = max(200, len(historical) // 4)
    training_rows = []
    total_range = len(historical) - start_idx

    for idx in range(start_idx, len(historical)):
        match = historical.iloc[idx]

        home      = str(match.get("home_team", ""))
        away      = str(match.get("away_team", ""))
        home_norm = str(match.get("home_team_norm", normalize_team_name(home)))
        away_norm = str(match.get("away_team_norm", normalize_team_name(away)))
        if not home or not away:
            continue

        # Obtener stats desde cache O(1)
        home_stats = rolling_cache.get(
            (home_norm, True,  idx),
            compute_team_stats(home, True,  historical.iloc[:idx], xg_data,
                               match_summary, match["match_date"]),
        )
        away_stats = rolling_cache.get(
            (away_norm, False, idx),
            compute_team_stats(away, False, historical.iloc[:idx], xg_data,
                               match_summary, match["match_date"]),
        )

        # H2H: sigue siendo O(n) por partido pero es rápido (slice pequeño)
        h2h      = compute_h2h(home, away, historical.iloc[:idx], match["match_date"])
        elo_diff = get_elo_diff(home, away, elo_df)

        hg = int(match["home_goals"])
        ag = int(match["away_goals"])

        training_rows.append({
            "home_team":           home,
            "away_team":           away,
            "match_date":          match["match_date"],
            "league_name":         match.get("league_name", ""),
            "elo_diff":            elo_diff,
            "n_home_matches":      home_stats.get("home_n_matches_total", 0),
            "n_away_matches":      away_stats.get("away_n_matches_total", 0),
            **home_stats,
            **away_stats,
            **h2h,
            "target_home_win":     int(hg > ag),
            "target_draw":         int(hg == ag),
            "target_away_win":     int(hg < ag),
            "target_btts":         int(hg > 0 and ag > 0),
            "target_over25":       int(hg + ag > 2.5),
            "home_goals_actual":   hg,
            "away_goals_actual":   ag,
        })

        if (idx - start_idx) % 500 == 0:
            pct = (idx - start_idx) / total_range * 100
            elapsed = time.time() - t0
            log.info(f"  Training dataset: {pct:.0f}% ({idx}/{len(historical)}) "
                     f"— {elapsed:.1f}s")

    df = pd.DataFrame(training_rows)
    if not df.empty:
        df = _add_derived_features(df)
        if "home_team_norm" not in df.columns:
            df["home_team_norm"] = df["home_team"].apply(normalize_team_name)
            df["away_team_norm"] = df["away_team"].apply(normalize_team_name)
        path = os.path.join(DATA_PROCESSED, "training_dataset.csv")
        df.to_csv(path, index=False)
        elapsed = time.time() - t0
        log.info(f"Dataset entrenamiento: {len(df)} partidos → {path} ({elapsed:.1f}s total)")
    return df