"""
_02_feature_builder.py
Construye todas las variables predictivas para cada partido.

CAMBIOS v3:
  1. normalize_team_name usa rapidfuzz para matching robusto en lugar de
     un diccionario hardcodeado. Fallback a limpieza básica si rapidfuzz
     no está instalado.
  2. TeamNameResolver construye un vocabulario canónico desde los datos
     históricos reales y resuelve nombres de la API contra ese vocabulario.
     Esto elimina el problema silencioso de equipos sin histórico por
     diferencias de nombre entre fuentes.
  3. build_training_dataset: el feature market_prob_* se excluye del
     training set (data leakage) y se añade solo en inferencia.
  4. build_training_dataset: optimización O(n log n) via índice por equipo.
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

# Prefijos y sufijos comunes a eliminar antes del matching
_STRIP_TOKENS = [
    "fc", "cf", "ac", "sc", "rc", "cd", "sd", "ud", "rcd", "real",
    "atletico", "athletic", "sporting", "deportivo",
    "united", "city", "town", "rovers", "wanderers", "hotspur",
    "saint", "st", "borussia", "bayer", "rb", "ss", "as",
]

# Umbral de similaridad mínimo para aceptar un match fuzzy
_FUZZY_THRESHOLD = 82


def _clean_name(name: str) -> str:
    """Limpieza básica: minúsculas, sin puntuación, sin prefijos comunes."""
    import re
    s = name.lower().strip()
    s = re.sub(r"['\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Eliminar tokens genéricos solo si no es el nombre completo
    tokens = s.split()
    if len(tokens) > 1:
        tokens = [t for t in tokens if t not in _STRIP_TOKENS]
    return " ".join(tokens) if tokens else s


class TeamNameResolver:
    """
    Construye un vocabulario canónico de nombres de equipo a partir de los
    datos históricos (Football-Data.co.uk) y resuelve nombres externos
    (football-data.org, StatsBomb) contra ese vocabulario usando fuzzy matching.

    Uso:
        resolver = TeamNameResolver()
        resolver.build_from_historical(historical_df)
        canonical = resolver.resolve("Manchester City FC")  # → "man city"
    """

    def __init__(self):
        self._canonical: list[str] = []          # nombres canónicos (ya normalizados)
        self._raw_to_canonical: dict[str, str] = {}  # caché de resoluciones
        self._built = False

    def build_from_historical(self, historical: pd.DataFrame):
        """Extrae todos los nombres únicos del histórico como vocabulario canónico."""
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
        """
        Resuelve un nombre externo al nombre canónico más cercano.
        Primero busca en caché; si no, intenta fuzzy match; si falla, devuelve
        la versión limpiada del nombre original.
        """
        if not name:
            return ""

        # 1. Caché
        if name in self._raw_to_canonical:
            return self._raw_to_canonical[name]

        cleaned = _clean_name(name)

        # 2. Match exacto tras limpieza
        if cleaned in self._canonical:
            self._raw_to_canonical[name] = cleaned
            return cleaned

        # 3. Fuzzy match
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

        # 4. Fallback: nombre limpiado sin match
        log.debug(f"Sin match para '{name}', usando nombre limpiado: '{cleaned}'")
        self._raw_to_canonical[name] = cleaned
        return cleaned

    def resolve_series(self, series: pd.Series) -> pd.Series:
        return series.apply(self.resolve)


# Instancia global — se inicializa con build_from_historical() al cargar datos
_resolver = TeamNameResolver()


def normalize_team_name(name: str) -> str:
    """
    Punto de entrada público. Usa el resolver global si está construido,
    o limpieza básica en caso contrario.
    """
    if _resolver._built:
        return _resolver.resolve(name)
    return _clean_name(name)


def init_resolver(historical: pd.DataFrame):
    """Inicializa el resolver global con el histórico. Llamar antes del pipeline."""
    _resolver.build_from_historical(historical)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def exponential_weight(days_ago: float, lam: float = LAMBDA_DECAY) -> float:
    return np.exp(-lam * max(days_ago, 0))


# ─── CARGA DE DATOS ───────────────────────────────────────────────────────────

def load_historical_results() -> pd.DataFrame:
    frames = []
    for _, (league_name, fd_code, _) in LIGAS.items():
        path = os.path.join(DATA_RAW, f"fd_{fd_code}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
            df["fd_code"]     = fd_code
            df["league_name"] = league_name
            frames.append(df)

    if not frames:
        log.warning("Sin datos Football-Data. Ejecuta primero _01_data_collector.")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={
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

    raw["match_date"] = raw["date_str"].apply(parse_date)
    raw = raw.dropna(subset=["match_date", "home_goals", "away_goals"])
    raw["home_goals"] = pd.to_numeric(raw["home_goals"], errors="coerce")
    raw["away_goals"] = pd.to_numeric(raw["away_goals"], errors="coerce")
    raw = raw.dropna(subset=["home_goals", "away_goals"])

    # Inicializar resolver con nombres canónicos del histórico
    init_resolver(raw)

    # Normalizar usando el resolver ya construido
    raw["home_team_norm"] = raw["home_team"].apply(normalize_team_name)
    raw["away_team_norm"] = raw["away_team"].apply(normalize_team_name)

    for col in ["B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
                "B365>2.5", "B365<2.5"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    log.info(f"Datos históricos cargados: {len(raw)} partidos")
    return raw.sort_values("match_date").reset_index(drop=True)


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
    path = os.path.join(DATA_RAW, "elo_ratings.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Club" in df.columns:
        df["team_norm"] = df["Club"].apply(normalize_team_name)
    return df


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
        f"{p}_n_matches_total": n_total,
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

def build_features_for_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    log.info("Cargando datos históricos para feature building...")
    historical    = load_historical_results()
    xg_data       = load_xg_data()
    match_summary = load_match_summary()
    elo_df        = load_elo()

    if historical.empty:
        log.error("Sin datos históricos. Imposible calcular features.")
        return pd.DataFrame()

    ref_date     = datetime.now()
    all_features = []

    for _, fixture in fixtures.iterrows():
        home = fixture["home_team"]
        away = fixture["away_team"]

        # Log de resolución de nombres para debugging
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

        all_features.append({
            "fixture_id":      fixture.get("fixture_id", 0),
            "league_id":       fixture.get("league_id", 0),
            "league_name":     fixture.get("league_name", ""),
            "home_team":       home,
            "away_team":       away,
            "match_date":      fixture.get("date", str(ref_date.date())),
            "n_home_matches":  home_stats.get("home_n_matches_total", 0),
            "n_away_matches":  away_stats.get("away_n_matches_total", 0),
            "elo_diff":        elo_diff,
            **home_stats,
            **away_stats,
            **h2h,
            **weather,
            **market,
        })

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

def build_training_dataset(historical: pd.DataFrame) -> pd.DataFrame:
    """Dataset walk-forward optimizado y sin data leakage."""
    log.info("Construyendo dataset walk-forward (optimizado, sin leakage)...")

    xg_data       = load_xg_data()
    match_summary = load_match_summary()
    elo_df        = load_elo()

    historical = historical.sort_values("match_date").reset_index(drop=True)
    start_idx  = max(200, len(historical) // 4)

    training_rows = []
    total_range   = len(historical) - start_idx

    for idx in range(start_idx, len(historical)):
        match    = historical.iloc[idx]
        ref_date = match["match_date"]
        past     = historical.iloc[:idx]

        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))
        if not home or not away:
            continue

        home_stats = compute_team_stats(home, True,  past, xg_data,
                                        match_summary, ref_date)
        away_stats = compute_team_stats(away, False, past, xg_data,
                                        match_summary, ref_date)
        h2h        = compute_h2h(home, away, past, ref_date)
        elo_diff   = get_elo_diff(home, away, elo_df)

        hg = int(match["home_goals"])
        ag = int(match["away_goals"])

        training_rows.append({
            "home_team":           home,
            "away_team":           away,
            "match_date":          ref_date,
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

        if (idx - start_idx) % 200 == 0:
            pct = (idx - start_idx) / total_range * 100
            log.info(f"  Training dataset: {pct:.0f}% ({idx}/{len(historical)})")

    df = pd.DataFrame(training_rows)
    if not df.empty:
        df = _add_derived_features(df)
        if "home_team_norm" not in df.columns:
            df["home_team_norm"] = df["home_team"].apply(normalize_team_name)
            df["away_team_norm"] = df["away_team"].apply(normalize_team_name)
        path = os.path.join(DATA_PROCESSED, "training_dataset.csv")
        df.to_csv(path, index=False)
        log.info(f"Dataset entrenamiento: {len(df)} partidos → {path}")
    return df