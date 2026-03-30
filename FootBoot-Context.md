# FOOTBOT — Contexto del Proyecto

## Descripción general

Bot de predicción de apuestas deportivas 100% gratuito basado en modelos
estadísticos. Corre en local (Windows, venv Python) durante desarrollo.
Destino final: Oracle Cloud ARM (Always Free).

**Stack:**
- Modelo: Dixon-Coles por liga + XGBoost ensemble con calibración isotónica
- Datos históricos: ESPN API (sin key) — ligas LATAM y copas
- Fixtures del día: football-data.org (7 ligas EU) + ESPN API
- ELO ratings: ClubElo.com (sin key) — no cubre equipos LATAM (devuelve 0.0)
- Clima: Open-Meteo (sin key) — bug activo para fechas de hoy (400 Bad Request)
- Base de datos: Supabase (✅ conectado — `supabase==2.3.0`, tablas creadas con DDL v2)
- Notificaciones: Telegram Bot API (✅ conectado — token y chat_id configurados)

---

## Estructura del proyecto

```
footbot/
├── config/
│   └── settings.py              # Constantes globales, umbrales, API keys
├── src/
│   ├── _01_data_collector.py    # Descarga fixtures, histórico, ELO
│   ├── _02_feature_builder.py   # Features por partido (forma, H2H, xG, clima)
│   ├── _03_model_engine.py      # Dixon-Coles + XGBoost + blend weights
│   ├── _04_value_detector.py    # Edge%, Kelly, clasificación de confianza
│   ├── _05_result_updater.py    # Cierre de predicciones con resultado real
│   ├── espn_collector.py        # Cliente ESPN unificado (v3)
│   ├── nacional_features.py     # Features para selecciones nacionales
│   ├── telegram_sender.py       # Formateo y envío de reportes (v6)
│   ├── supabase_client.py       # Cliente Supabase (v1 — insert/select predicciones)
│   └── utils.py                 # Clima, rate limiting, coordenadas estadios
├── scheduler.py                 # Pipeline principal (clubes)
├── scheduler_nacional.py        # Pipeline selecciones nacionales
├── start.sh                     # Orquestador bash (predict/results/live/all)
├── tests/
│   ├── conftest.py              # Fixtures compartidos (sample_historical, bets_df, etc.)
│   ├── test_conexiones.py       # Test integración: Supabase + Telegram (4/4 OK)
│   ├── test_data_collector.py
│   ├── test_espn_collector.py
│   ├── test_feature_builder.py
│   ├── test_model_engine.py
│   ├── test_result_updater.py
│   ├── test_telegram_sender.py
│   └── test_value_detector.py
├── data/
│   ├── raw/                     # CSVs ESPN, ELO, fixtures del día
│   └── processed/               # Features y datasets de entrenamiento
└── models/                      # Modelos .pkl versionados por fecha
```

---

## Estado actual del sistema

### Lo que funciona ✅
- Pipeline completo corre sin errores críticos
- ESPN API devuelve fixtures y cuotas para col.1 (Liga BetPlay)
- Cuotas ESPN en tiempo real: 3/3 partidos con DraftKings
- Dixon-Coles entrenado por liga (Liga BetPlay, Brasileirão, Champions, Liga MX, Liga Profesional Argentina)
- Fuzzy matching de nombres (rapidfuzz) resuelve correctamente equipos colombianos
- Histórico ESPN: 4226 partidos de 7 ligas (2024–2026)
- Blend weights optimizados por mercado (dc_weight=0.70 para todos)
- Fix league_id lookup: ligas ESPN resuelven correctamente (v6)
- Sistema de 3 niveles de confianza: alta, media, baja (v6)
- Suite de tests unitarios completa (pytest, 7 módulos)
- DDL Supabase generado y ejecutado (3 tablas con bug `'baja'` corregido — DDL v2)
- Supabase conectado: `supabase==2.3.0`, insert/delete verificados (tabla `predicciones`)
- Telegram conectado: token + chat_id reales, mensaje de prueba enviado OK
- Test de integración `test_conexiones.py`: 4/4 tests OK

### Pendiente ⚠️
- **Open-Meteo:** da 400 Bad Request para fechas de hoy (bug en cálculo de días adelante)
- **Football-Data.co.uk:** no descargado (ESPN_ONLY=true en .env)
- **ELO:** ClubElo no cubre equipos LATAM — devuelve 0.0 para col.1
- **`_05_result_updater.py`:** faltan evaluadores para over15, under15, over45, home_over05, home_over15, away_over05, away_over15, exact_0..4plus, combinadas (home_and_btts, etc.) y AH — ROI stats incompletas

---

## Datos históricos ESPN disponibles

| Liga | Slug | league_id | Partidos |
|------|------|-----------|---------|
| Liga BetPlay | col.1 | 501 | 1006 |
| Liga Profesional Argentina | arg.1 | 502 | 1050 |
| Brasileirão Serie A | bra.1 | 503 | 804 |
| Copa Libertadores | conmebol.libertadores | 511 | 256 |
| Copa Sudamericana | conmebol.sudamericana | 512 | 165 |
| Champions League | uefa.champions | 514 | 328 |
| Liga MX | mex.1 | 518 | 617 |

**Temporadas disponibles:** 2024, 2025, 2026

---

## Modelos entrenados (última versión: 20260328)

### Dixon-Coles por liga
Entrenado con L-BFGS-B (reemplazó SLSQP — más rápido con 100+ equipos).

| Liga | Equipos | Partidos | home_adv | rho | Converge |
|------|---------|---------|---------|-----|---------|
| Liga BetPlay | 23 | 688 | 0.361 | -0.096 | ✅ |
| Brasileirão | 27 | 551 | 0.390 | 0.024 | ✅ |
| Champions League | 54 | 315 | 0.302 | 0.092 | ✅ |
| Liga MX | 18 | 526 | 0.291 | -0.063 | ✅ |
| Liga Profesional Argentina | 32 | 835 | 0.283 | -0.146 | ✅ |
| Copa Libertadores | — | 146 | — | — | ❌ (< 200 mín) |
| Copa Sudamericana | — | 109 | — | — | ❌ (< 200 mín) |
| Global | — | 201 equipos | — | — | ❌ (> 80 máx) |

**Constantes clave (`_03_model_engine.py`):**
- `MIN_MATCHES_PER_LIGA = 200` — mínimo para entrenar DC por liga
- `MIN_MATCHES_GLOBAL_DC = 300` — mínimo para DC global
- `MAX_TEAMS_GLOBAL_DC = 80` — si hay más equipos, DC global se omite
- `DEFAULT_DC_WEIGHT = 0.35` — fallback si no hay optimización de blend

### XGBoost validation metrics
| Mercado | Accuracy | ROI flat | dc_weight |
|---------|---------|---------|----------|
| home_win | 55.4% | +17.4% | 0.700 |
| draw | 72.7% | +0.0% | 0.700 |
| away_win | 73.3% | +0.0% | 0.700 |
| btts | 53.8% | -3.6% | 0.675 |
| over25 | 56.3% | +7.1% | 0.700 |

> **Nota:** `draw` y `away_win` con accuracy alta y ROI 0.0% sugieren posible
> clase mayoritaria o ausencia de edge real en esos mercados. Revisar distribución
> de clases antes de confiar en estas métricas.

---

## Bug crítico corregido: league_id lookup en DixonColesEnsemble (v6)

**Problema:** Las ligas ESPN se guardan en `dc.models` con `league_name` como
clave string (ej: `'Liga BetPlay'`) porque no están en `LIGAS` (que solo tiene
las 7 ligas EU con IDs numéricos). `predict_proba` recibía `league_id=501` y
buscaba `dc.models[501]` — no encontraba nada y caía al global con probs por defecto.

**Síntoma:** Los 3 partidos del día tenían exactamente las mismas probabilidades
(47.2% / 27.3% / 25.4%) — valor por defecto de Dixon-Coles sin fit.

**Fix (`_03_model_engine.py` v6):**
```python
def _resolve_league_name(league_id: int | str) -> str | None:
    from config.settings import LIGAS_ESPN, COMPETICIONES_NACIONALES_ESPN
    _todos = {**LIGAS_ESPN, **COMPETICIONES_NACIONALES_ESPN}
    return next(
        (name for slug, (lid, name) in _todos.items() if lid == league_id),
        None,
    )

# En predict_proba — paso 2:
league_name = _resolve_league_name(league_id)
if league_name and league_name in self.models:
    model = self.models[league_name]
    if model.fitted:
        return model.predict_proba(home_team, away_team)
```

El mismo fix se aplica en `FootbotEnsemble.fit` al pre-calcular probs DC para
la optimización de blend weights.

**Resultado post-fix:**
```
Cúcuta vs Boyacá:   home=64.6% (antes 47.2%)
Tolima vs Jaguares: home=81.3% (antes 47.2%)
Cali vs Pereira:    home=49.1% (antes 47.2%)
```

---

## Sistema de confianza (3 niveles — v6)

### Umbrales en `config/settings.py`
```python
# Alta
UMBRAL_EDGE_ALTA   = 8.0
UMBRAL_PROB_ALTA   = 0.62
MIN_PARTIDOS_ALTA  = 30

# Media
UMBRAL_EDGE_MEDIA  = 4.0
UMBRAL_PROB_MEDIA  = 0.55
MIN_PARTIDOS_MEDIA = 15

# Baja
UMBRAL_EDGE_BAJA   = 2.0
UMBRAL_PROB_BAJA   = 0.52
MIN_PARTIDOS_BAJA  = 6

# Nacionales (menos partidos disponibles)
MIN_PARTIDOS_ALTA_NACIONAL  = 15
MIN_PARTIDOS_MEDIA_NACIONAL = 8

# ESPN sin cuotas reales (más conservador)
UMBRAL_EDGE_ALTA_ESPN  = 10.0
UMBRAL_EDGE_MEDIA_ESPN =  6.0
```

### Restricciones del nivel BAJA
1. Solo con cuotas reales (`espn_live`, `exact_match`, `contextual_avg`, `fd_historical`)
2. Solo mercados estándar (`_MERCADOS_NIVEL_BAJA`): 1X2, BTTS, Over/Under 1.5–3.5, Doble oportunidad, AH ±0.5
3. Kelly al 50% de la fracción normal (`kelly_fraction(..., confidence="baja")`)
4. Mensaje Telegram incluye `"señal débil — stake reducido"`

### Mercados bloqueados con model_implied
```python
_MERCADOS_BLOQUEADOS_MODEL_IMPLIED = {
    "home_win", "draw", "away_win",
    "over25", "under25",
    "double_1x", "double_x2", "double_12",
}
```

---

## Cuotas ESPN — schema confirmado (Core API /odds)

```
ESPN Core API: /v2/sports/soccer/leagues/{slug}/events/{id}/competitions/{id}/odds

Campos relevantes por item:
  provider.id / name / priority  → Bet365 priority=0, DraftKings priority=1
  homeTeamOdds.moneyLine         → 1X2 local (americano)
  awayTeamOdds.moneyLine         → 1X2 visitante (americano)
  drawOdds.moneyLine             → empate (solo fútbol)
  overUnder                      → línea de goles totales (float: 2.5)
  overOdds / underOdds           → cuotas Over/Under (americano)
  spread                         → handicap de línea (ej: -0.5 para local)
  homeTeamOdds.spreadOdds        → cuota spread local
  awayTeamOdds.spreadOdds        → cuota spread visitante
  open.over.value / open.spread.home.line → apertura (movimiento de línea)
```

**Notas confirmadas:**
- Liga BetPlay (col.1): solo 1 provider (DraftKings, priority=1)
- Premier League (eng.1): Bet365 (priority=0) tiene `moneyLine: null` → el código lo salta y usa DraftKings correctamente
- El sort ascendente por priority es correcto — no necesita cambios
- `_to_decimal` normaliza coma decimal antes de `float()` (FIX A3 en espn_collector.py)

---

## Mercados soportados (`_04_value_detector.py`)

| Categoría | Mercados |
|-----------|---------|
| 1X2 | home_win, draw, away_win |
| Doble oportunidad | double_1x, double_x2, double_12 |
| Goles totales | over05–over45, under05–under45 |
| BTTS | btts_si, btts_no |
| Por equipo | home_over05/15, away_over05/15 (+ unders) |
| Goles exactos | exact_0, exact_1, exact_2, exact_3, exact_4plus |
| Combinadas | home_and_btts, draw_and_btts, away_and_btts |
| Asian Handicap | ah_home/away ±0.5, ah_home/away -1 |

---

## Columnas ESPN en features_df (post-enrich)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| espn_odds_available | bool | True si hay cuotas válidas |
| espn_odds_home | float | Cuota decimal local 1X2 |
| espn_odds_draw | float | Cuota decimal empate |
| espn_odds_away | float | Cuota decimal visitante |
| espn_odds_provider | str | Nombre del provider |
| espn_total_line | float | Línea O/U (ej: 2.5) |
| espn_over_odds | float | Cuota decimal Over |
| espn_under_odds | float | Cuota decimal Under |
| espn_spread_line | float | Handicap de línea (ej: -0.5) |
| espn_spread_home_odds | float | Cuota spread local |
| espn_spread_away_odds | float | Cuota spread visitante |
| espn_open_total_line | float | Línea de apertura |
| espn_open_spread_home | float | Spread de apertura |

---

## Formato del mensaje Telegram (v6)

```
⚽ FOOTBOT · Sábado 28 Mar
_N apuestas_

🟢 ALTA CONFIANZA
────────────────────────────
*Home vs Away*
🏆 Liga BetPlay
📈 Esperados: 2.5 – 0.5 (total 3.0)
_🏆 Resultado_
  📌 *Victoria Home*
     Prob 81.3% · Cuota 1.339 · Edge +8.93%
     Kelly 1.2% | forma local 2.2pts/PJ

🟡 MEDIA CONFIANZA
────────────────────────────
...

🔵 BAJA CONFIANZA
_Edge reducido — señales exploratorias_
────────────────────────────
  📌 *Over 2.5 goles*
     Prob 57.3% · Cuota 1.657 · Edge +2.1%
     Kelly 0.3% | señal débil — stake reducido

────────────────────────────
📊 Rendimiento (N apuestas): Acierto X% · ROI +Y%
  🟢 Alta: X% · ROI +Y%
  🟡 Media: X% · ROI +Y%
  🔵 Baja: X% · ROI +Y%

⚠️ Modelo estadístico experimental. Apuesta con responsabilidad.
```

**Grupos de mercados en el mensaje:**
`resultado` | `goles` | `btts` | `por_equipo` | `exactos` | `combinadas` | `handicap`

---

## Standings ESPN — URL correcta

```powershell
# INCORRECTO — devuelve {} vacío:
Invoke-RestMethod "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/standings"

# CORRECTO — devuelve tabla completa:
Invoke-RestMethod "https://site.api.espn.com/apis/v2/sports/soccer/eng.1/standings"
```
El código usa `ESPN_SITE_V2B` que apunta a `/apis/v2/` — correcto.

---

## Slugs ESPN confirmados

### Ligas LATAM activas (LIGAS_ESPN_ACTIVAS)
| Slug | Liga | league_id |
|------|------|-----------|
| col.1 | Liga BetPlay Colombia | 501 |
| arg.1 | Liga Profesional Argentina | 502 |
| bra.1 | Brasileirão Serie A | 503 |
| conmebol.libertadores | Copa Libertadores | 511 |
| conmebol.sudamericana | Copa Sudamericana | 512 |
| uefa.champions | Champions League | 514 |
| mex.1 | Liga MX | 518 |

### Ligas EU en ESPN (slugs correctos, no usadas como activas aún)
| Slug | Liga |
|------|------|
| eng.1 | Premier League |
| esp.1 | La Liga |
| ger.1 | Bundesliga |
| ita.1 | Serie A |
| fra.1 | Ligue 1 |
| ned.1 | Eredivisie |
| por.1 | Primeira Liga |

### Competiciones nacionales (COMPETICIONES_NACIONALES_ESPN)
| Slug | Competición | league_id |
|------|-------------|-----------|
| fifa.worldq.conmebol | Eliminatorias CONMEBOL | 361 |
| conmebol.america | Copa América | 271 |
| fifa.world | Copa del Mundo | 1 |
| uefa.nations | UEFA Nations League | 5 |

### Ligas a agregar cuando haya más datos
`ksa.1` (Saudi Pro League), `tur.1` (Super Lig), `sco.1` (Scottish Prem)

---

## Problemas conocidos y soluciones

### Open-Meteo 400 Bad Request
**Causa:** Para partidos de hoy `days_ahead = 0`, pero Open-Meteo rechaza `start_date`
igual a `end_date` cuando es la fecha actual en algunos timezones. Además hay
partidos con fecha ESPN de mañana que se procesan como si fueran hoy.
**Fix pendiente:** Usar `forecast_days=1` sin `start_date/end_date` para partidos de hoy.
**Nota:** fechas > 16 días devuelven valores neutros sin llamar a la API (ya implementado).

### Supabase — error de versión
```
'typing.Union' object has no attribute '__module__'
```
**Causa:** Versión incompatible entre `supabase-py` y `pydantic`.
**Fix pendiente:** `pip install supabase==1.2.0` (o la versión compatible confirmada).

### Supabase DDL — nivel baja no incluido (BUG)
**Causa:** El CHECK de la tabla `predicciones` solo permite `('alta','media')`.
Con la v6 se emiten apuestas de nivel `'baja'` que Supabase rechazaba (error 23514).
**Fix aplicado en DDL v2:**
```sql
-- Anterior (incorrecto):
CHECK (confianza IN ('alta','media'))
-- DDL v2 (correcto — ejecutado en Supabase SQL Editor):
CHECK (confianza IN ('alta','media','baja'))
```
✅ **Resuelto** — insert nivel `'baja'` verificado en `test_conexiones.py` (id=2, registro eliminado).

### Cúcuta Deportivo — pocos partidos históricos
Solo 6 partidos como local. Con `MIN_PARTIDOS_BAJA=6` ya genera alertas de nivel baja.

### btts_si prob=0% (resuelto en v6)
Causado por el bug de league_id lookup — corregido junto con el fix principal.

### `_parse_fixture` — score como string vs dict (FIX A1)
`/scoreboard` devuelve `score` como string plano; `/schedule` lo devuelve como
dict `{"value": 3.0, "displayValue": "3"}`. En entornos ES/COL `value` tiene
coma decimal ("3,0") que hace fallar `float()`. `_parse_score()` usa
`displayValue` como fuente primaria — ya corregido.

### `_safe_int` — coma decimal en standings (FIX B1)
ESPN standings devuelve stats como `"27,0"` en entornos ES/COL.
`int("27,0")` explota. `_safe_int()` convierte con `str(val).replace(",", ".")`.

---

## Tests unitarios

```
tests/
├── conftest.py               # sample_historical (240p), sample_training_df,
│                             # sample_fixture_row, sample_predictions, sample_bets_df
├── test_data_collector.py    # _make_session, get_fixtures_today, download_elo_ratings
├── test_espn_collector.py    # _parse_score (FIX A1), _to_decimal (FIX A3),
│                             # _safe_int (FIX B1), _norm_status, ESPNClient
├── test_feature_builder.py   # _clean_name, TeamNameResolver, exponential_weight,
│                             # compute_team_stats, compute_h2h, get_elo_diff
├── test_model_engine.py      # _resolve_league_name (FIX v6), DixonColesModel,
│                             # DixonColesEnsemble, blend_predictions,
│                             # FootbotEnsemble (@slow)
├── test_result_updater.py    # evaluate_bet (parametrizado, todos los mercados),
│                             # DDL Supabase (incluye test del BUG 'baja')
├── test_telegram_sender.py   # format_message (3 niveles), format_date_es
└── test_value_detector.py    # _poisson_matrix, compute_all_market_probs,
                              # calculate_edge, kelly_fraction, classify_confidence,
                              # build_odds_dict, analyze_fixture
```

**Ejecución:**
```powershell
pytest tests/                   # todos los tests rápidos
pytest tests/ -m slow           # incluye entrenamiento de modelos
pytest tests/ --cov=src         # con cobertura
```

**BUG confirmado en tests:** `test_supabase_ddl_includes_baja` falla hasta
que se corrija el DDL.

---

## Comandos útiles de diagnóstico

### Verificar histórico por liga
```powershell
python -c "
from src._01_data_collector import load_espn_historical
hist = load_espn_historical()
print(hist.groupby('league_name').size().to_string())
"
```

### Verificar DC lookup por liga
```python
# diag3.py
import pandas as pd
from src._02_feature_builder import load_historical_results, normalize_team_name
from src._03_model_engine import load_models, predict_match

features = pd.read_csv('data/processed/features_YYYY-MM-DD.csv')
dc, ensemble = load_models()

for _, row in features.iterrows():
    preds = predict_match(row['home_team'], row['away_team'], row.to_dict(), dc, ensemble)
    print(row['home_team'] + ' vs ' + row['away_team'])
    print('  league_id:        ' + str(row.get('league_id')))
    print('  dc_exp_home:      ' + str(preds.get('dc_exp_home_goals')))
    print('  prob_home_win:    ' + str(preds.get('prob_home_win')))
    print('  ligas entrenadas: ' + str(list(dc.models.keys())))
    print()
```

### Verificar edges y probabilidades
```python
# diag2.py
import pandas as pd
from src._03_model_engine import load_models, predict_match
from src._04_value_detector import compute_all_market_probs, build_odds_dict, calculate_edge
from src._02_feature_builder import load_historical_results

features  = pd.read_csv('data/processed/features_YYYY-MM-DD.csv')
historical = load_historical_results()
dc, ensemble = load_models()

for _, row in features.iterrows():
    preds = predict_match(row['home_team'], row['away_team'], row.to_dict(), dc, ensemble)
    mu  = preds.get('dc_exp_home_goals', 1.4)
    lam = preds.get('dc_exp_away_goals', 1.1)
    market_probs = compute_all_market_probs(mu, lam)
    odds, is_real, method = build_odds_dict(row['home_team'], row['away_team'],
                                             historical, row, market_probs)
    print(row['home_team'] + ' vs ' + row['away_team'] + ' [' + method + ']')
    for market in ['home_win','draw','away_win','btts_si','over25','under25']:
        prob = market_probs.get('prob_' + market, 0)
        odd  = odds.get(market, 0)
        edge = calculate_edge(prob, odd)
        print(f'  {market}: prob={round(prob*100,1)}% | cuota={odd} | edge={edge}%')
    print()
```

### Eliminar modelos y forzar reentrenamiento
```powershell
python -c "
import os, glob
for f in glob.glob('models/*.pkl'):
    os.remove(f)
    print('Eliminado: ' + f)
"
python scheduler.py
```

---

## Variables de entorno (.env)

```
TELEGRAM_TOKEN=<configurado>             # ✅ real
TELEGRAM_CHAT_ID=<configurado>           # ✅ real
SUPABASE_URL=<configurado>               # ✅ real
SUPABASE_KEY=<configurado>               # ✅ real (sb_secret_tS...)
FOOTBALL_DATA_ORG_KEY=                   # opcional — ligas EU (10 req/min free tier)
API_FOOTBALL_KEY=                        # opcional — fallback resultados
ESPN_ONLY=true                           # activo — saltea fd.co.uk y StatsBomb
```

---

## URLs base de APIs

```python
ESPN_SITE_V2  = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_SITE_V2B = "https://site.api.espn.com/apis/v2/sports/soccer"       # standings
ESPN_CORE_V2  = "https://sports.core.api.espn.com/v2/sports/soccer/leagues"
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
CLUBELO_URL   = "http://api.clubelo.com"
FOOTBALL_DATA_ORG_URL = "https://api.football-data.org/v4"
```

---

## Versiones de archivos

| Archivo | Versión | Cambio principal |
|---------|---------|-----------------|
| src/_03_model_engine.py | v6 | Fix league_id lookup ESPN en predict_proba y FootbotEnsemble.fit |
| config/settings.py | v6 | Umbrales nivel BAJA, ESPN_ONLY flag |
| src/_04_value_detector.py | v6 | Nivel baja en classify_confidence, kelly reducido al 50% |
| src/telegram_sender.py | v6 | Sección 🔵 BAJA CONFIANZA en format_message |
| src/espn_collector.py | v3 | FIX A1 _parse_score, FIX A2 providers, FIX A3 _to_decimal, FIX B1 _safe_int, FIX D1/D2 seasons |
| src/supabase_client.py | v1 | Cliente Supabase — insert/select/delete predicciones |
| supabase_ddl_v2.sql | v2 | CHECK corregido: `'baja'` incluido en constraint confianza |
| src/_01_data_collector.py | v3 | Retry backoff para WinError 10054, fusión fd.org + ESPN |
| src/_02_feature_builder.py | v4 | Fusión histórico EU + ESPN, TeamNameResolver, leakage fix |
| src/nacional_features.py | v4 | Eliminado import roto de nacional_collector |

---

## Notas para próximas sesiones

1. ~~**Supabase:** `pip install supabase==1.2.0` y corregir DDL (`'baja'` en CHECK)~~ ✅ **RESUELTO** — `supabase==2.3.0`, DDL v2 ejecutado, 4/4 tests OK
2. ~~**Telegram:** Configurar token real y verificar envío~~ ✅ **RESUELTO** — token y chat_id configurados, mensaje de prueba enviado
3. **Open-Meteo:** Usar `forecast_days=1` sin `start_date/end_date` para partidos de hoy
4. **`evaluate_bet`:** Añadir evaluadores para over15, under15, over45, by-team markets, exactos, combinadas y AH — sin esto el ROI tracking es incompleto
5. **`build_training_dataset` O(n²):** ~90s con 4226 partidos. Pre-computar rolling stats por equipo antes de agregar ligas EU
6. **`_parse_fixture`:** Reemplazar `int(home_score)` por `_parse_score(home_score)` para consistencia con el endpoint `/schedule`
7. **ELO LATAM:** ClubElo no cubre equipos colombianos/argentinos (devuelve 0.0). Considerar FBref Club Rankings o ELO propio calculado desde histórico ESPN
8. **Copa Libertadores / Sudamericana:** < 200 partidos → sin DC propio. El modelo global tampoco converge (> 80 equipos). Pendiente solución arquitectural (DC cross-liga o umbral dinámico)
9. **Tests @slow:** `FootbotEnsemble.fit` no corre en CI normal — marcar y ejecutar explícitamente con `pytest -m slow`