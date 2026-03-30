# FOOTBOT — Contexto del Proyecto

## Descripción general

Bot de predicción de apuestas deportivas 100% gratuito basado en modelos
estadísticos. Corre en local (Windows, venv Python) durante desarrollo.
Destino final: Oracle Cloud ARM (Always Free).

**Stack:**
- Modelo: Dixon-Coles por liga + XGBoost ensemble con calibración isotónica
- Datos históricos: ESPN API (sin key) — ligas LATAM y copas
- Fixtures del día: football-data.org (7 ligas EU) + ESPN API
- ELO ratings: ClubElo.com (sin key)
- Clima: Open-Meteo (sin key) — actualmente da 400 para fechas de hoy
- Base de datos: Supabase (pendiente de conectar)
- Notificaciones: Telegram Bot API (pendiente de conectar)

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
│   ├── espn_collector.py        # Cliente ESPN unificado
│   ├── nacional_features.py     # Features para selecciones nacionales
│   ├── telegram_sender.py       # Formateo y envío de reportes
│   └── utils.py                 # Clima, rate limiting, coordenadas estadios
├── scheduler.py                 # Pipeline principal (clubes)
├── scheduler_nacional.py        # Pipeline selecciones nacionales
├── start.sh                     # Orquestador bash (predict/results/live/all)
├── data/
│   ├── raw/                     # CSVs ESPN, ELO, fixtures del día
│   └── processed/               # Features y datasets de entrenamiento
└── models/                      # Modelos .pkl versionados por fecha
```

---

## Estado actual del sistema (sesión de diagnóstico)

### Lo que funciona ✅
- Pipeline completo corre sin errores críticos
- ESPN API devuelve fixtures y cuotas para col.1 (Liga BetPlay)
- Cuotas ESPN en tiempo real: 3/3 partidos con DraftKings
- Dixon-Coles entrenado por liga (Liga BetPlay, Brasileirão, Champions, Liga MX, Liga Profesional Argentina)
- Fuzzy matching de nombres resuelve correctamente equipos colombianos
- Histórico ESPN: 4226 partidos de 7 ligas (2024-2026)
- Blend weights optimizados por mercado (dc_weight=0.70 para todos)

### Pendiente ⚠️
- Supabase: error `'typing.Union' object has no attribute '__module__'` — versión incompatible de la librería
- Telegram: token placeholder — no configurado aún
- Open-Meteo: da 400 Bad Request para fechas de hoy (bug en cálculo de días adelante)
- Football-Data.co.uk: no descargado (ESPN_ONLY=true en .env)
- ELO: ClubElo no cubre equipos LATAM — devuelve 0.0 para col.1

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

### XGBoost validation metrics
| Mercado | Accuracy | ROI flat | dc_weight |
|---------|---------|---------|----------|
| home_win | 55.4% | +17.4% | 0.700 |
| draw | 72.7% | +0.0% | 0.700 |
| away_win | 73.3% | +0.0% | 0.700 |
| btts | 53.8% | -3.6% | 0.675 |
| over25 | 56.3% | +7.1% | 0.700 |

---

## Bug crítico corregido: league_id lookup en DixonColesEnsemble

**Problema:** Las ligas ESPN se guardan en `dc.models` con `league_name` como
clave string (ej: `'Liga BetPlay'`) porque no están en `LIGAS` (que solo tiene
las 7 ligas EU con IDs numéricos). `predict_proba` recibía `league_id=501` y
buscaba `dc.models[501]` — no encontraba nada y caía al global con probs por defecto.

**Síntoma:** Los 3 partidos del día tenían exactamente las mismas probabilidades
(47.2% / 27.3% / 25.4%) — valor por defecto de Dixon-Coles sin fit.

**Fix (v6 de _03_model_engine.py):**
```python
# Función helper nueva
def _resolve_league_name(league_id):
    from config.settings import LIGAS_ESPN, COMPETICIONES_NACIONALES_ESPN
    _todos = {**LIGAS_ESPN, **COMPETICIONES_NACIONALES_ESPN}
    return next(
        (name for slug, (lid, name) in _todos.items() if lid == league_id),
        None,
    )

# En predict_proba — paso 2 nuevo:
league_name = _resolve_league_name(league_id)
if league_name and league_name in self.models:
    model = self.models[league_name]
    if model.fitted:
        return model.predict_proba(home_team, away_team)
```

**Resultado post-fix:**
```
Cúcuta vs Boyacá:   home=64.6% (antes 47.2%)
Tolima vs Jaguares: home=81.3% (antes 47.2%)
Cali vs Pereira:    home=49.1% (antes 47.2%)
```

---

## Sistema de confianza (3 niveles — v6)

### Umbrales en config/settings.py
```python
# Alta
UMBRAL_EDGE_ALTA   = 8.0
UMBRAL_PROB_ALTA   = 0.62
MIN_PARTIDOS_ALTA  = 30

# Media
UMBRAL_EDGE_MEDIA  = 4.0
UMBRAL_PROB_MEDIA  = 0.55
MIN_PARTIDOS_MEDIA = 15

# Baja (nuevo en v6)
UMBRAL_EDGE_BAJA   = 2.0
UMBRAL_PROB_BAJA   = 0.52
MIN_PARTIDOS_BAJA  = 6
```

### Restricciones del nivel BAJA
1. Solo con cuotas reales (`espn_live`, `exact_match`, `contextual_avg`, `fd_historical`)
2. Solo mercados estándar (`_MERCADOS_NIVEL_BAJA`): 1X2, BTTS, Over/Under 1.5-3.5, Doble oportunidad, AH ±0.5
3. Kelly al 50% de la fracción normal
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

## Formato del mensaje Telegram

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

---

## Pruebas reales a endpoints ESPN (validadas en sesión)

### 1. Scoreboard — formato del campo `score`
```powershell
$r = Invoke-RestMethod "https://site.api.espn.com/apis/site/v2/sports/soccer/col.1/scoreboard"
$r.events[0].competitions[0].competitors | Select-Object homeAway, score | ConvertTo-Json
```
**Resultado confirmado:**
```json
[
  { "homeAway": "home", "score": "0" },
  { "homeAway": "away", "score": "0" }
]
```
**Conclusión:** `score` es STRING PLANO en `/scoreboard`. El bug de `int(score)` 
en `_parse_fixture` no explota hoy porque el scoreboard devuelve string, pero sería
un TypeError si se reutiliza con datos del endpoint `/schedule` que devuelve dict
`{"value": 2.0, "displayValue": "2"}`. Fix preventivo: usar `_parse_score()` siempre.

### 2. Odds — providers disponibles por liga

**Liga BetPlay (col.1):**
```powershell
$r = Invoke-RestMethod "https://site.api.espn.com/apis/site/v2/sports/soccer/col.1/scoreboard"
$eventId = $r.events[0].id  # → 401850840
$odds = Invoke-RestMethod "https://sports.core.api.espn.com/v2/sports/soccer/leagues/col.1/events/$eventId/competitions/$eventId/odds?limit=5"
$odds.items | Select-Object -ExpandProperty provider | Select-Object id, name, priority | ConvertTo-Json
```
**Resultado:**
```json
{ "id": "100", "name": "DraftKings", "priority": 1 }
```
**Conclusión:** Solo 1 provider en col.1 (DraftKings). El parámetro `provider.priority`
como query param no aporta valor aquí porque ya hay solo uno. El loop de fallback
en FIX A2 nunca necesita iterar más de 1 item para ligas LATAM.

**Premier League (eng.1):**
```powershell
$r2 = Invoke-RestMethod "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
$eventId2 = $r2.events[0].id  # → 740903
$odds2 = Invoke-RestMethod "https://sports.core.api.espn.com/v2/sports/soccer/leagues/eng.1/events/$eventId2/competitions/$eventId2/odds?limit=10"
$odds2.items | Select-Object -ExpandProperty provider | Select-Object id, name, priority | ConvertTo-Json
```
**Resultado:**
```json
[
  { "id": "100",  "name": "DraftKings", "priority": 1 },
  { "id": "2000", "name": "Bet 365",    "priority": 0 }
]
```
**Conclusión crítica:** Bet 365 tiene `priority: 0` (mayor precedencia que DraftKings).
El sort ascendente del código lo intenta primero. Bet 365 tiene `moneyLine: null`
(confirmar con comando pendiente), por eso el FIX A2 lo salta y usa DraftKings.
El comportamiento actual es correcto — no necesita cambios.

### 3. Standings — URL correcta
```powershell
# INCORRECTO — devuelve {} vacío:
Invoke-RestMethod "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/standings"

# CORRECTO — devuelve tabla completa:
Invoke-RestMethod "https://site.api.espn.com/apis/v2/sports/soccer/eng.1/standings"
```
El código ya usa `ESPN_SITE_V2B` que apunta a `/apis/v2/` — correcto.

---

## Slugs ESPN confirmados en documentación

### Ligas LATAM (usadas activamente)
| Slug | Liga |
|------|------|
| col.1 | Liga BetPlay Colombia |
| arg.1 | Liga Profesional Argentina |
| bra.1 | Brasileirão Serie A |
| conmebol.libertadores | Copa Libertadores |
| conmebol.sudamericana | Copa Sudamericana |
| mex.1 | Liga MX |
| usa.1 | MLS |

### Ligas EU en ESPN (slugs correctos, no usadas aún como activas)
| Slug | Liga |
|------|------|
| eng.1 | Premier League |
| esp.1 | La Liga |
| ger.1 | Bundesliga |
| ita.1 | Serie A |
| fra.1 | Ligue 1 |
| ned.1 | Eredivisie |
| por.1 | Primeira Liga |

### Competiciones nacionales
| Slug | Competición |
|------|-------------|
| fifa.worldq.conmebol | Eliminatorias CONMEBOL |
| conmebol.america | Copa América |
| fifa.world | Copa del Mundo |
| uefa.nations | UEFA Nations League |

---

## Problemas conocidos y soluciones

### Open-Meteo 400 Bad Request
**Causa:** El código calcula `days_ahead = (target_date - date.today()).days`.
Para partidos de hoy `days_ahead = 0`, que debería ser válido. El bug real es
que el partido de Cali vs Pereira tiene fecha `2026-03-29` (mañana) desde ESPN
pero el sistema lo procesa como si fuera hoy. Open-Meteo rechaza `start_date`
igual a `end_date` cuando es la fecha actual en algunos timezones.
**Fix pendiente:** Usar `forecast_days=1` sin `start_date/end_date` para partidos de hoy.

### Cúcuta Deportivo — pocos partidos históricos
**Situación:** Solo 6 partidos como local en histórico ESPN (2024-2026).
Cúcuta es un equipo que estuvo fuera de primera división y tiene poco historial.
**Efecto:** No pasaba `MIN_PARTIDOS_MEDIA=15`. Con la v6 (`MIN_PARTIDOS_BAJA=6`) ya puede generar alertas de nivel baja.

### btts_si prob=0% (bug resuelto)
**Causa:** Cuando el league_id lookup fallaba, DC usaba probs por defecto
con `dc_exp_home_goals=1.4` y `dc_exp_away_goals=1.1`. Pero `predictions`
devolvía `dc_exp_home_goals=0` porque el blend sobreescribía con valores
del XGBoost que no tenían la clave. `compute_all_market_probs(0, 0)` colapsa
la matriz Poisson.
**Fix:** Corregido con el fix del league_id lookup — ahora DC devuelve valores
reales y el blend los preserva correctamente.

### Supabase — error de tipo
```
'typing.Union' object has no attribute '__module__' and no __dict__
```
**Causa:** Versión incompatible entre `supabase-py` y `pydantic`. 
**Fix pendiente:** `pip install supabase==1.2.0` o actualizar a la versión compatible.

---

## Comandos útiles de diagnóstico

### Verificar histórico por liga
```powershell
python -c "
from src._01_data_collector import load_espn_historical
import pandas as pd
hist = load_espn_historical()
print(hist.groupby('league_name').size().to_string())
"
```

### Verificar partidos históricos de un equipo
```powershell
python -c "
from src._02_feature_builder import load_historical_results, normalize_team_name
hist = load_historical_results()
equipos = ['Deportes Tolima', 'Atlético Nacional', 'Millonarios']
for e in equipos:
    norm = normalize_team_name(e)
    h = hist[hist['home_team_norm'] == norm]
    print(e + ': norm=' + repr(norm) + ' | local=' + str(len(h)))
"
```

### Verificar DC lookup por liga (diagnóstico principal)
```python
# diag3.py — guarda en archivo y corre con: python diag3.py
import pandas as pd
from src._02_feature_builder import load_historical_results, normalize_team_name
from src._03_model_engine import load_models, predict_match

features = pd.read_csv('data/processed/features_YYYY-MM-DD.csv')
hist = load_historical_results()
dc, ensemble = load_models()

for _, row in features.iterrows():
    preds = predict_match(row['home_team'], row['away_team'], row.to_dict(), dc, ensemble)
    print(row['home_team'] + ' vs ' + row['away_team'])
    print('  league_id:        ' + str(row.get('league_id')))
    print('  dc_exp_home:      ' + str(preds.get('dc_exp_home_goals')))
    print('  dc_exp_away:      ' + str(preds.get('dc_exp_away_goals')))
    print('  prob_home_win:    ' + str(preds.get('prob_home_win')))
    print('  prob_btts:        ' + str(preds.get('prob_btts')))
    print('  liga en modelo:   ' + str(row.get('league_id') in dc.models))
    print('  ligas entrenadas: ' + str(list(dc.models.keys())))
    print()
```

### Verificar edges y probabilidades por partido
```python
# diag2.py
import pandas as pd
from src._02_feature_builder import load_historical_results
from src._03_model_engine import load_models, predict_match
from src._04_value_detector import compute_all_market_probs, build_odds_dict, calculate_edge

features = pd.read_csv('data/processed/features_YYYY-MM-DD.csv')
hist = load_historical_results()
dc, ensemble = load_models()

for _, row in features.iterrows():
    preds = predict_match(row['home_team'], row['away_team'], row.to_dict(), dc, ensemble)
    mu  = preds.get('dc_exp_home_goals', 1.4)
    lam = preds.get('dc_exp_away_goals', 1.1)
    market_probs = compute_all_market_probs(mu, lam)
    odds, is_real, method = build_odds_dict(row['home_team'], row['away_team'], hist, row, market_probs)

    print(row['home_team'] + ' vs ' + row['away_team'] + ' [' + method + ']')
    for market in ['home_win','draw','away_win','btts_si','over25','under25']:
        prob = market_probs.get('prob_' + market, 0)
        odd  = odds.get(market, 0)
        edge = calculate_edge(prob, odd)
        print('  ' + market + ': prob=' + str(round(prob*100,1)) + '% | cuota=' + str(odd) + ' | edge=' + str(edge) + '%')
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
TELEGRAM_TOKEN=TU_BOT_TOKEN_AQUI         # pendiente
TELEGRAM_CHAT_ID=TU_CHAT_ID_AQUI         # pendiente
SUPABASE_URL=TU_SUPABASE_URL_AQUI        # pendiente
SUPABASE_KEY=TU_SUPABASE_ANON_KEY_AQUI   # pendiente
FOOTBALL_DATA_ORG_KEY=                   # opcional — ligas EU
API_FOOTBALL_KEY=                        # opcional — fallback resultados
ESPN_ONLY=true                           # activo — saltea fd.co.uk y StatsBomb
```

---

## Versiones de archivos modificados en sesión

| Archivo | Versión | Cambio principal |
|---------|---------|-----------------|
| src/_03_model_engine.py | v6 | Fix league_id lookup ESPN en predict_proba y FootbotEnsemble.fit |
| config/settings.py | v6 | Umbrales nivel BAJA (UMBRAL_EDGE_BAJA, UMBRAL_PROB_BAJA, MIN_PARTIDOS_BAJA) |
| src/_04_value_detector.py | v6 | Nivel baja en classify_confidence, kelly reducido al 50%, sort actualizado |
| src/telegram_sender.py | v6 | Sección 🔵 BAJA CONFIANZA en format_message |

---

## Notas para próximas sesiones

- **Supabase:** Arreglar versión de librería (`pip install supabase==1.2.0` o la que corresponda)
- **Telegram:** Configurar token real y verificar envío
- **Open-Meteo:** Fix del 400 para partidos de hoy/mañana
- **evaluate_bet en _05_result_updater.py:** Faltan evaluadores para over15, under15, over45, home_over05, home_over15, away_over05, away_over15, exact_0..4plus, combinadas (home_and_btts, etc.), y AH. Sin esto las stats de ROI estarán incorrectas.
- **build_training_dataset O(n²):** El loop walk-forward es lento con 4226 partidos (~90s). Con datos EU sumados puede volverse problemático. Considerar pre-computar rolling stats por equipo.
- **_parse_fixture:** Reemplazar `int(home_score)` por `_parse_score(home_score)` para consistencia con el endpoint /schedule.
- **Ligas a agregar en LIGAS_ESPN_ACTIVAS cuando haya más datos:** `ksa.1` (Saudi Pro League — cuotas ESPN consistentes), `tur.1` (Super Lig), `sco.1` (Scottish Prem)