# Метод прогнозирования нагрузки информационных систем
## Полная инструкция по установке и запуску

---

## Структура проекта

```
project/
├── config.yaml                    ← единый конфигурационный файл (все параметры)
├── requirements.txt               ← Python-зависимости
├── Dockerfile.webapp              ← образ Flask-приложения + API
├── Dockerfile.controller          ← образ управляющего контроллера
│
├── app/
│   ├── main.py                    ← Flask: 9 маршрутов, 3 класса запросов
│   └── api.py                     ← REST API для веб-интерфейса (порт 5001)
│
├── predictor/
│   ├── preprocessor.py            ← МКР + STL + нормализация (формулы 3.8–3.10)
│   ├── model.py                   ← GRUQuantileNet + GRUTrainer (таблица 4.2)
│   └── forecaster.py              ← тренд + сезонность + GRU (формулы 3.11–3.15)
│
├── controller/
│   ├── decision.py                ← гистерезис + Kubernetes PATCH (форм. 3.16–3.18)
│   ├── prometheus_collector.py    ← PromQL-запросы для m_t и φ_t
│   └── control_loop.py            ← главный управляющий цикл Δt=5 мин
│
├── frontend/                      ← Vue.js 3 интерфейс
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.js
│       ├── App.vue                ← навигация, шапка, статус системы
│       └── views/
│           ├── ViewMonitor.vue    ← мониторинг в реальном времени
│           ├── ViewLoad.vue       ← управление нагрузкой по классам
│           ├── ViewTraining.vue   ← обучение модели, динамика потерь
│           └── ViewCompare.vue    ← сравнение методов, таблицы 4.4–4.11
│
├── k8s/
│   └── manifests.yaml             ← все ресурсы Kubernetes
│
├── locust/
│   └── locustfile.py              ← сценарии нагрузочного тестирования
│
├── tests/
│   ├── test_experiments.py        ← все эксперименты глав 4.2–4.3
│   └── baselines.py               ← SARIMA, Prophet, GRU, LSTM, HPA + генераторы
│
├── scripts/
│   └── preprocess_alibaba.py      ← предобработка датасета Alibaba
│
└── data/                          ← создать вручную, положить датасет
    └── alibaba_cluster_trace_2018.csv
```

---

## Шаг 1. Требования к системе

| Инструмент | Версия | Где скачать |
|------------|--------|-------------|
| Docker Desktop | ≥ 4.25 | https://www.docker.com/products/docker-desktop |
| kind | 0.20 | https://kind.sigs.k8s.io/docs/user/quick-start/#installation |
| kubectl | ≥ 1.28 | https://kubernetes.io/docs/tasks/tools/ |
| Python | 3.10 | https://www.python.org/downloads/ |
| Node.js | ≥ 18 | https://nodejs.org/ |

**Минимальные ресурсы машины:** 8 ГБ ОЗУ, 4 ядра CPU, 15 ГБ свободного диска.

Проверьте что всё установлено:
```bash
docker --version          # Docker version 24.x.x
kind --version            # kind v0.20.0
kubectl version --client
python --version          # Python 3.10.x
node --version            # v18.x.x или выше
```

---

## Шаг 2. Установка Python-зависимостей - 1 терминал

```bash
cd project/

# Создать виртуальное окружение
python -m venv venv

# Активировать
#source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows (cmd)
 venv\Scripts\Activate.ps1        # Windows (PowerShell)

# Установить все зависимости (~10 минут из-за PyTorch)
pip install -r requirements.txt
```

**Что устанавливается и зачем:**

| Пакет | Назначение |
|-------|------------|
| `flask==3.0.3` | Веб-приложение (9 маршрутов, 3 класса запросов) |
| `flask-cors==4.0.1` | CORS для запросов с Vue.js фронтенда |
| `prometheus-flask-exporter==0.23.1` | Автоматический экспорт метрик HTTP в Prometheus |
| `sqlalchemy==2.0.30` | ORM для работы с PostgreSQL |
| `psycopg2-binary==2.9.9` | Драйвер PostgreSQL |
| `pyyaml==6.0.1` | Чтение config.yaml |
| `numpy==1.26.4` | Численные операции |
| `scipy==1.13.0` | Линейная регрессия МНК для тренда |
| `pandas==2.2.2` | Обработка датасетов |
| `statsmodels==0.14.2` | STL-декомпозиция |
| `torch==2.1.2` | Нейросетевая модель GRU (CPU; ~800 МБ) |
| `prophet==1.1.5` | Базовый метод сравнения |
| `kubernetes==28.1.0` | Клиент Kubernetes API (PATCH spec.replicas) |
| `locust==2.20.0` | Генерация нагрузки по сценариям |
| `pytest==8.2.0`, `pytest-html==4.1.1` | Запуск экспериментов, HTML-отчёт |

---

## Шаг 3. Установка зависимостей фронтенда - 2 терминал

```bash
cd project/frontend/
npm install
# Скачивает: Vue 3, Vue Router, Pinia, Axios, Chart.js, vue-chartjs, Vite
# папка node_modules
```

---

## Шаг 4. Создание кластера Kubernetes - можно в том же, без venv

```bash
kind create cluster --name webapp-cluster --config - <<'EOF'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
  - role: worker
EOF

# Проверить
kubectl cluster-info --context kind-webapp-cluster
kubectl get nodes
```

Ожидаемый вывод:
```
NAME                          STATUS   ROLES
webapp-cluster-control-plane  Ready    control-plane
webapp-cluster-worker         Ready    <none>
webapp-cluster-worker2        Ready    <none>
webapp-cluster-worker3        Ready    <none>
```

---

## Шаг 5. Сборка Docker-образов и загрузка в кластер - можно новый

```bash
cd project/

# Собрать образы
docker build -f Dockerfile.webapp -t webapp:latest .
docker build -f Dockerfile.controller -t controller:latest .

# Загрузить в кластер kind (обязательно! иначе ImagePullBackOff)
kind load docker-image webapp:latest --name webapp-cluster
kind load docker-image controller:latest --name webapp-cluster
```

> Первая сборка: 5–10 минут. Повторная — быстро (кэш Docker).

---

## Шаг 6. Развёртывание в Kubernetes  - там же 

```bash
# Применить все ресурсы: PostgreSQL, webapp (2 реплики), controller, Prometheus
kubectl apply -f k8s/manifests.yaml

# Ждать пока все поды станут Running (2–3 минуты)
kubectl get pods -w
```

Ожидаемый вывод:
```
NAME                       READY   STATUS
controller-xxx             1/1     Running
postgres-xxx               1/1     Running
prometheus-xxx             1/1     Running
webapp-xxx-1               1/1     Running
webapp-xxx-2               1/1     Running
```

```bash
# Проверить число реплик webapp
kubectl get deployment webapp

# Смотреть логи контроллера в реальном времени
kubectl logs -f deployment/controller
```

---

## Шаг 7. Проброс портов

Открыть **отдельный терминал**, выполнить и **не закрывать**:

```bash
kubectl port-forward service/webapp 8080:80 &
kubectl port-forward deployment/webapp 5001:5001 &
kubectl port-forward service/prometheus 9090:9090 &
```

Проверка:
```bash
curl http://localhost:8080/health
# {"status": "ok"}

curl http://localhost:5001/api/status
# {"metrics": {...}, "replicas": {...}, "forecast": {...}}
```

---

## Шаг 8. Запуск веб-интерфейса

```bash
cd project/frontend/
npm run dev
```

Открыть браузер: **http://localhost:3000**

**Четыре раздела интерфейса:**

| Раздел | Что показывает |
|--------|----------------|
| **Мониторинг** | График cpu_t + прогноз с доверительным интервалом, 8 подов (активные/неактивные), прогноз t+1/t+2/t+3, гистограмма RPS по классам, история реплик |
| **Нагрузка** | Ползунки пользователей на каждый класс (0–30), предустановленные сценарии одним кликом, ожидаемый φ_t, таблица ограничений СУБД |
| **Обучение** | Запуск обучения GRU, живой график train/val loss, архитектура модели, разбивка входного вектора d_in=69, квантильная функция потерь |
| **Сравнение** | Таблицы 4.4–4.11 из диссертации, гистограмма MAE, scatter SLA vs утилизация |

> **Без Kubernetes** — интерфейс работает в **демо-режиме**: API генерирует
> синтетические данные с суточной сезонностью и периодической сменой классов.
> Все графики и таблицы отображаются корректно.

---

## Шаг 9. Запуск без Kubernetes (только демо-режим)

Если кластер Kubernetes не нужен — только интерфейс и демо-данные:

```bash
# Терминал 1
cd project/
source venv/bin/activate
python app/api.py
# Запускается на http://0.0.0.0:5001

# Терминал 2
cd project/frontend/
npm run dev
# Открыть http://localhost:3000
```

---

## Шаг 10. Датасет Alibaba Cluster Trace 2018 (для научных тестов)

```bash
mkdir -p project/data

# Скачать с официального репозитория Alibaba:
# https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018
# Файл: container_usage.tar.gz (~300 МБ)

# После скачивания в папку project/data/:
cd project/data/
tar -xzf container_usage.tar.gz

# Предобработка (агрегация до Δt=5 мин, нормировка в [0,1])
cd project/
python scripts/preprocess_alibaba.py
# Результат: data/alibaba_cluster_trace_2018.csv (2304 строки = 8 суток)
```

> Без датасета тесты с ним будут пропущены (`pytest.skip`).
> Все синтетические тесты работают без скачивания.

---

## Шаг 11. Запуск экспериментальных тестов

```bash
cd project/
venv\Scripts\activate
mkdir -p results

# Полный прогон всех тестов (20–40 минут)
python -m pytest tests/test_experiments.py -v -s \
  --html=results/report.html \
  2>&1 | tee results/output.txt
```

**Быстрые варианты по отдельным таблицам:**

```bash
# Unit-тесты формул, без обучения (~30 секунд)
python -m pytest tests/test_experiments.py -k "Unit" -v -s

# Таблица 4.11: вычислительное время (~2 минуты)
python -m pytest tests/test_experiments.py -k "TestComputationTime" -v -s

# Таблица 4.9: вклад признаков φ_t (~10 минут)
python -m pytest tests/test_experiments.py -k "TestMixedLoadScenario" -v -s

# Таблица 4.10: анализ абляции (~15 минут)
python -m pytest tests/test_experiments.py -k "TestAblation" -v -s

# Таблица 4.7: зависимость от горизонта h (~15 минут)
python -m pytest tests/test_experiments.py -k "TestHorizonDependence" -v -s

# Таблица 4.8: влияние дообучения (~10 минут)
python -m pytest tests/test_experiments.py -k "TestRetrainingEffect" -v -s

# Таблицы 4.4–4.6: точность и управление (требует Alibaba или синт. данные, ~20 минут)
python -m pytest tests/test_experiments.py -k "TestForecast or TestManagement" -v -s
```

**Как читать вывод и заполнять таблицы:**

Найдите в `results/output.txt` строки `[METRIC]`:
```
[METRIC] ТЕСТ: Alibaba | Разработанный метод | MAE=0.0582±0.0041 | ...
[METRIC] MGMT ТЕСТ: Alibaba | SLA_VIOLATIONS=3.2% | AVG_UTIL=70.8% | SCALE_OPS=61
[METRIC] ТЕСТ: Аблация без STL | Полный метод MAE=0.058 | Без STL MAE=0.083 | Ухудшение=43.1%
[METRIC] Вычислительное время | Среднее=80.4мс ± 11.8мс | Доля от Δt=0.027%
```

Перенесите числа в соответствующие таблицы диссертации.

---

## Шаг 12. Нагрузочные тесты Locust

```bash
# Убедитесь что port-forward запущен (Шаг 7)
venv\Scripts\activate

# Сценарий 1: только вычислительные запросы (класс 1)
locust -f locust/locustfile.py \
  --headless --users 20 --spawn-rate 5 --run-time 30m \
  --host http://localhost:8080 --only-use ComputeUser

# Сценарий 2: только запросы к БД (класс 2)
locust -f locust/locustfile.py \
  --headless --users 20 --spawn-rate 5 --run-time 30m \
  --host http://localhost:8080 --only-use DBUser

# Сценарий 3: только память (класс 3)
locust -f locust/locustfile.py \
  --headless --users 20 --spawn-rate 5 --run-time 30m \
  --host http://localhost:8080 --only-use MemoryUser

# Смешанный сценарий (все классы, параграф 4.2)
locust -f locust/locustfile.py \
  --headless --users 30 --spawn-rate 5 --run-time 120m \
  --host http://localhost:8080

# С веб-интерфейсом Locust
locust -f locust/locustfile.py --host http://localhost:8080
# → открыть http://localhost:8089
```

---

## Полная шпаргалка (все команды подряд)

```bash
# ── Один раз при установке ────────────────────────────────────────────────
cd project/
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # ~10 мин
cd frontend/ && npm install && cd ..

kind create cluster --name webapp-cluster --config - <<'EOF'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
  - role: worker
EOF

docker build -f Dockerfile.webapp -t webapp:latest .
docker build -f Dockerfile.controller -t controller:latest .
kind load docker-image webapp:latest --name webapp-cluster
kind load docker-image controller:latest --name webapp-cluster
kubectl apply -f k8s/manifests.yaml
kubectl get pods -w                      # ждём Running

# ── Каждый раз при запуске сессии ─────────────────────────────────────────
source venv/bin/activate
kubectl port-forward service/webapp 8080:80 &
kubectl port-forward deployment/webapp 5001:5001 &
kubectl port-forward service/prometheus 9090:9090 &
cd frontend/ && npm run dev &            # http://localhost:3000

# ── Тесты ────────────────────────────────────────────────────────────────
cd ../ && mkdir -p results
python -m pytest tests/test_experiments.py -v -s \
  --html=results/report.html 2>&1 | tee results/output.txt
```

---

## Устранение типичных проблем

| Проблема | Причина | Решение |
|----------|---------|---------|
| `pip install` зависает на `torch` | Большой пакет (~800 МБ) | Подождать 5–10 мин; проверить интернет |
| `kind create cluster` зависает | Docker Desktop не запущен или мало RAM | Открыть Docker Desktop → Settings → Resources → RAM ≥ 6 ГБ |
| Поды в состоянии `Pending` | Нехватка ресурсов на узлах | `kubectl describe pod <имя>` → уменьшить `memory requests` в манифесте |
| `ImagePullBackOff` у webapp | Образ не загружен в kind | `kind load docker-image webapp:latest --name webapp-cluster` |
| Интерфейс не обновляется | Port-forward не запущен | Выполнить команды из Шага 7 в отдельном терминале |
| `Connection refused` на 5001 | API не запущен | `python app/api.py` или port-forward 5001:5001 |
| Тест Alibaba пропускается | Датасет не скачан | Выполнить Шаг 10 |
| Медленное обучение в тестах | CPU медленный | В `config.yaml` установить `max_epochs: 20` для отладки |
| `SARIMA fit failed` в логах | Нормальная ситуация | Используется автоматический fallback-прогноз |
