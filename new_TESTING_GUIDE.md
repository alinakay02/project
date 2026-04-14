# Руководство по тестированию

Пошаговая инструкция: запуск стенда, нагрузочное тестирование
на 5 000 — 12 000 пользователей, эксперименты pytest, мониторинг.

---

## 0. Архитектура метода

Разработанный метод: **STL-декомпозиция + GRU**.
- STL разделяет ряд на тренд T_t, сезонность S_t и остаток R_t
- GRU предсказывает нормализованный CPU на h шагов вперёд
- Конформная калибровка обеспечивает ~95% покрытие ДИ

Все гиперпараметры хранятся в **config.yaml** (единый источник).

---

## 0.1. Схема терминалов

Для полного цикла тестирования нужны **7 терминалов**.

| #  | Что работает                  | venv? | Закрывать? |
|----|-------------------------------|-------|------------|
| T1 | kubectl, docker, тесты pytest | да    | нет        |
| T2 | port-forward webapp 8080:80   | нет   | нет        |
| T3 | port-forward api 5001:5001    | нет   | нет        |
| T4 | port-forward prometheus 9090  | нет   | нет        |
| T5 | npm run dev (фронтенд)        | нет   | нет        |
| T6 | locust master                 | да    | нет        |
| T7 | locust worker(ы)              | да    | нет        |

---

## 1. Повторный запуск стенда (кластер уже был создан ранее)

Все команды ниже — в **T1**.

### 1.1. Проверить, жив ли кластер

```powershell
kind get clusters
```

Если в списке есть `webapp-cluster` — кластер жив, переходи к п. 1.3.
Если пустой — создай заново (п. 1.2).

### 1.2. Кластер не найден — создать заново

```powershell
cd C:\diplom\project
```

Сохрани `kind-config.yaml` в корне проекта:

```yaml
# kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
  - role: worker
  - role: worker
  - role: worker
```

```powershell
kind create cluster --name webapp-cluster --config kind-config.yaml
```

Подожди 2-3 минуты, затем проверь:

```powershell
kubectl get nodes
```

Все 6 нод должны быть `Ready`.

### 1.3. Пересобрать и задеплоить (если код менялся)

```powershell
cd C:\diplom\project

docker build -f Dockerfile.webapp -t webapp:latest .
docker build -f Dockerfile.controller -t controller:latest .

kind load docker-image webapp:latest --name webapp-cluster
kind load docker-image controller:latest --name webapp-cluster

kubectl apply -f k8s/manifests.yaml
```

### 1.4. Подождать, пока поды запустятся

```powershell
kubectl get pods -w
```

Жди, пока **все** поды станут `Running` (2-3 минуты). `Ctrl+C` когда готово.

### 1.5. Запустить port-forward (три отдельных терминала)

**T2:** `kubectl port-forward service/webapp 8080:80`
**T3:** `kubectl port-forward deployment/webapp 5001:5001`
**T4:** `kubectl port-forward service/prometheus 9090:9090`

### 1.6. Запустить фронтенд

**T5:**
```powershell
cd C:\diplom\project\frontend
npm run dev
```

### 1.7. Проверить

- http://localhost:3000 — веб-интерфейс
- http://localhost:8080/health — `{"status": "ok"}`
- http://localhost:9090 — Prometheus UI

---

## 2. Настройка под высокую нагрузку (5 000 — 12 000 пользователей)

### 2.1. Увеличить лимиты в config.yaml

```yaml
decision:
  r_max_cluster: 20      # было 8
  db:
    max_conn: 300         # было 100
    conn_reserve: 20
```

### 2.2. Обновить манифесты Kubernetes

В `k8s/manifests.yaml`:

**PostgreSQL:**
```yaml
- name: POSTGRES_INITDB_ARGS
  value: "-c max_connections=300"
# resources → limits:
cpu: "2000m"
memory: "2Gi"
```

**Webapp:**
```yaml
replicas: 4
# resources → limits:
cpu: "1000m"
memory: "1Gi"
```

### 2.3. Применить

```powershell
kubectl apply -f k8s/manifests.yaml
kubectl rollout restart deployment webapp
kubectl rollout restart deployment postgres
kubectl get pods -w
```

Если ошибка с PostgreSQL:
```powershell
kubectl delete deployment postgres
kubectl delete pvc postgres-pvc
kubectl apply -f k8s/manifests.yaml
kubectl get pods -w
```

---

## 3. Нагрузочное тестирование с Locust

### 3.1. Разогрев — 1 000 пользователей (5 мин)

**T6:**
```powershell
cd C:\diplom\project
venv\Scripts\activate
locust -f locust/locustfile.py --host http://localhost:8080 --headless --users 1000 --spawn-rate 50 --run-time 5m
```

Норма: `Avg < 500мс`, `Fails = 0`.

### 3.2. Средняя нагрузка — 5 000 пользователей

**T6 — мастер:**
```powershell
locust -f locust/locustfile.py --host http://localhost:8080 --master --expect-workers=4
```

**T7 — 4 воркера:**
```powershell
for ($i=1; $i -le 4; $i++) { Start-Process powershell -ArgumentList "-Command cd C:\diplom\project; venv\Scripts\activate; locust -f locust/locustfile.py --worker --master-host=127.0.0.1" }
```

Откройте http://localhost:8089 → Users: 5000, Spawn rate: 100 → Start.

### 3.3. Высокая нагрузка — 8 000 пользователей

Та же схема, в Locust UI: Users: 8000, Spawn rate: 150.

### 3.4. Пиковая нагрузка — 12 000 пользователей

**T6:** `locust ... --master --expect-workers=6`
**T7:** 6 воркеров:
```powershell
for ($i=1; $i -le 6; $i++) { Start-Process powershell -ArgumentList "-Command cd C:\diplom\project; venv\Scripts\activate; locust -f locust/locustfile.py --worker --master-host=127.0.0.1" }
```
Users: 12000, Spawn rate: 200.

### 3.5. Headless с сохранением CSV

```powershell
locust -f locust/locustfile.py --host http://localhost:8080 --master --expect-workers=4 --headless --users 5000 --spawn-rate 100 --run-time 30m --csv results/locust_5000
```

### 3.6. По одному классу нагрузки

```powershell
locust -f locust/locustfile.py --host http://localhost:8080 --headless --users 5000 --spawn-rate 100 --run-time 15m --csv results/locust_compute -t ComputeUser
locust -f locust/locustfile.py --host http://localhost:8080 --headless --users 5000 --spawn-rate 100 --run-time 15m --csv results/locust_db -t DBUser
locust -f locust/locustfile.py --host http://localhost:8080 --headless --users 5000 --spawn-rate 100 --run-time 15m --csv results/locust_memory -t MemoryUser
```

---

## 4. Мониторинг во время нагрузки

### 4.1. Веб-интерфейс (http://localhost:3000)

Раздел **Мониторинг**: CPU, прогноз, реплики, RPS по классам.

### 4.2. Prometheus (http://localhost:9090)

```promql
# CPU утилизация
sum(rate(container_cpu_usage_seconds_total{namespace="default",pod=~"webapp-.*"}[5m])) / sum(kube_pod_container_resource_limits{namespace="default",resource="cpu",pod=~"webapp-.*"})

# RPS
sum(rate(http_requests_total{namespace="default",pod=~"webapp-.*"}[5m]))

# P95 задержка (мс)
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{namespace="default",pod=~"webapp-.*"}[5m])) by (le)) * 1000

# Доля ошибок
sum(rate(http_requests_total{namespace="default",pod=~"webapp-.*",status=~"5.."}[5m])) / sum(rate(http_requests_total{namespace="default",pod=~"webapp-.*"}[5m]))
```

### 4.3. kubectl (T1)

```powershell
kubectl get pods -o wide
kubectl get deployment webapp -w          # автоскейлинг в реальном времени
kubectl logs deployment/controller -f     # логи контроллера
kubectl top pods                          # потребление ресурсов
```

---

## 5. Эксперименты pytest (точность модели)

Все параметры модели читаются из `config.yaml`.
Все команды — в **T1** с активированным venv.

### 5.1. Подготовка

```powershell
cd C:\diplom\project
venv\Scripts\activate
```

### 5.2. Поиск лучших гиперпараметров (~2-4 часа)

```powershell
# Все 12 свипов в 2 потока
python -u scripts/run_hp_search.py --parallel

# Или конкретные
python -u scripts/run_hp_search.py --only HiddenDim Dropout LearningRate

# Список доступных свипов
python -u scripts/run_hp_search.py --list
```

Результат: `results/hp_search.txt`.
Найдите строки `BEST` и обновите `config.yaml` лучшими значениями.

Доступные свипы: HiddenDim, Dropout, WInput, LearningRate, BatchSize,
MaxEpochs, Patience, GradClip, NTrend, WNorm, IqrAlpha, WAnomaly.

### 5.3. Очистить кэш после обновления config.yaml

```powershell
Remove-Item -Recurse -Force models\cache
```

### 5.4. Тесты только разработанного метода (~30 мин)

```powershell
# Все 7 датасетов
python -u scripts/run_proposed_only.py

# Один датасет
python -u scripts/run_proposed_only.py --dataset alibaba
python -u scripts/run_proposed_only.py --dataset trend
```

Результат: `results/proposed_output.txt`

### 5.5. Полное сравнение всех методов (~3-5 часов)

9 методов: Разработанный (STL+GRU), ARIMA, SARIMA, Хольт-Винтерс,
Случайный лес, LSTM, GRU, CNN-LSTM, TFT.

```powershell
# Параллельно в 2 потока (рекомендуется)
python -u scripts/run_all_tests.py --parallel

# Один датасет
python -u scripts/run_all_tests.py --dataset alibaba
python -u scripts/run_all_tests.py --dataset google

# Только сравнение (без horizon/timing)
python -u scripts/run_all_tests.py --only-compare

# Список датасетов
python -u scripts/run_all_tests.py --list-datasets
```

Результат: `results/experiments_output.txt`

### 5.6. Зависимость MAE от горизонта h

Входит в полный прогон (`run_all_tests.py`).
Тестирует горизонты h=1, 2, 3, 4, 6 на Alibaba.

### 5.7. Вычислительное время

Входит в полный прогон.
Замеряет среднее время одной итерации predict (ожидание: <10мс).

---

## 6. Рекомендуемый порядок полного тестирования

### Этап 1: Проверка стенда (10 мин)
1. Запусти стенд (раздел 1)
2. Проверь http://localhost:3000, :8080/health, :9090

### Этап 2: Разогрев нагрузки (5 мин)
3. Locust на 1000 пользователей (п. 3.1)
4. Убедись, что ошибок нет

### Этап 3: Нагрузочное тестирование (1.5 часа)
5. 5 000 пользователей — 30 мин (п. 3.2)
6. 8 000 пользователей — 30 мин (п. 3.3)
7. 12 000 пользователей — 30 мин (п. 3.4)
8. Параллельно наблюдай: Locust UI, дашборд, Prometheus, kubectl

### Этап 4: По классам нагрузки (45 мин)
9. Отдельно compute, db, memory по 15 мин (п. 3.6)

### Этап 5: HP-поиск (~2-4 часа)
10. `python -u scripts/run_hp_search.py --parallel`
11. Обновить `config.yaml` лучшими значениями
12. `Remove-Item -Recurse -Force models\cache`

### Этап 6: Тесты разработанного метода (~30 мин)
13. `python -u scripts/run_proposed_only.py`

### Этап 7: Полное сравнение (~3-5 часов)
14. `python -u scripts/run_all_tests.py --parallel`

---

## 7. Что делать, если что-то сломалось

### Поды в CrashLoopBackOff
```powershell
kubectl logs deployment/webapp --previous --tail=50
kubectl rollout restart deployment webapp
```

### Port-forward отвалился
Перезапусти в соответствующем терминале.

### Locust — 100% ошибок
```powershell
curl http://localhost:8080/health
kubectl logs deployment/webapp --tail=20
```

### PostgreSQL — too many connections
Увеличь `max_conn` в config.yaml и manifests.yaml (п. 2).

### Ошибка version mismatch при загрузке модели
```powershell
Remove-Item -Recurse -Force models\cache
```

### Полный сброс кластера
```powershell
kind delete cluster --name webapp-cluster
# Затем начни с п. 1.2
```

---

## 8. Сбор результатов для диссертации

| Источник | Файл | Что содержит |
|----------|------|--------------|
| HP-search | `results/hp_search.txt` | Лучшие гиперпараметры для каждого параметра |
| Proposed only | `results/proposed_output.txt` | Метрики разработанного метода на всех датасетах |
| Полные тесты | `results/experiments_output.txt` | Сравнение 9 методов (строки [METRIC]) |
| Locust | `results/locust_*_stats.csv` | RPS, задержки, ошибки по эндпоинтам |
| Locust | `results/locust_*_stats_history.csv` | Метрики по времени (для графиков) |
| Prometheus | http://localhost:9090 | Графики CPU, RPS, latency |
| Дашборд | http://localhost:3000 | Скриншоты прогнозов и автоскейлинга |
| Конфигурация | `config.yaml` | Финальные гиперпараметры (таблица 4.2) |

Строки из логов вида:
```
[METRIC] PROPOSED: Alibaba | MAE=0.0445±0.0003 | RMSE=0.0599±0.0007 ...
[METRIC] COMPARE: Alibaba | ARIMA | MAE=0.0419±0.0000 ...
[METRIC] HPSEARCH: dataset=Azure | param=hidden_dim | value=96 | MAE=0.0288 ...
[METRIC] HORIZON: h=1 | MAE=0.0415±0.0002 ...
[METRIC] TIMING: Среднее=7.5мс ± 0.9мс | Доля от Dt=0.003%
```
копируются в таблицы 4.4 — 4.11 диссертации.
