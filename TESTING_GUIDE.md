# Руководство по тестированию

Пошаговая инструкция: повторный запуск стенда, нагрузочное тестирование
на 5 000 — 12 000 пользователей, запуск экспериментов pytest, мониторинг.

---

## 0. Схема терминалов

Для полного цикла тестирования нужны **7 терминалов**.
Ниже — краткая карта, подробности по каждому далее.

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

Для высокой нагрузки (5000+ пользователей) нужно больше воркеров.
Сохрани этот файл как `kind-config.yaml` в корне проекта:

```powershell
cd C:\diplom\project
```

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

Создать кластер:

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

Жди, пока **все** поды станут `Running` (обычно 2-3 минуты).
Нажми `Ctrl+C` когда всё готово.

Проверка:

```powershell
kubectl get pods
```

Ожидаемый результат:

```
NAME                          READY   STATUS    RESTARTS   AGE
controller-xxxxx              1/1     Running   0          2m
postgres-xxxxx                1/1     Running   0          2m
prometheus-xxxxx              1/1     Running   0          2m
webapp-xxxxx                  1/1     Running   0          2m
webapp-yyyyy                  1/1     Running   0          2m
```

### 1.5. Запустить port-forward (три отдельных терминала)

**Терминал T2:**
```powershell
kubectl port-forward service/webapp 8080:80
```

**Терминал T3:**
```powershell
kubectl port-forward deployment/webapp 5001:5001
```

**Терминал T4:**
```powershell
kubectl port-forward service/prometheus 9090:9090
```

Каждый терминал будет висеть — это нормально. Не закрывай их.

### 1.6. Запустить фронтенд

**Терминал T5:**
```powershell
cd C:\diplom\project\frontend
npm run dev
```

### 1.7. Проверить, что всё работает

Открой в браузере:

- http://localhost:3000 — веб-интерфейс (дашборд)
- http://localhost:8080/health — должен вернуть `{"status": "ok"}`
- http://localhost:9090 — Prometheus UI

Если всё отвечает — стенд готов к тестированию.

---

## 2. Настройка стенда под высокую нагрузку (5 000 — 12 000 пользователей)

По умолчанию стенд рассчитан на 30 пользователей.
Для 5000+ нужно увеличить лимиты.

### 2.1. Увеличить максимум реплик и подключений к БД

Открой `config.yaml` и измени:

```yaml
decision:
  r_max_cluster: 20      # было 8, стало 20
  db:
    max_conn: 300         # было 100, стало 300
    conn_reserve: 20      # было 10
    pool_size: 5
```

### 2.2. Обновить манифесты Kubernetes

В файле `k8s/manifests.yaml` нужно изменить:

**PostgreSQL** — увеличить max_connections и ресурсы:

```yaml
# В секции postgres → env
- name: POSTGRES_INITDB_ARGS
  value: "-c max_connections=300"
# В секции postgres → resources → limits
cpu: "2000m"        # было 1000m
memory: "2Gi"       # было 1Gi
```

**Webapp** — увеличить начальное число реплик и ресурсы:

```yaml
# В секции webapp → spec
replicas: 4              # было 2, чтобы сразу принять первую волну

# В секции webapp → resources → limits
cpu: "1000m"             # было 500m
memory: "1Gi"            # было 512Mi
```

**ConfigMap** — синхронизировать r_max_cluster и max_conn в блоке `data: config.yaml`
(те же значения, что в п. 2.1).

### 2.3. Применить изменения

В **T1**:

```powershell
cd C:\diplom\project
kubectl apply -f k8s/manifests.yaml
kubectl rollout restart deployment webapp
kubectl rollout restart deployment postgres
kubectl get pods -w
```
--
если тут ошибка то:
# 1. Удалить деплоймент postgres (чтобы освободить PVC)
kubectl delete deployment postgres

# 2. Удалить повреждённый PVC
kubectl delete pvc postgres-pvc

# 3. Применить манифесты заново (создаст новый PVC и чистую БД)
kubectl apply -f k8s/manifests.yaml

# 4. Подождать запуска
kubectl get pods -w
---

Подожди, пока все поды станут Running.

---

## 3. Нагрузочное тестирование с Locust

### Зачем распределённый режим?

Один процесс Locust может поддерживать ~1 000 — 2 000 виртуальных пользователей.
Для 5 000+ нужен **distributed mode**: один мастер + несколько воркеров.

### 3.1. Сценарий 1: Разогрев — 1000 пользователей (проверка стабильности)

Цель: убедиться, что стенд стабилен до начала серьёзной нагрузки.

**Терминал T6** (нужен venv):

```powershell
cd C:\diplom\project
venv\Scripts\activate

locust -f locust/locustfile.py --host http://localhost:8080 --headless --users 1000 --spawn-rate 50 --run-time 5m
```

Что смотреть в выводе:
- `Avg (ms)` — средняя задержка. Норма: < 500 мс.
- `Fails` — количество ошибок. Норма: 0.
- `RPS` — запросов в секунду.

Если ошибок нет — переходи к следующему сценарию.
Если есть ошибки 5xx — смотри логи: `kubectl logs deployment/webapp --tail=50`

### 3.2. Сценарий 2: Средняя нагрузка — 5 000 пользователей

Distributed mode: 1 мастер + 4 воркера.

**Терминал T6** — мастер:

```powershell
cd C:\diplom\project
venv\Scripts\activate

locust -f locust/locustfile.py --host http://localhost:8080 --master --expect-workers=4
```

**Терминал T7** — запустить 4 воркера (одной командой):

```powershell
cd C:\diplom\project
venv\Scripts\activate

Start-Process powershell -ArgumentList "-Command cd C:\diplom\project; venv\Scripts\activate; locust -f locust/locustfile.py --worker --master-host=127.0.0.1"
Start-Process powershell -ArgumentList "-Command cd C:\diplom\project; venv\Scripts\activate; locust -f locust/locustfile.py --worker --master-host=127.0.0.1"
Start-Process powershell -ArgumentList "-Command cd C:\diplom\project; venv\Scripts\activate; locust -f locust/locustfile.py --worker --master-host=127.0.0.1"
Start-Process powershell -ArgumentList "-Command cd C:\diplom\project; venv\Scripts\activate; locust -f locust/locustfile.py --worker --master-host=127.0.0.1"
```

Когда мастер покажет `4 workers connected` — откройте **http://localhost:8089**

В веб-интерфейсе Locust:
- **Number of users**: 5000
- **Spawn rate**: 100 (пользователей в секунду)
- Нажми **Start swarming**

### 3.3. Сценарий 3: Высокая нагрузка — 8 000 пользователей

Та же схема что в 3.2, но в Locust UI ставишь:
- **Number of users**: 8000
- **Spawn rate**: 150

### 3.4. Сценарий 4: Пиковая нагрузка — 12 000 пользователей

Для 12 000 нужно 6 воркеров:

**Терминал T6** — мастер:

```powershell
locust -f locust/locustfile.py --host http://localhost:8080 --master --expect-workers=6
```

**Терминал T7** — 6 воркеров:

```powershell
for ($i=1; $i -le 6; $i++) { Start-Process powershell -ArgumentList "-Command cd C:\diplom\project; venv\Scripts\activate; locust -f locust/locustfile.py --worker --master-host=127.0.0.1" }
```

В Locust UI:
- **Number of users**: 12000
- **Spawn rate**: 200

### 3.5. Сценарий 5: Headless-режим (без браузера, результаты в CSV)

Если нужно запустить тест без UI и сохранить результаты:

**Мастер (T6):**
```powershell
locust -f locust/locustfile.py --host http://localhost:8080 --master --expect-workers=4 --headless --users 5000 --spawn-rate 100 --run-time 30m --csv results/locust_5000
```

**Воркеры (T7):** аналогично п. 3.2.

После завершения в `results/` появятся файлы:
- `locust_5000_stats.csv` — общая статистика по эндпоинтам
- `locust_5000_stats_history.csv` — метрики по времени (для графиков)
- `locust_5000_failures.csv` — список ошибок
- `locust_5000_exceptions.csv` — исключения

### 3.6. Сценарий 6: Тестирование по одному классу нагрузки

Запуск только класса 1 (вычисления):

```powershell
locust -f locust/locustfile.py --host http://localhost:8080 --headless --users 5000 --spawn-rate 100 --run-time 15m --csv results/locust_compute --class-picker -t ComputeUser
```

Аналогично для класса 2 (БД) и класса 3 (память):

```powershell
locust -f locust/locustfile.py --host http://localhost:8080 --headless --users 5000 --spawn-rate 100 --run-time 15m --csv results/locust_db -t DBUser
```

```powershell
locust -f locust/locustfile.py --host http://localhost:8080 --headless --users 5000 --spawn-rate 100 --run-time 15m --csv results/locust_memory -t MemoryUser
```

---

## 4. Мониторинг во время нагрузочного тестирования

Пока Locust работает, нужно отслеживать поведение системы.

### 4.1. Веб-интерфейс проекта (http://localhost:3000)

Открой раздел **Мониторинг**:
- **График CPU** — должен расти при увеличении нагрузки
- **Прогноз** — линия прогноза и доверительный интервал
- **Количество реплик** — должно автоматически расти
- **RPS по классам** — гистограмма распределения запросов

### 4.2. Prometheus (http://localhost:9090)

Полезные PromQL-запросы для ввода в поле Expression:

**CPU утилизация:**
```promql
sum(rate(container_cpu_usage_seconds_total{namespace="default",pod=~"webapp-.*"}[5m])) / sum(kube_pod_container_resource_limits{namespace="default",resource="cpu",pod=~"webapp-.*"})
```

**Запросов в секунду (RPS):**
```promql
sum(rate(http_requests_total{namespace="default",pod=~"webapp-.*"}[5m]))
```

**P95 задержка (мс):**
```promql
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{namespace="default",pod=~"webapp-.*"}[5m])) by (le)) * 1000
```

**Доля ошибок:**
```promql
sum(rate(http_requests_total{namespace="default",pod=~"webapp-.*",status=~"5.."}[5m])) / sum(rate(http_requests_total{namespace="default",pod=~"webapp-.*"}[5m]))
```

**Распределение по классам:**
```promql
sum by (class_id) (rate(http_requests_total{namespace="default",pod=~"webapp-.*"}[5m]))
```

### 4.3. kubectl — состояние подов (T1)

Текущее состояние подов:
```powershell
kubectl get pods -o wide
```

Количество реплик webapp:
```powershell
kubectl get deployment webapp -o jsonpath="{.spec.replicas}"
```

Наблюдение за автоскейлингом в реальном времени:
```powershell
kubectl get deployment webapp -w
```

Логи контроллера (прогнозирование и решения о масштабировании):
```powershell
kubectl logs deployment/controller --tail=100 -f
```

Логи webapp (ошибки):
```powershell
kubectl logs deployment/webapp --tail=50
```

Потребление ресурсов по подам:
```powershell
kubectl top pods
```

---

## 5. Запуск экспериментов pytest (точность модели)

Это отдельная группа тестов: они проверяют точность прогнозирования,
сравнивают метод с базовыми линиями (SARIMA, Prophet, GRU, LSTM, HPA)
и выводят метрики для таблиц диссертации.

Все команды ниже — в **T1** с активированным venv.

### 5.1. Подготовка

```powershell
cd C:\diplom\project
venv\Scripts\activate
mkdir -p results
```

### 5.2. Быстрая проверка — unit-тесты (2-3 минуты)

Проверяет корректность реализации формул 3.3, 3.7, 3.8, 3.9, 3.10, 3.16, 3.17:

```powershell
python -m pytest tests/test_experiments.py -k "Unit" -v -s
```

Что ожидать:
- 10 тестов, все PASSED
- Вывод проверяет каждую формулу из главы 3

### 5.3. Время одной итерации (таблица 4.11, ~5 минут)

Проверяет, что полный цикл прогноза укладывается в 1% от Dt=5 мин:

```powershell
python -m pytest tests/test_experiments.py -k "TestComputationTime" -v -s
```

Ожидаемый вывод:
```
[METRIC] ТЕСТ: Вычислительное время | Среднее=XXмс ± Yмс | Доля от Dt=Z%
```

### 5.4. Влияние признаков phi_t (таблица 4.9, ~10 минут)

Проверяет, что включение классовых признаков снижает MAE на 15%+:

```powershell
python -m pytest tests/test_experiments.py -k "TestMixedLoadScenario" -v -s
```

### 5.5. Анализ абляции (таблица 4.10, ~15 минут)

Последовательно отключает компоненты метода и сравнивает результат:

```powershell
python -m pytest tests/test_experiments.py -k "TestAblation" -v -s
```

Что ожидать — 4 теста:
- Без STL-декомпозиции: MAE ухудшается
- Без гистерезиса: число операций масштабирования растёт
- Без квантильной оценки: SLA нарушения растут
- С фиксированным порогом: утилизация снижается

### 5.6. Точность на всех наборах данных (таблицы 4.4—4.5, ~30 минут)

```powershell
python -m pytest tests/test_experiments.py -k "TestForecastAllDatasets" -v -s
```

Тестирует на 4 синтетических наборах: стационарный, трендовый, всплесковый, смешанный.
Для каждого — 5 прогонов с разными seed.

### 5.7. Эффективность управления (таблица 4.6, ~40 минут)

```powershell
python -m pytest tests/test_experiments.py -k "TestManagementEfficiency" -v -s
```

Проверяет:
- SLA нарушения < 5% (8% на всплесковом)
- Утилизация >= 55%
- Операции масштабирования <= 60% от реактивного HPA

### 5.8. Зависимость от горизонта h (таблица 4.7, ~20 минут)

```powershell
python -m pytest tests/test_experiments.py -k "TestHorizonDependence" -v -s
```

Тестирует горизонты h=1, 2, 3, 4, 6. MAE должна расти с горизонтом.

### 5.9. Влияние дообучения (таблица 4.8, ~15 минут)

```powershell
python -m pytest tests/test_experiments.py -k "TestRetrainingEffect" -v -s
```

### 5.10. Устойчивость к всплескам (таблица 4.5, ~15 минут)

```powershell
python -m pytest tests/test_experiments.py -k "TestSpikeResilience" -v -s
```

Тестирует амплитуды 2sigma, 3sigma, 4sigma, 5sigma.

### 5.11. ВСЕ тесты + HTML-отчёт (полный прогон, ~2-3 часа)

```powershell
python -m pytest tests/test_experiments.py -v -s --html=results/report.html 2>&1 | tee results/output.txt
```

После завершения:
- `results/report.html` — открой в браузере, красивый отчёт с PASSED/FAILED
- `results/output.txt` — полный текстовый лог с метриками [METRIC]

---

## 6. Рекомендуемый порядок полного тестирования

### Этап 1: Проверка стенда (10 минут)
1. Запусти стенд (раздел 1)
2. Открой http://localhost:3000 — убедись, что дашборд работает
3. Открой http://localhost:8080/health — убедись, что webapp отвечает

### Этап 2: Unit-тесты (5 минут)
4. Запусти unit-тесты (п. 5.2)
5. Убедись, что все PASSED

### Этап 3: Разогрев нагрузки (5 минут)
6. Запусти Locust на 1000 пользователей (п. 3.1)
7. Смотри, что ошибок нет

### Этап 4: Основная нагрузка — 5 000 пользователей (30 минут)
8. Запусти distributed Locust на 5000 (п. 3.2)
9. Параллельно наблюдай:
   - http://localhost:8089 — Locust UI (RPS, задержки, ошибки)
   - http://localhost:3000 — дашборд (CPU, прогноз, реплики)
   - http://localhost:9090 — Prometheus (графики метрик)
   - `kubectl get deployment webapp -w` — автоскейлинг
10. Дождись стабилизации (~10 минут), потом останови

### Этап 5: Высокая нагрузка — 8 000 пользователей (30 минут)
11. Запусти на 8000 (п. 3.3)
12. Отслеживай те же показатели
13. Обрати внимание на RESOURCE_SATURATION в логах контроллера

### Этап 6: Пиковая нагрузка — 12 000 пользователей (30 минут)
14. Запусти на 12000 (п. 3.4)
15. Здесь система должна достигнуть предела масштабирования
16. Зафиксируй: при какой нагрузке SLA начинает нарушаться

### Этап 7: Тесты по одному классу (45 минут)
17. Запусти 3 отдельных теста (п. 3.6): только compute, только db, только memory
18. Каждый по 15 минут на 5000 пользователей
19. Сохрани CSV для сравнения классов

### Этап 8: Pytest эксперименты (2-3 часа)
20. Запусти полный прогон (п. 5.11)
21. Дождись завершения, открой results/report.html

---

## 7. Что делать, если что-то сломалось

### Поды в CrashLoopBackOff

```powershell
kubectl logs deployment/webapp --previous --tail=50
kubectl describe pod <имя-пода>
```

Часто помогает:
```powershell
kubectl rollout restart deployment webapp
```

### Port-forward отвалился

Просто перезапусти его в соответствующем терминале.

### Locust показывает 100% ошибок

1. Проверь, что port-forward webapp работает: `curl http://localhost:8080/health`
2. Проверь логи: `kubectl logs deployment/webapp --tail=20`
3. Возможно, поды перегружены — подожди минуту и попробуй снова

### PostgreSQL — слишком много подключений

```powershell
kubectl logs deployment/postgres --tail=20
```

Если видишь "too many connections" — увеличь max_conn (п. 2.1).

### Нужно полностью пересоздать кластер

```powershell
kind delete cluster --name webapp-cluster
```

Затем начни с п. 1.2.

---

## 8. Сбор результатов для диссертации

После всех тестов у тебя будут:

| Источник | Файл / место | Что содержит |
|----------|-------------|--------------|
| pytest   | `results/report.html` | Таблица тестов PASSED/FAILED |
| pytest   | `results/output.txt` | Строки [METRIC] для таблиц 4.4-4.11 |
| Locust   | `results/locust_*_stats.csv` | RPS, задержки, ошибки по эндпоинтам |
| Locust   | `results/locust_*_stats_history.csv` | Метрики по времени (для графиков) |
| Prometheus | http://localhost:9090 | Графики CPU, RPS, latency (экспорт PNG) |
| Дашборд  | http://localhost:3000 | Скриншоты прогнозов и автоскейлинга |

Строки из `output.txt` вида:
```
[METRIC] ТЕСТ: Alibaba | Разработанный метод | MAE=0.058±0.004 ...
[METRIC] MGMT ТЕСТ: Alibaba | SLA_VIOLATIONS=3.2% | AVG_UTIL=70.8% | SCALE_OPS=61
```
копируются напрямую в таблицы 4.4 — 4.11 диссертации.
