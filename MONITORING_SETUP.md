# Настройка мониторинга и масштабирования

Инструкция по дополнительной настройке стенда после развертывания (после
шагов 1-8 из `README.md`). Решает две проблемы:

1. На дашборде CPU показывает ~10% при реальной высокой нагрузке.
2. Реплики webapp не масштабируются автоматически даже при перегрузе.

---

## Часть 1. Почему CPU показывает ~10% и как это починить

### Причина

PromQL-формула CPU-утилизации в `config.yaml` использует метрику
`kube_pod_container_resource_limits`:

```promql
sum(rate(container_cpu_usage_seconds_total{pod=~"webapp-.*"}[5m]))
  /
sum(kube_pod_container_resource_limits{resource="cpu", pod=~"webapp-.*"})
```

Метрика `kube_pod_container_resource_limits` приходит из компонента
**kube-state-metrics**, которого в kind по умолчанию нет. Без него
знаменатель = NaN/0 → дашборд показывает околонулевое значение, не
зависящее от реальной нагрузки.

### Решение: установить kube-state-metrics

В **T1** (PowerShell, активированный venv не нужен):

```powershell
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/cluster-role.yaml
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/cluster-role-binding.yaml
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/service-account.yaml
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/deployment.yaml
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/service.yaml
```

Подождать 30-60 секунд, проверить:

```powershell
kubectl get pods -n kube-system | findstr kube-state-metrics
```

Под должен быть в состоянии `Running`. Если нет — `kubectl describe
pod -n kube-system <имя-пода>` для диагностики.

### Решение, шаг 2: добавить scrape-job в Prometheus

Сейчас в `k8s/manifests.yaml` (ConfigMap `prometheus-config`) Prometheus
скрейпит только webapp и cadvisor. Нужно добавить ещё один job для
kube-state-metrics.

Открыть `k8s/manifests.yaml`, найти секцию `prometheus-config` (строка
~390) и в `scrape_configs` добавить:

```yaml
      - job_name: 'kube-state-metrics'
        static_configs:
          - targets: ['kube-state-metrics.kube-system.svc:8080']
```

Применить изменения и перезапустить Prometheus:

```powershell
kubectl apply -f k8s/manifests.yaml
kubectl rollout restart deployment/prometheus
```

### Проверка

В http://localhost:9090:

1. **Status → Targets** — должна появиться цель `kube-state-metrics`
   в состоянии `UP`.
2. Вкладка **Graph**, ввести запрос:
   ```
   kube_pod_container_resource_limits{pod=~"webapp-.*"}
   ```
   Должны вернуться значения по каждому поду webapp.
3. После этого на дашборде http://localhost:3000 в течение 10-20 секунд
   CPU начнёт показывать реальную утилизацию.

---

## Часть 2. Альтернатива без kube-state-metrics

Если ставить kube-state-metrics не хочется, можно изменить формулу CPU
в `config.yaml`, чтобы она использовала **только** метрики kubelet
(они есть всегда):

```yaml
prometheus:
  queries:
    cpu: >
      sum(rate(container_cpu_usage_seconds_total{namespace="default",
      pod=~"webapp-.*", container="webapp"}[1m]))
      /
      count(up{job="webapp"})
```

Это считает «среднее ядро на под». В числителе — сколько ядер сейчас
жгут все поды webapp. В знаменателе — сколько подов живо. Деление
даёт ядра на под; делите ещё на 1.0 (CPU limit = 1 ядро по
манифесту) для нормировки в [0, 1].

После правки нужно перезагрузить config.yaml в ConfigMap и
перезапустить webapp + controller:

```powershell
kubectl apply -f k8s/manifests.yaml
kubectl rollout restart deployment/webapp
kubectl rollout restart deployment/controller
```

---

## Часть 3. Полезные PromQL-запросы для http://localhost:9090

Эти запросы работают **без** kube-state-metrics (только kubelet и
prometheus-flask-exporter из webapp):

```promql
# CPU подов webapp в ядрах (сколько жгут реально)
sum by (pod) (rate(container_cpu_usage_seconds_total{pod=~"webapp-.*",container="webapp"}[1m]))

# Память подов webapp в МБ
sum by (pod) (container_memory_working_set_bytes{pod=~"webapp-.*",container="webapp"}) / 1024 / 1024

# RPS, который реально доходит до webapp
sum(rate(http_requests_total{pod=~"webapp-.*"}[1m]))

# RPS по классам нагрузки (compute / db / memory)
sum by (class_id) (rate(http_requests_total{pod=~"webapp-.*"}[1m]))

# 5xx-ошибки (failures от Locust)
sum(rate(http_requests_total{pod=~"webapp-.*",status=~"5.."}[1m]))

# P95 латентность в мс
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket{pod=~"webapp-.*"}[1m])) by (le)
) * 1000

# Сколько реплик webapp сейчас живо (из метрик Prometheus)
count(up{job="webapp"} == 1)
```

---

## Часть 4. Включение реального масштабирования в коротких тестах

### Почему по умолчанию scaling не работает

1. **Модель не обучена.** Условие активации (`control_loop.py`):
   `len(cpu_buffer) >= 2 * period = 576` точек. При Δt=5 мин это
   **48 часов** непрерывной работы. На 30-минутном тесте модель ни
   разу не активируется.
2. **Баг fallback'а.** Функция `_reactive_hpa` обновляла только
   внутренний `_r_cur` и не вызывала `_apply_k8s` — реальное число
   реплик в кластере не менялось.
3. **Рассинхрон.** Контроллер стартовал с `_r_cur = r_min = 2`, а в
   манифесте `replicas=4`. Решения принимались на основе неверного
   значения.

### Готовый рецепт: pre-train на Alibaba + два фикса в коде

В проекте всё уже сделано, нужно только пересобрать образ controller'а
и применить.

**В коде уже исправлено:**
- [`controller/control_loop.py`](controller/control_loop.py) —
  `_reactive_hpa` теперь вызывает `_apply_k8s` (фикс бага №2).
- [`controller/control_loop.py`](controller/control_loop.py) —
  при старте `_r_cur` синхронизируется с реальным числом реплик через
  `read_namespaced_deployment_scale` (фикс №3).
- [`controller/bootstrap.py`](controller/bootstrap.py) — новая точка
  входа: загружает Alibaba Cluster Trace 2018 (`data/alibaba_cluster_trace_2018.csv`),
  выполняет `_initial_fit()`, прогревает буферы хвостом датасета,
  только после этого стартует основной цикл (фикс №1).
- [`Dockerfile.controller`](Dockerfile.controller) — копирует
  датасет в образ, CMD меняется на `python -m controller.bootstrap`.
- [`requirements-controller.txt`](requirements-controller.txt) —
  добавлен `pandas==2.2.2` для чтения CSV.

### Шаг 4. Пересобрать controller и задеплоить

В **T1** (PowerShell, venv активировать НЕ обязательно):

```powershell
cd C:\diplom\project

# 1. Пересобрать образ controller'а с новым кодом и датасетом внутри
docker build -f Dockerfile.controller -t controller:latest .

# 2. Загрузить в kind (только controller, webapp трогать не нужно)
kind load docker-image controller:latest --name webapp-cluster

# 3. Перезапустить под controller'а, чтобы он подхватил новый образ
kubectl rollout restart deployment/controller

# 4. Дождаться, пока новый под станет Running
kubectl get pods -l app=controller -w
```

`Ctrl+C` после `Running`. Сборка займёт 3-7 минут (controller тоже тянет
torch+pandas).

### Шаг 5. Проверить, что pre-train отработал и модель готова

```powershell
kubectl logs deployment/controller --tail=80
```

Должны увидеть последовательность:

```
[INFO] controller.bootstrap: Loading dataset: /app/data/alibaba_cluster_trace_2018.csv
[INFO] controller.bootstrap: Loaded 2016 points (cpu range: 0.123 - 0.892)
[INFO] controller.control_loop: Synced _r_cur with cluster: 4 replicas
[INFO] predictor.forecaster: ...                  ← обучение GRU
[INFO] controller.bootstrap: Initial fit done. Pre-filling buffers...
[INFO] controller.bootstrap: Pre-filled buffers with 576 points
[INFO] controller.control_loop: Control loop starting.
```

Если видите `Initial fit done` и `Synced _r_cur with cluster: 4 replicas` —
всё готово. Модель будет принимать решения с **первой же** итерации
управляющего цикла (через ~5 минут после старта).

### Шаг 6. Проверить, что scaling реально патчит K8s

Запустите Locust с разогрева, например 1000 пользователей:

```powershell
python -m locust -f locust/locustfile.py --host http://localhost:8080 --headless --users 1000 --spawn-rate 50 --run-time 10m
```

В отдельном терминале наблюдайте за репликами в реальном времени:

```powershell
kubectl get deployment webapp -w
```

Через 5-10 минут после начала нагрузки в логах controller'а должны
появиться записи:
```
[INFO] controller.control_loop: Reactive HPA: 4 -> 6 (cpu_t=0.85)
[INFO] controller.decision: K8s PATCH applied: webapp -> 6 replicas
```
или (если модель уже отрабатывает):
```
[INFO] controller.decision: Scale UP -> 7 replicas (r_req=7)
[INFO] controller.decision: K8s PATCH applied: webapp -> 7 replicas
```

И в `kubectl get deployment webapp -w` число `READY` начнёт расти.

### Если что-то пошло не так

| Симптом | Причина и решение |
|---|---|
| В логах `Dataset not found at /app/data/...` | Образ controller'а собран без датасета. Проверь, что `data/alibaba_cluster_trace_2018.csv` существует на хосте, повтори `docker build`. |
| В логах `Could not sync replicas with cluster: ...403...` | RBAC контроллера не покрывает `deployments/scale`. Проверь `controller-role` в `k8s/manifests.yaml` — должен включать `deployments/scale` в resources. |
| В логах `Kubernetes in-cluster config failed` | `in_cluster: false` где-то остался. Проверь ConfigMap: `kubectl get configmap app-config -o yaml \| findstr in_cluster` — должно быть `true`. |
| `Reactive HPA` видно, но `K8s PATCH applied` нет | `_apply_k8s` упал. Полный стектрейс: `kubectl logs deployment/controller \| findstr "K8s PATCH failed"`. |
| Реплики не растут даже при `K8s PATCH applied` | Кластер не может разместить новый под (нет ресурсов). `kubectl get pods` — ищи `Pending`, потом `kubectl describe pod <pending-pod>`. |

---

## Часть 5. Сводный чеклист

После того как стенд развернут и до запуска Locust:

- [ ] Установлен kube-state-metrics (Часть 1) **или** изменена формула
      CPU в `config.yaml` (Часть 2).
- [ ] В Prometheus (`Status → Targets`) видны цели webapp, cadvisor
      (и kube-state-metrics, если установлен) в состоянии **UP**.
- [ ] PromQL `sum(rate(http_requests_total{pod=~"webapp-.*"}[1m]))`
      возвращает ненулевые значения после старта Locust.
- [ ] Образ controller'а пересобран и загружен в kind (Часть 4, Шаг 4).
- [ ] `kubectl logs deployment/controller --tail=80` содержит
      `Initial fit done` и `Synced _r_cur with cluster: N replicas`.
- [ ] При тесте на 1000 пользователей `kubectl get deployment webapp -w`
      показывает рост реплик через 5-10 минут.

После этого можно запускать полноценные нагрузочные тесты — масштабирование
будет реально срабатывать.

---

## Часть 6. Повторный запуск стенда (после перезагрузки ноута / Docker)

### Сценарий 1: Просто закрыли ноут или перезагрузили Windows

Кластер kind хранится в Docker volume — при выключении не удаляется.

```powershell
# 1. Запустите Docker Desktop (вручную или из меню Пуск).
#    Дождитесь иконки в трее (~30 сек).

# 2. Проверьте, что кластер вернулся
kubectl get nodes
# Все 6 нод должны быть Ready (через 1-2 минуты после старта Docker).

# если тут ошибка было, то надо выполнить: 
#docker restart webapp-cluster-control-plane webapp-cluster-worker webapp-cluster-worker2 webapp-cluster-worker3

# 3. Проверьте поды
kubectl get pods
# Все должны стать Running через 1-3 минуты. Если какой-то застрял
# в Pending/Error — kubectl rollout restart deployment/<имя>.

# 4. Дождитесь, пока controller завершит initial fit (10-15 минут!)
kubectl logs deployment/controller -f | findstr "Initial fit"
# Ctrl+C после строки "Initial fit done."

# 5. Запустите фронтенд (T2)
cd C:\diplom\project\frontend
npm run dev

# 6. Запустите Locust (T3)
cd C:\diplom\project
venv\Scripts\Activate.ps1
python -m locust -f locust/locustfile.py --host http://localhost:8080 --headless --users 1000 --spawn-rate 50 --run-time 10m
```

> **Важно:** controller на каждом старте заново обучает модель на Alibaba
> (10-15 минут). До завершения fit'а он работает в режиме `_reactive_hpa` —
> масштабирование уже работает, но без прогноза. Для чистоты эксперимента
> ждите `Initial fit done` перед запуском Locust.

### Сценарий 2: Поменяли код, нужно перевыкатить

Пересобирать нужно **только** тот образ, чьи исходники меняли:

```powershell
cd C:\diplom\project

# Если меняли controller/, predictor/ или config.yaml для контроллера:
docker build -f Dockerfile.controller -t controller:latest .
kind load docker-image controller:latest --name webapp-cluster
kubectl rollout restart deployment/controller

# Если меняли app/, predictor/ для webapp:
docker build -f Dockerfile.webapp -t webapp:latest .
kind load docker-image webapp:latest --name webapp-cluster
kubectl rollout restart deployment/webapp
kubectl rollout status deployment/webapp

# Если меняли только манифесты k8s/:
kubectl apply -f k8s/manifests.yaml
kubectl rollout restart deployment/webapp        # если правили webapp-секцию
kubectl rollout restart deployment/controller    # если правили controller-секцию
```

`kind load` после `docker build` — обязательно, иначе kind возьмёт
кешированный образ и `rollout restart` ничего не изменит.

### Сценарий 3: Кластер удалили или сборка с нуля

Это полная переустановка — выполняется по `README.md` шаги 4-7, плюс
дополнительно ставится kube-state-metrics из Части 1 этого файла.
Краткая шпаргалка:

```powershell
cd C:\diplom\project

# 1. Кластер
kind create cluster --name webapp-cluster --config kind-config.yaml

# 2. Образы (5-10 минут на первую сборку)
docker build -f Dockerfile.webapp -t webapp:latest .
docker build -f Dockerfile.controller -t controller:latest .
kind load docker-image webapp:latest --name webapp-cluster
kind load docker-image controller:latest --name webapp-cluster

# 3. Основные манифесты
kubectl apply -f k8s/manifests.yaml

# 4. kube-state-metrics (если используете формулу с kube_pod_container_resource_limits)
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/cluster-role.yaml
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/cluster-role-binding.yaml
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/service-account.yaml
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/deployment.yaml
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/service.yaml

# 5. Дождаться Running всех подов
kubectl get pods -w
kubectl get pods -n kube-system | findstr kube-state-metrics

# 6. Дальше — как в Сценарии 1, шаги 4-6.
```

### Полная остановка стенда (когда совсем не нужен)

```powershell
# Удалить кластер (всё внутри пропадёт, образы в Docker остаются)
kind delete cluster --name webapp-cluster

# Опционально — выйти из Docker Desktop через трей
```

При следующем запуске идите по Сценарию 3.

### Если controller не должен переучиваться при каждом старте

Каждый рестарт пода controller'а заново выполняет `_initial_fit` на
Alibaba (10-15 минут). Если это раздражает — можно сохранить
обученную модель в PersistentVolume и грузить её. Это отдельная задача,
но если нужно — спросите, сделаю.
