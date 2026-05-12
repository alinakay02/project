# Чистый старт на локальном ноуте (i5-1135G7, 16 ГБ RAM)

Инструкция: полностью снести старый стенд и развернуть с нуля с
актуальным кодом и параметрами под слабое железо. Цель — стабильно
выдерживать 1000-2000 пользователей в Locust и видеть масштабирование
в реальном времени.

> Все команды — в **PowerShell** из директории `C:\diplom\project`.
> Где сказано «venv» — `venv\Scripts\Activate.ps1` сначала.

---

## Шаг 1. Снести старый стенд

```powershell
cd C:\diplom\project

# 1. Удалить старый kind-кластер вместе со всеми подами и данными
kind delete cluster --name webapp-cluster

# 2. Удалить старые образы webapp и controller (они без scipy + без bootstrap.py)
docker rmi webapp:latest controller:latest

# 3. Очистить висящие dangling-слои Docker (освободит несколько ГБ)
docker image prune -f
docker builder prune -f
```

Проверка:
```powershell
kind get clusters          # должно быть пусто
docker images | findstr "webapp\|controller"   # должно быть пусто
```

---

## Шаг 2. Настроить Docker Desktop под ваше железо

**Это критично — сейчас Docker видит только 3.82 ГБ из ваших 16.**

1. Открыть Docker Desktop → ⚙️ **Settings** → **Resources**
2. Выставить:
   - **Memory: 10 GB**
   - **CPUs: 6**
   - **Swap: 2 GB**
   - **Disk image size: 40 GB**
3. **Apply & Restart** — Docker перезагрузится (~30 секунд)

Проверка после рестарта:
```powershell
docker info --format '{{json .}}' | ConvertFrom-Json | Select-Object @{n='RAM_GB';e={[math]::Round($_.MemTotal/1GB,2)}},NCPU
# Должно показать RAM_GB=10, NCPU=6
```

---

## Шаг 3. Подкрутить конфиги под ваш ноут

Три файла нужно поправить, чтобы стенд занимал меньше ресурсов и
быстрее стартовал.

### 3.1. `kind-config.yaml` — 3 воркера вместо 5

Откройте `kind-config.yaml` и замените блок `nodes:` на:

```yaml
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30080
        hostPort: 8080
        protocol: TCP
      - containerPort: 30051
        hostPort: 5001
        protocol: TCP
      - containerPort: 30090
        hostPort: 9090
        protocol: TCP
  - role: worker
  - role: worker
  - role: worker
```

Сохраните.

### 3.2. `k8s/manifests.yaml` — стартовать с 2 реплик webapp

Найдите строку 169 (`replicas: 4`) и поменяйте на `replicas: 2`:

```yaml
spec:
  replicas: 2          # старт с 2, controller сам отскейлит при нагрузке
```

### 3.3. `config.yaml` — короткое обучение GRU

В секции `model:` поменяйте две строки:

```yaml
model:
  ...
  max_epochs: 30      # было 80 — сэкономит 5-10 минут на старте контроллера
  patience: 5         # было 10
  ...
```

И **в той же правке** в `k8s/manifests.yaml` (строка ~310, ConfigMap
`app-config`) синхронизировать те же значения — иначе подам внутри
кластера достанется старый конфиг. Найдите внутри `app-config` блок
`model:` и поправьте `max_epochs: 30` и `patience: 5` там тоже.

---

## Шаг 4. Создать новый кластер

```powershell
cd C:\diplom\project
kind create cluster --name webapp-cluster --config kind-config.yaml

# Дождаться, пока все 4 ноды станут Ready (1-2 минуты)
kubectl get nodes
```

Все 4 ноды должны быть `Ready`. Если какая-то `NotReady` — подождите
ещё минуту.

---

## Шаг 5. Собрать образы с новым кодом

В коде уже есть все правки: scipy в webapp, bootstrap.py с pre-train на
Alibaba, фикс `_reactive_hpa`, sync `_r_cur` со стартом. Просто
пересоберите.

```powershell
cd C:\diplom\project

# Сборка webapp (~5-10 минут — тянет torch+scipy)
docker build -f Dockerfile.webapp -t webapp:latest .

# Сборка controller (~3-7 минут — тоже torch)
docker build -f Dockerfile.controller -t controller:latest .
```

Проверка:
```powershell
docker images | findstr "webapp\|controller"
# webapp:latest должен быть ~2-3 ГБ (раньше был 564 МБ — это был БЕЗ scipy)
# controller:latest должен быть ~1.5-2 ГБ
```

---

## Шаг 6. Загрузить образы в kind

```powershell
# Грузим только в воркеры (control-plane не нужен) — экономит ~30%
kind load docker-image webapp:latest --name webapp-cluster --nodes webapp-cluster-worker,webapp-cluster-worker2,webapp-cluster-worker3
kind load docker-image controller:latest --name webapp-cluster --nodes webapp-cluster-worker,webapp-cluster-worker2,webapp-cluster-worker3
```

Это займёт ещё 5-10 минут (грузим в 3 ноды × ~3 ГБ).

---

## Шаг 7. Развернуть стенд

```powershell
kubectl apply -f k8s/manifests.yaml

# Смотрим, как поды поднимаются
kubectl get pods -w
```

Ждите, пока **все** поды станут `Running`:
- `controller-...` — 1 шт
- `postgres-...` — 1 шт
- `prometheus-...` — 1 шт
- `webapp-...` — 2 шт

Это занимает 2-5 минут (нужно дождаться, пока контейнеры распакуются и
пройдут readiness-пробы). `Ctrl+C` после того, как всё `Running`.

> **Если кто-то завис в `Pending`:** скорее всего не хватает ресурсов.
> Проверка: `kubectl describe pod <имя-Pending-пода>`. Если в Events
> видно «Insufficient memory» — вернитесь к Шагу 2 и поднимите Docker
> RAM ещё на 2 ГБ.
--- тут зачастую: kubectl get pods -w
NAME                          READY   STATUS              RESTARTS     AGE
controller-7db6b9fb56-lh2rf   1/1     Running             0            12s
postgres-fd9bd67c8-thn7t      0/1     ContainerCreating   0            12s
prometheus-7768c56945-8jg6q   0/1     ContainerCreating   0            12s
webapp-67d9dcb4d8-85wrx       0/1     CrashLoopBackOff    1 (5s ago)   12s
webapp-67d9dcb4d8-rr5v9       0/1     CrashLoopBackOff    1 (5s ago)   12s
webapp-67d9dcb4d8-rr5v9       0/1     Running             2 (17s ago)   24s
webapp-67d9dcb4d8-85wrx       0/1     Running             2 (19s ago)   26s
prometheus-7768c56945-8jg6q   1/1     Running             0             27s
webapp-67d9dcb4d8-rr5v9       0/1     Error               2 (25s ago)   32s
webapp-67d9dcb4d8-85wrx       0/1     Error               2 (25s ago)   32s
webapp-67d9dcb4d8-rr5v9       0/1     CrashLoopBackOff    2 (13s ago)   40s
webapp-67d9dcb4d8-85wrx       0/1     CrashLoopBackOff    2 (9s ago)    40
webapp-67d9dcb4d8-rr5v9       0/1     Error               3 (37s ago)   64s
webapp-67d9dcb4d8-rr5v9       0/1     CrashLoopBackOff    3 (13s ago)   70s
---

## Шаг 8. Установить kube-state-metrics

Без него дашборд показывает CPU=0%, потому что метрика
`kube_pod_container_resource_limits` не существует.

```powershell
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/cluster-role.yaml
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/cluster-role-binding.yaml
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/service-account.yaml
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/deployment.yaml
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/raw/main/examples/standard/service.yaml

# Подождать 30-60 секунд
kubectl get pods -n kube-system | findstr kube-state-metrics
# Должен быть Running
```

**Дополнительно** — нужно сказать Prometheus, чтобы он его скрейпил.
Откройте `k8s/manifests.yaml`, найдите секцию `prometheus-config`
(строка ~390), внутри `scrape_configs` добавьте после `cadvisor`:

```yaml
      - job_name: 'kube-state-metrics'
        static_configs:
          - targets: ['kube-state-metrics.kube-system.svc:8080']
```

Применить и перезапустить Prometheus:

```powershell
kubectl apply -f k8s/manifests.yaml
kubectl rollout restart deployment/prometheus
```

---

## Шаг 9. Проверить, что всё живо

```powershell
# 1. Все поды Running
kubectl get pods

# 2. Сервисы доступны с хоста
curl http://localhost:8080/health
# Ожидаем: {"status":"ok"}

(Invoke-WebRequest http://localhost:5001/api/status).Content | Select-String '"mode"'
# Ожидаем: "mode":"real"

curl http://localhost:9090/-/ready
# Ожидаем: Prometheus is Ready.

# 3. В Prometheus есть kube-state-metrics
# Откройте http://localhost:9090 → Status → Targets
# Должно быть 3+ цели в состоянии UP: webapp, cadvisor, kube-state-metrics
```

---

## Шаг 10. Дождаться обучения GRU в controller'е

Controller на старте выполняет initial fit на Alibaba. С `max_epochs: 30`
это займёт **5-10 минут**.

```powershell
kubectl logs deployment/controller -f --tail=20
```

Дожидайтесь строк:
```
[INFO] controller.bootstrap: Loaded 2016 points (cpu range: 0.123 - 0.892)
[INFO] controller.control_loop: Synced _r_cur with cluster: 2 replicas
[INFO] predictor.model: epoch 1 train=0.123 val=0.118
...
[INFO] predictor.model: Training finished, best val=0.087
[INFO] controller.bootstrap: Initial fit done. Pre-filling buffers...
[INFO] controller.bootstrap: Pre-filled buffers with 576 points
[INFO] controller.control_loop: Control loop starting.
```

`Ctrl+C` после `Control loop starting.`. После этого модель готова
делать прогнозы и принимать решения о масштабировании.

---

## Шаг 11. Запустить фронтенд (T2)

В **новом** терминале (T2):

```powershell
cd C:\diplom\project\frontend
npm run dev
```

Откройте http://localhost:3000 — должен открыться дашборд с реальными
метриками (CPU, RPS, реплики).

---

## Шаг 12. Запустить нагрузочный тест на 1000 пользователей

В **третьем** терминале (T3):

```powershell
cd C:\diplom\project
venv\Scripts\Activate.ps1

python -m locust -f locust/locustfile.py --host http://localhost:8080 --headless --users 1000 --spawn-rate 50 --run-time 10m
```

В **четвёртом** терминале (T4) — наблюдение за реальным масштабированием:

```powershell
kubectl get deployment webapp -w
```

Через 5-10 минут после старта Locust в этом окне число `READY` должно
вырасти с `2/2` до `4/4`, `5/5` и далее. В дашборде http://localhost:3000
будет видно повышение CPU и появление новых подов.

В Locust UI должно быть:
- **Failures: <5%** (если больше — см. диагностику ниже)
- **Avg response time:** до 500-800 мс

---

## Шаг 13. Что делать, если 1000 юзеров всё равно даёт failures

| Симптом | Причина | Решение |
|---|---|---|
| Failures 50%+ сразу | Webapp поды OOM'ятся | `kubectl describe pod -l app=webapp` — ищите `OOMKilled`. Поднять `memory: 2Gi` в манифесте. |
| Failures растут со временем | PostgreSQL `too many connections` | `kubectl logs deployment/postgres --tail=20` — если `FATAL: too many connections`, поднять `max_connections=500` в манифесте postgres |
| Failures на /api/* | Vite proxy error | Перезапустить `npm run dev`, проверить что `webapp-api` под Running |
| Дашборд показывает CPU=0% | kube-state-metrics не настроен | Вернуться к Шагу 8, проверить Targets в Prometheus |
| Реплики не растут | Controller fit не завершён или K8s PATCH падает | `kubectl logs deployment/controller --tail=50` — должно быть `Initial fit done` и `K8s PATCH applied` после нагрузки |

---

## Что дальше

После Шага 12 у вас работающий стенд под слабое железо. Можно:
- Прогнать тесты по разным сценариям из `TESTING_GUIDE.md` (только не
  больше 2000 пользователей).
- Сохранять CSV из Locust: добавьте `--csv results/locust_1000` к
  команде.
- Запустить эксперименты по точности модели через `pytest`
  (см. `TESTING_GUIDE.md` Часть 5) — это **не** требует Kubernetes,
  работает только с venv.

При следующих запусках идите по **Сценарию 1** из
[`MONITORING_SETUP.md`](MONITORING_SETUP.md) (Часть 6) — там короткая
шпаргалка после простой перезагрузки ноута.
