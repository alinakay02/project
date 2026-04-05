<template>
  <div>
    <div class="page-header">
      <h2 class="page-title">Управление нагрузкой</h2>
      <p class="page-desc">
        Настройка числа виртуальных пользователей по каждому функциональному классу запросов.
      </p>
    </div>

    <div class="grid-2 mt-4">
      <!-- ── Панель управления ─────────────────────────────────────── -->
      <div>
        <!-- Класс 1 -->
        <div class="card load-card">
          <div class="load-header">
            <div class="load-badge" style="background:#ebf8ff;color:#2b6cb0">Класс 1</div>
            <div class="load-name">Вычислительные операции</div>
            <div class="load-cpu">cpu ~ 0.65-0.85</div>
          </div>
          <div class="load-routes">
            <span class="route-tag">/compute/light</span>
            <span class="route-tag">/compute/medium</span>
            <span class="route-tag">/compute/heavy</span>
          </div>
          <div class="load-desc">
            Решето Эратосфена N=5000, факторизация N=10^6, число pi методом Монте-Карло 10^6 итераций.
            Интенсивно нагружает процессор, не обращается к базе данных.
          </div>
          <div class="slider-row">
            <label>Пользователей: <strong style="color:#2b6cb0">{{ users1 }}</strong></label>
            <input type="range" v-model.number="users1" min="0" max="1000" step="10"/>
            <div class="slider-marks"><span>0</span><span>250</span><span>500</span><span>750</span><span>1000</span></div>
          </div>
        </div>

        <!-- Класс 2 -->
        <div class="card load-card mt-4">
          <div class="load-header">
            <div class="load-badge" style="background:#f0fff4;color:#276749">Класс 2</div>
            <div class="load-name">Запросы к базе данных</div>
            <div class="load-cpu">cpu ~ 0.20-0.35</div>
          </div>
          <div class="load-routes">
            <span class="route-tag">/db/read</span>
            <span class="route-tag">/db/write</span>
            <span class="route-tag">/db/aggregate</span>
          </div>
          <div class="load-desc">
            Выборка по ключу из 10^6 строк, вставка, агрегация с оконной функцией (PostgreSQL 15, pgBouncer).
            Умеренная нагрузка на CPU, высокое потребление соединений из пула.
          </div>
          <div class="slider-row">
            <label>Пользователей: <strong style="color:#276749">{{ users2 }}</strong></label>
            <input type="range" v-model.number="users2" min="0" max="1000" step="10"/>
            <div class="slider-marks"><span>0</span><span>250</span><span>500</span><span>750</span><span>1000</span></div>
          </div>
        </div>

        <!-- Класс 3 -->
        <div class="card load-card mt-4">
          <div class="load-header">
            <div class="load-badge" style="background:#faf5ff;color:#6b46c1">Класс 3</div>
            <div class="load-name">Управление памятью</div>
            <div class="load-cpu">cpu ~ 0.30-0.55</div>
          </div>
          <div class="load-routes">
            <span class="route-tag">/memory/alloc</span>
            <span class="route-tag">/memory/process</span>
            <span class="route-tag">/memory/gc</span>
          </div>
          <div class="load-desc">
            NumPy-массив 50 МБ, потоковая обработка 10^5 объектов, gc.collect() с 10^4 цикл. ссылок.
            Нагружает подсистему управления памятью; нерегулярные паузы сборщика мусора.
          </div>
          <div class="slider-row">
            <label>Пользователей: <strong style="color:#6b46c1">{{ users3 }}</strong></label>
            <input type="range" v-model.number="users3" min="0" max="1000" step="10"/>
            <div class="slider-marks"><span>0</span><span>250</span><span>500</span><span>750</span><span>1000</span></div>
          </div>
        </div>

        <!-- Сценарии -->
        <div class="btn-row-presets mt-4">
          <button class="btn btn-ghost" @click="setPreset('compute')">Сценарий: Класс 1</button>
          <button class="btn btn-ghost" @click="setPreset('db')">Сценарий: Класс 2</button>
          <button class="btn btn-ghost" @click="setPreset('memory')">Сценарий: Класс 3</button>
          <button class="btn btn-ghost" @click="setPreset('mixed')">Смешанный</button>
        </div>
        <!-- Кнопки запуска -->
        <div class="btn-row mt-4">
          <button class="btn btn-success" :disabled="!canStart" @click="startLoad">
            Запустить нагрузку
          </button>
          <button class="btn btn-danger" :disabled="!isRunning" @click="stopLoad">
            Остановить
          </button>
        </div>
      </div>

      <!-- ── Правая колонка: статус ────────────────────────────────── -->
      <div>
        <!-- Статус нагрузки -->
        <div class="card">
          <div class="card-title">Текущее состояние нагрузки</div>
          <div class="status-block" :class="isRunning ? 'status-block--on' : 'status-block--off'">
            <span class="status-indicator" :class="isRunning ? 'si--on' : 'si--off'"></span>
            <span>{{ isRunning ? 'Нагрузка активна' : 'Нагрузка остановлена' }}</span>
          </div>

          <div class="phi-preview mt-4">
            <div class="card-title">Ожидаемый состав классов</div>
            <div v-for="(v, i) in expectedPhi" :key="i" class="phi-row">
              <div class="phi-label" :style="{ color: classColors[i] }">Класс {{ i+1 }}</div>
              <div class="phi-bar-wrap">
                <div class="phi-bar-fill" :style="{ width: (v*100)+'%', background: classColors[i] }"></div>
              </div>
              <div class="phi-pct">{{ (v*100).toFixed(1) }}%</div>
            </div>
          </div>

        </div>

        <!-- Фактические метрики системы (live от бэкенда) -->
        <div class="card mt-4">
          <div class="card-title">Фактические метрики системы</div>
          <div class="live-metrics">
            <div class="lm-row">
              <span class="lm-k">CPU утилизация</span>
              <span class="lm-v" :style="{ color: liveMetrics.cpu_t > config.cpu_target ? '#dd6b20' : '#276749' }">
                {{ (liveMetrics.cpu_t * 100).toFixed(1) }}%
              </span>
            </div>
            <div class="lm-row">
              <span class="lm-k">RPS</span>
              <span class="lm-v">{{ liveMetrics.rps_t?.toFixed(1) }}</span>
            </div>
            <div class="lm-row">
              <span class="lm-k">Память</span>
              <span class="lm-v">{{ (liveMetrics.mem_t * 100).toFixed(1) }}%</span>
            </div>
            <div class="lm-row">
              <span class="lm-k">Задержка P95</span>
              <span class="lm-v">{{ liveMetrics.lat_t?.toFixed(0) }} мс</span>
            </div>
            <div class="lm-row">
              <span class="lm-k">Реплик</span>
              <span class="lm-v">{{ liveMetrics.r_cur }}</span>
            </div>
          </div>
          <div class="phi-preview mt-4">
            <div class="card-title">Фактический состав классов (φ)</div>
            <div v-for="(v, i) in liveMetrics.phi" :key="'actual-'+i" class="phi-row">
              <div class="phi-label" :style="{ color: classColors[i] }">Класс {{ i+1 }}</div>
              <div class="phi-bar-wrap">
                <div class="phi-bar-fill" :style="{ width: (v*100)+'%', background: classColors[i] }"></div>
              </div>
              <div class="phi-pct">{{ (v*100).toFixed(1) }}%</div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import axios from 'axios'

const users1 = ref(100)
const users2 = ref(100)
const users3 = ref(100)
const isRunning = ref(false)
const classColors = ['#3182ce', '#38a169', '#805ad5']

const config = ref({ cpu_target: 0.70, r_max_cluster: 20 })

// Фактические метрики от бэкенда (обновляются каждые 5 сек)
const liveMetrics = ref({
  cpu_t: 0, mem_t: 0, rps_t: 0, lat_t: 0, err_t: 0,
  phi: [0.33, 0.33, 0.34], r_cur: 2
})

const canStart = computed(() => !isRunning.value && (users1.value + users2.value + users3.value) > 0)

const totalUsers = computed(() => users1.value + users2.value + users3.value)
const expectedPhi = computed(() => {
  const t = totalUsers.value || 1
  return [users1.value/t, users2.value/t, users3.value/t]
})

async function fetchLoadState() {
  try {
    const { data } = await axios.get('/api/load')
    isRunning.value = data.running || false
    if (data.users_class1 != null) users1.value = data.users_class1
    if (data.users_class2 != null) users2.value = data.users_class2
    if (data.users_class3 != null) users3.value = data.users_class3
  } catch {}
}

async function fetchConfig() {
  try {
    const { data } = await axios.get('/api/config')
    config.value = { ...config.value, ...data }
  } catch {}
}

async function pollMetrics() {
  try {
    const { data } = await axios.get('/api/status')
    liveMetrics.value = {
      cpu_t: data.metrics?.cpu_t || 0,
      mem_t: data.metrics?.mem_t || 0,
      rps_t: data.metrics?.rps_t || 0,
      lat_t: data.metrics?.lat_t || 0,
      err_t: data.metrics?.err_t || 0,
      phi: data.metrics?.phi || [0.33, 0.33, 0.34],
      r_cur: data.replicas?.current || 2,
    }
  } catch {}
}

// Автоматическая отправка изменений слайдеров, пока нагрузка запущена
async function sendLoadUpdate() {
  if (!isRunning.value) return
  try {
    await axios.post('/api/load', {
      running: true,
      users_class1: users1.value,
      users_class2: users2.value,
      users_class3: users3.value,
    })
  } catch {}
}

watch([users1, users2, users3], () => {
  if (isRunning.value) sendLoadUpdate()
})

async function startLoad() {
  try {
    await axios.post('/api/load', {
      running: true,
      users_class1: users1.value,
      users_class2: users2.value,
      users_class3: users3.value,
    })
    isRunning.value = true
  } catch {}
}

async function stopLoad() {
  try {
    await axios.post('/api/load', { running: false, users_class1: 0, users_class2: 0, users_class3: 0 })
    isRunning.value = false
  } catch {}
}

function setPreset(name) {
  const presets = {
    compute: [800, 100, 100],
    db:      [100, 800, 100],
    memory:  [100, 100, 800],
    mixed:   [500, 500, 500],
  }
  const [u1,u2,u3] = presets[name] || [10,10,10]
  users1.value = u1; users2.value = u2; users3.value = u3
}

let metricsTimer = null
onMounted(() => {
  fetchConfig()
  fetchLoadState()
  pollMetrics()
  metricsTimer = setInterval(pollMetrics, 5000)
})
onUnmounted(() => clearInterval(metricsTimer))
</script>

<style scoped>
.page-header { margin-bottom: 4px }
.page-title  { font-size: 20px; font-weight: 700; color: #1a202c }
.page-desc   { font-size: 13px; color: #718096; margin-top: 4px }

.load-card   { padding: 16px }
.load-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px }
.load-badge  { padding: 2px 10px; border-radius: 99px; font-size: 11px; font-weight: 700 }
.load-name   { font-size: 14px; font-weight: 600; color: #1a202c }
.load-cpu    { font-size: 11px; color: #718096; margin-left: auto }
.load-routes { display: flex; gap: 6px; margin-bottom: 8px; flex-wrap: wrap }
.route-tag   { background: #edf2f7; border: 1px solid #e2e8f0;
               padding: 2px 8px; border-radius: 4px; font-size: 11px; color: #2b6cb0;
               font-family: monospace }
.load-desc   { font-size: 12px; color: #718096; line-height: 1.5; margin-bottom: 12px }

.slider-row  { display: flex; flex-direction: column; gap: 6px }
.slider-marks { display: flex; justify-content: space-between; font-size: 10px; color: #a0aec0 }

.btn-row-presets { display: flex; gap: 8px }
.btn-row { display: flex; gap: 8px }

.status-block { display: flex; align-items: center; gap: 10px; padding: 12px;
                border-radius: 8px; font-size: 14px; font-weight: 600 }
.status-block--on  { background: #f0fff4; color: #276749 }
.status-block--off { background: #edf2f7; color: #a0aec0 }

.status-indicator { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0 }
.si--on  { background: #48bb78 }
.si--off { background: #cbd5e0 }

.phi-row  { display: flex; align-items: center; gap: 10px; margin-bottom: 8px }
.phi-label { width: 60px; font-size: 12px; font-weight: 600 }
.phi-bar-wrap { flex: 1; height: 8px; background: #edf2f7; border-radius: 4px; overflow: hidden }
.phi-bar-fill { height: 100%; border-radius: 4px; transition: width .4s }
.phi-pct { width: 42px; font-size: 12px; color: #1a202c; text-align: right }

.live-metrics { display: flex; flex-direction: column; gap: 0 }
.lm-row { display: flex; justify-content: space-between; align-items: center;
          padding: 6px 0; border-bottom: 1px solid #edf2f7 }
.lm-k   { font-size: 12px; color: #718096 }
.lm-v   { font-size: 14px; font-weight: 600; color: #1a202c }

</style>
