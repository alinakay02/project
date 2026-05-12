<template>
  <div>
    <!-- ── Индикатор активной нагрузки ────────────────────────────────── -->
    <div v-if="loadActive" class="load-banner">
      <span class="badge badge-green">Нагрузка активна</span>
      <span class="load-banner-info">
        Класс 1: {{ loadUsers.c1 }} | Класс 2: {{ loadUsers.c2 }} | Класс 3: {{ loadUsers.c3 }}
        пользователей ({{ loadUsers.total }} всего)
      </span>
    </div>

    <!-- ── Верхняя строка метрик ────────────────────────────────────────── -->
    <div class="grid-5">
      <div class="card metric-card">
        <div class="metric-label">CPU утилизация</div>
        <div class="metric-value" :style="{ color: cpuColor }">
          {{ pct(status.metrics.cpu_t) }}
        </div>
        <div class="metric-sub">Целевая: {{ (config.cpu_target * 100).toFixed(0) }}%</div>
        <div class="progress-bar mt-8">
          <div class="progress-fill" :style="{ width: pct(status.metrics.cpu_t), background: cpuColor }"></div>
        </div>
      </div>

      <div class="card metric-card">
        <div class="metric-label">Память</div>
        <div class="metric-value">{{ pct(status.metrics.mem_t) }}</div>
        <div class="metric-sub">Лимит: 1024 МБ</div>
        <div class="progress-bar mt-8">
          <div class="progress-fill" :style="{ width: pct(status.metrics.mem_t), background: '#805ad5' }"></div>
        </div>
      </div>

      <div class="card metric-card">
        <div class="metric-label">RPS</div>
        <div class="metric-value">{{ status.metrics.rps_t?.toFixed(1) }}</div>
        <div class="metric-sub">запросов в секунду</div>
      </div>

      <div class="card metric-card">
        <div class="metric-label">Задержка P95</div>
        <div class="metric-value" :style="{ color: latColor }">
          {{ status.metrics.lat_t?.toFixed(0) }} <span style="font-size:14px">мс</span>
        </div>
        <div class="metric-sub">95-й перцентиль</div>
      </div>

      <div class="card metric-card">
        <div class="metric-label">Ошибки 5xx</div>
        <div class="metric-value" :style="{ color: status.metrics.err_t > 0.01 ? '#e53e3e' : '#38a169' }">
          {{ (status.metrics.err_t * 100)?.toFixed(2) }}%
        </div>
        <div class="metric-sub">доля запросов</div>
      </div>
    </div>

    <!-- ── Переключатель горизонта ─────────────────────────────────── -->
    <div class="chart-tabs mt-4">
      <button :class="['tab', chartView === 'realtime' ? 'tab--active' : '']"
              @click="chartView = 'realtime'">60 мин</button>
      <button :class="['tab', chartView === 'daily' ? 'tab--active' : '']"
              @click="chartView = 'daily'; fetchDaily()">24 часа</button>
    </div>

    <!-- ── Режим «60 мин»: линейные графики с прогнозом ────────────── -->
    <template v-if="chartView === 'realtime'">
      <div class="mt-4">
        <div class="card">
          <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
            <span>Утилизация процессора</span>
            <span class="badge badge-blue">интервал {{ config.dt_minutes }} мин</span>
          </div>
          <div class="chart-wrap">
            <LineChart :data="cpuChartData" :options="cpuChartOptions" />
          </div>
        </div>
      </div>
      <div class="mt-4">
        <div class="card">
          <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
            <span>Число запросов в секунду</span>
            <span class="badge badge-blue">интервал {{ config.dt_minutes }} мин</span>
          </div>
          <div class="chart-wrap">
            <LineChart :data="rpsLineChartData" :options="rpsLineChartOptions" />
          </div>
        </div>
      </div>
    </template>

    <!-- ── Режим «24 часа»: гистограммы по часам ──────────────────── -->
    <template v-if="chartView === 'daily'">
      <div class="mt-4">
        <div class="card">
          <div class="card-title">Утилизация процессора</div>
          <div class="chart-wrap">
            <BarChart :data="cpuDailyData" :options="dailyCpuOptions" />
          </div>
        </div>
      </div>
      <div class="mt-4">
        <div class="card">
          <div class="card-title">Число запросов в секунду</div>
          <div class="chart-wrap">
            <BarChart :data="rpsDailyData" :options="dailyRpsOptions" />
          </div>
        </div>
      </div>
    </template>

    <!-- ── Масштабирование ────────────────────────────────────────────── -->
    <div class="mt-4">
      <div class="card">
        <div class="card-title">Управление репликами</div>

        <div class="replicas-visual">
          <div v-for="i in config.r_max_cluster" :key="i"
               :class="['replica-pod', i <= status.replicas.current ? 'replica-pod--active' : 'replica-pod--idle']">
            <span>Pod</span><span class="pod-num">{{ i }}</span>
          </div>
        </div>

        <div class="stats-row mt-4">
          <div class="stat-item">
            <div class="stat-label">Текущих реплик</div>
            <div class="stat-val">{{ status.replicas.current }} / {{ config.r_max_cluster }}</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Последнее действие</div>
            <div class="stat-val">
              <span :class="actionBadgeClass">{{ actionLabel }}</span>
            </div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Насыщение</div>
            <div class="stat-val">
              <span v-if="status.replicas.saturation" class="badge badge-orange">RESOURCE_SATURATION</span>
              <span v-else class="badge badge-green">Норма</span>
            </div>
          </div>
        </div>

        <div class="forecast-steps mt-4">
          <div class="card-title" style="margin-bottom:8px">Прогноз нагрузки ({{ config.horizon_h }} шага вперёд)</div>
          <div class="forecast-grid">
            <div v-for="(v, k) in status.forecast.cpu_hat" :key="k" class="forecast-step">
              <div class="fs-label">t+{{ k+1 }}</div>
              <div class="fs-val">CPU {{ (v*100).toFixed(1) }}%</div>
              <div class="fs-ci">
                [{{ (status.forecast.q_lower[k]*100).toFixed(1) }},
                 {{ (status.forecast.q_upper[k]*100).toFixed(1) }}]
              </div>
              <div class="fs-val" style="color:#805ad5;font-size:16px;margin-top:4px">
                RPS {{ status.forecast.rps_hat?.[k]?.toFixed(1) || '—' }}
              </div>
              <div class="fs-ci">
                [{{ status.forecast.rps_lower?.[k]?.toFixed(1) || '—' }},
                 {{ status.forecast.rps_upper?.[k]?.toFixed(1) || '—' }}]
              </div>
              <div class="fs-r">r_req = {{ rReq(status.forecast.q_upper[k]) }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ── RPS по классам ──────────────────────────────────────────────── -->
    <div class="mt-4">
      <div class="card">
        <div class="card-title">Интенсивность запросов по классам (RPS)</div>
        <div class="chart-wrap-small">
          <BarChart :data="rpsChartData" :options="barOptions" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import axios from 'axios'
import { useApiStream } from '../composables/useApiStream'
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, BarElement, Title, Tooltip, Legend, Filler
} from 'chart.js'
import { Line as LineChart, Bar as BarChart } from 'vue-chartjs'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement,
                  BarElement, Title, Tooltip, Legend, Filler)

const status  = ref({ metrics: { cpu_t:0, mem_t:0, rps_t:0, lat_t:0, err_t:0, phi:[.33,.33,.34] },
                       replicas: { current:2, action:'no_change', saturation:false },
                       forecast: { cpu_hat:[0,0,0], q_lower:[0,0,0], q_upper:[0,0,0] } })
const history = ref([])
const config  = ref({ cpu_target: 0.70, r_min: 2, r_max_cluster: 20, horizon_h: 3, dt_minutes: 5 })
const loadActive = ref(false)
const loadUsers  = ref({ c1: 0, c2: 0, c3: 0, total: 0 })
const chartView  = ref('realtime')
const hourlyData = ref([])

const classColors = ['#3182ce', '#38a169', '#805ad5']

const pct = v => ((v || 0) * 100).toFixed(1) + '%'
const cpuColor = computed(() => {
  const c = status.value.metrics.cpu_t
  if (c > config.value.cpu_target + 0.15) return '#e53e3e'
  if (c > config.value.cpu_target) return '#dd6b20'
  return '#38a169'
})
const latColor = computed(() => status.value.metrics.lat_t > 200 ? '#e53e3e' : '#1a202c')
const rReq = q => Math.max(config.value.r_min, Math.ceil((q || 0) / config.value.cpu_target))

const fmtTime = ts => {
  const d = new Date(ts * 1000)
  return d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0')
}

const actionLabel = computed(() => ({
  scale_up:     'Масштабирование вверх',
  scale_down:   'Масштабирование вниз',
  no_change:    'Без изменений',
  hpa_reactive: 'Реактивный HPA',
  saturated:    'Насыщение',
}[status.value.replicas.action] || '—'))

const actionBadgeClass = computed(() => ({
  scale_up:   'badge badge-green',
  scale_down: 'badge badge-blue',
  no_change:  'badge badge-gray',
  saturated:  'badge badge-orange',
}[status.value.replicas.action] || 'badge badge-gray'))

const cpuChartData = computed(() => {
  const labels = history.value.map(h => fmtTime(h.ts))
  const cpuHist = history.value.map(h => +(h.cpu * 100).toFixed(2))
  const cpuPred = history.value.map(h => h.cpu_pred != null ? +(h.cpu_pred * 100).toFixed(2) : null)
  const h = config.value.horizon_h || 3
  const dt = (config.value.dt_minutes || 5) * 60
  const lastUnix = history.value.length ? history.value[history.value.length-1].ts : Math.floor(Date.now()/1000)
  const fLabels = Array.from({length: h + 1}, (_, i) => fmtTime(lastUnix + i * dt))
  const fc      = status.value.forecast
  const lastCpu = cpuHist[cpuHist.length-1] || 0
  const nullFuture = new Array(fLabels.length).fill(null)
  const nullHist   = new Array(labels.length).fill(null)

  return {
    labels: [...labels, ...fLabels],
    datasets: [
      // 1. Факт — сплошная синяя (только история)
      {
        label: 'CPU факт., %',
        data: [...cpuHist, ...nullFuture],
        borderColor: '#3182ce', backgroundColor: 'rgba(49,130,206,.12)',
        borderWidth: 2, fill: true, tension: .3, pointRadius: 0,
      },
      // 2. Прогноз на истории — оранжевый пунктир (сравнение с фактом)
      {
        label: 'CPU прогноз, %',
        data: [...cpuPred, ...nullFuture],
        borderColor: '#dd6b20', borderWidth: 2,
        borderDash: [5,3], tension: .3, pointRadius: 0, fill: false,
        spanGaps: true,
      },
      // 3. Будущий прогноз — медиана (всегда видна)
      {
        label: 'Прогноз (медиана)',
        data: [...nullHist, lastCpu,
               ...(fc.cpu_hat || []).map(v => +(v*100).toFixed(2))],
        borderColor: '#dd6b20', borderWidth: 2,
        borderDash: [5,3], pointRadius: 3, fill: false, tension: .3,
      },
      // 4. Верхний квантиль (будущий прогноз)
      {
        label: 'Верхний квантиль (q0.975)',
        data: [...nullHist, lastCpu,
               ...(fc.q_upper || []).map(v => +(v*100).toFixed(2))],
        borderColor: '#e53e3e', borderWidth: 1,
        borderDash: [3,3], pointRadius: 0, fill: false,
      },
      // 5. Нижний квантиль (будущий прогноз) + заливка
      {
        label: 'Нижний квантиль (q0.025)',
        data: [...nullHist, lastCpu,
               ...(fc.q_lower || []).map(v => +(v*100).toFixed(2))],
        borderColor: '#38a169', borderWidth: 1,
        borderDash: [3,3], pointRadius: 0, fill: '-1',
        backgroundColor: 'rgba(56,161,105,.12)',
      },
      // 6. Целевой уровень
      {
        label: `Цель ${(config.value.cpu_target*100).toFixed(0)}%`,
        data: [...labels, ...fLabels].map(() => config.value.cpu_target * 100),
        borderColor: '#a0aec0', borderWidth: 1,
        borderDash: [6,4], pointRadius: 0, fill: false,
      },
    ]
  }
})
const cpuChartOptions = {
  responsive: true, maintainAspectRatio: false,
  animation: { duration: 0 },
  scales: {
    x: { ticks: { color: '#4a5568', maxTicksLimit: 10 }, grid: { color: '#edf2f7' } },
    y: { ticks: { color: '#4a5568', callback: v => v+'%' }, grid: { color: '#edf2f7' },
         min: 0, max: 100 }
  },
  plugins: { legend: { labels: { color: '#4a5568', font: { size: 11 } } } }
}

const rpsChartData = computed(() => {
  const phi = status.value.metrics.phi || [.33,.33,.34]
  const rps = status.value.metrics.rps_t || 0
  return {
    labels: ['Класс 1 (вычислительные)', 'Класс 2 (база данных)', 'Класс 3 (память)'],
    datasets: [{
      label: 'RPS',
      data: phi.map(p => +(rps * p).toFixed(1)),
      backgroundColor: classColors,
      borderRadius: 4,
    }]
  }
})
const barOptions = {
  responsive: true, maintainAspectRatio: false, animation: { duration: 200 },
  scales: {
    x: { ticks: { color: '#4a5568', font:{size:11} }, grid: { color: '#edf2f7' } },
    y: { ticks: { color: '#4a5568' }, grid: { color: '#edf2f7' }, beginAtZero: true, min: 0 }
  },
  plugins: { legend: { display: false } }
}

// ── График RPS (история + прогноз) ──────────────────────────────────
const rpsLineChartData = computed(() => {
  const labels = history.value.map(h => fmtTime(h.ts))
  const rpsHist = history.value.map(h => +(h.rps).toFixed(1))
  const rpsPred = history.value.map(h => h.rps_pred != null ? +h.rps_pred.toFixed(1) : null)
  const h = config.value.horizon_h || 3
  const dt = (config.value.dt_minutes || 5) * 60
  const lastUnix = history.value.length ? history.value[history.value.length-1].ts : Math.floor(Date.now()/1000)
  const fLabelsRps = Array.from({length: h + 1}, (_, i) => fmtTime(lastUnix + i * dt))
  const fc = status.value.forecast
  const lastRps = rpsHist[rpsHist.length-1] || 0
  const nullFuture = new Array(fLabelsRps.length).fill(null)
  const nullHist   = new Array(labels.length).fill(null)

  return {
    labels: [...labels, ...fLabelsRps],
    datasets: [
      // 1. Факт — сплошная фиолетовая (только история)
      {
        label: 'RPS факт.',
        data: [...rpsHist, ...nullFuture],
        borderColor: '#805ad5', backgroundColor: 'rgba(128,90,213,.10)',
        borderWidth: 2, fill: true, tension: .3, pointRadius: 0,
      },
      // 2. Прогноз на истории — оранжевый пунктир (сравнение с фактом)
      {
        label: 'RPS прогноз',
        data: [...rpsPred, ...nullFuture],
        borderColor: '#dd6b20', borderWidth: 2,
        borderDash: [5,3], tension: .3, pointRadius: 0, fill: false,
        spanGaps: true,
      },
      // 3. Будущий прогноз — медиана (всегда видна, отдельный датасет)
      {
        label: 'Прогноз (медиана)',
        data: [...nullHist, lastRps,
               ...(fc.rps_hat || []).map(v => +v.toFixed(1))],
        borderColor: '#dd6b20', borderWidth: 2,
        borderDash: [5,3], pointRadius: 3, fill: false, tension: .3,
      },
      // 4. Верхний квантиль (будущий прогноз)
      {
        label: 'Верхний квантиль',
        data: [...nullHist, lastRps,
               ...(fc.rps_upper || []).map(v => +v.toFixed(1))],
        borderColor: '#e53e3e', borderWidth: 1,
        borderDash: [3,3], pointRadius: 0, fill: false,
      },
      // 5. Нижний квантиль + заливка
      {
        label: 'Нижний квантиль',
        data: [...nullHist, lastRps,
               ...(fc.rps_lower || []).map(v => +v.toFixed(1))],
        borderColor: '#38a169', borderWidth: 1,
        borderDash: [3,3], pointRadius: 0, fill: '-1',
        backgroundColor: 'rgba(56,161,105,.10)',
      },
    ]
  }
})
const rpsLineChartOptions = {
  responsive: true, maintainAspectRatio: false,
  animation: { duration: 0 },
  scales: {
    x: { ticks: { color: '#4a5568', maxTicksLimit: 10 }, grid: { color: '#edf2f7' } },
    y: { ticks: { color: '#4a5568' }, grid: { color: '#edf2f7' },
         title: { display: true, text: 'запросов/сек', color: '#4a5568' }, min: 0 }
  },
  plugins: { legend: { labels: { color: '#4a5568', font: { size: 11 } } } }
}

// ── Суточные графики (гистограммы по часам) ─────────────────────────
async function fetchDaily() {
  try {
    const { data } = await axios.get('/api/history/daily')
    hourlyData.value = data.hourly || []
  } catch {}
}

const cpuDailyData = computed(() => {
  const hrs = hourlyData.value
  return {
    labels: hrs.map(h => h.hour),
    datasets: [
      {
        label: 'CPU факт., %',
        data: hrs.map(h => h.cpu_avg != null ? +(h.cpu_avg * 100).toFixed(1) : null),
        backgroundColor: 'rgba(49,130,206,.6)', borderRadius: 4,
      },
      {
        label: 'CPU прогноз, %',
        data: hrs.map(h => h.cpu_pred_avg != null ? +(h.cpu_pred_avg * 100).toFixed(1) : null),
        backgroundColor: 'rgba(221,107,32,.5)', borderRadius: 4,
      },
    ]
  }
})

const rpsDailyData = computed(() => {
  const hrs = hourlyData.value
  return {
    labels: hrs.map(h => h.hour),
    datasets: [
      {
        label: 'RPS факт.',
        data: hrs.map(h => h.rps_avg),
        backgroundColor: 'rgba(128,90,213,.6)', borderRadius: 4,
      },
      {
        label: 'RPS прогноз',
        data: hrs.map(h => h.rps_pred_avg),
        backgroundColor: 'rgba(221,107,32,.5)', borderRadius: 4,
      },
    ]
  }
})

const dailyCpuOptions = {
  responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
  scales: {
    x: { ticks: { color: '#4a5568', font:{size:11} }, grid: { color: '#edf2f7' } },
    y: { ticks: { color: '#4a5568', callback: v => v+'%' }, grid: { color: '#edf2f7' },
         min: 0, max: 100 }
  },
  plugins: { legend: { labels: { color: '#4a5568', font: { size: 11 } } } }
}

const dailyRpsOptions = {
  responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
  scales: {
    x: { ticks: { color: '#4a5568', font:{size:11} }, grid: { color: '#edf2f7' } },
    y: { ticks: { color: '#4a5568' }, grid: { color: '#edf2f7' },
         title: { display: true, text: 'запросов/сек', color: '#4a5568' }, min: 0 }
  },
  plugins: { legend: { labels: { color: '#4a5568', font: { size: 11 } } } }
}

// SSE-подписка на статус: сервер шлёт push при каждом новом сэмпле.
const { status: streamStatus } = useApiStream('/api/stream')
watch(streamStatus, (snap) => { if (snap) status.value = snap })

// История и состояние нагрузки всё ещё запрашиваются по таймеру —
// они меняются реже и не стоят отдельного стрима.
let timer = null
async function fetchConfig() {
  try {
    const { data } = await axios.get('/api/config')
    config.value = { ...config.value, ...data }
  } catch {}
}
async function pollSlow() {
  try {
    // range=60m + step=60s даёт 60 точек за час — соответствует UI-кнопке «60 мин».
    // step выбран кратным cadence публикации прогноза контроллером (5 мин),
    // поэтому на графике прогноз отображается ступенчатой оранжевой линией
    // с переходами каждые 5 точек = 5 минут.
    const [h, l] = await Promise.all([
      axios.get('/api/history?range=60m&step=60'),
      axios.get('/api/load'),
    ])
    history.value = h.data.history || []
    loadActive.value = l.data.running || false
    loadUsers.value = {
      c1: l.data.users_class1 || 0,
      c2: l.data.users_class2 || 0,
      c3: l.data.users_class3 || 0,
      total: (l.data.users_class1 || 0) + (l.data.users_class2 || 0) + (l.data.users_class3 || 0),
    }
  } catch {}
}
onMounted(() => { fetchConfig(); pollSlow(); timer = setInterval(pollSlow, 5000) })
onUnmounted(() => clearInterval(timer))
</script>

<style scoped>
.load-banner { display: flex; align-items: center; gap: 12px; padding: 10px 14px;
               background: #f0fff4; border: 1px solid #c6f6d5; border-radius: 8px; margin-bottom: 12px }
.load-banner-info { font-size: 12px; color: #718096 }

.chart-tabs { display: flex; gap: 4px }
.tab { padding: 8px 20px; border: 1px solid #e2e8f0; border-radius: 6px;
       background: #fff; color: #4a5568; font-size: 13px; font-weight: 600;
       cursor: pointer; transition: .15s }
.tab:hover { background: #edf2f7 }
.tab--active { background: #ebf8ff; color: #2b6cb0; border-color: #90cdf4 }

.grid-5 { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px }
.progress-bar { height: 4px; background: #edf2f7; border-radius: 2px; overflow: hidden }
.progress-fill { height: 100%; border-radius: 2px; transition: width .5s }

.chart-wrap { position: relative; height: 40vh; max-height: 380px }
.chart-wrap-small { position: relative; height: 30vh; max-height: 280px }

.replicas-visual { display: flex; gap: 8px; flex-wrap: wrap }
.replica-pod {
  display: flex; flex-direction: column; align-items: center;
  width: 48px; padding: 8px 4px; border-radius: 8px;
  font-size: 10px; border: 1px solid transparent;
}
.replica-pod--active { background: #ebf8ff; border-color: #3182ce; color: #2b6cb0 }
.replica-pod--idle   { background: #edf2f7; border-color: #e2e8f0; color: #a0aec0 }
.pod-num { font-size: 14px; font-weight: 700; margin-top: 2px }

.stats-row  { display: flex; gap: 20px }
.stat-item  { display: flex; flex-direction: column; gap: 4px }
.stat-label { font-size: 11px; color: #718096 }
.stat-val   { font-size: 13px; color: #1a202c; font-weight: 600 }

.forecast-grid { display: flex; gap: 12px }
.forecast-step {
  flex: 1; background: #f7fafc; border-radius: 8px;
  padding: 10px; border: 1px solid #e2e8f0;
}
.fs-label { font-size: 11px; color: #718096 }
.fs-val   { font-size: 20px; font-weight: 700; color: #dd6b20; margin: 4px 0 }
.fs-ci    { font-size: 10px; color: #a0aec0 }
.fs-r     { font-size: 11px; color: #38a169; margin-top: 4px }
</style>
