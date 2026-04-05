<template>
  <div>
    <!-- ── Верхняя строка метрик ────────────────────────────────────────── -->
    <div class="grid-5">
      <div class="card metric-card">
        <div class="metric-label">CPU утилизация</div>
        <div class="metric-value" :style="{ color: cpuColor }">
          {{ pct(status.metrics.cpu_t) }}
        </div>
        <div class="metric-sub">Целевая: 70%</div>
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

    <!-- ── График CPU + прогноз ────────────────────────────────────────── -->
    <div class="mt-4">
      <div class="card">
        <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
          <span>Утилизация процессора — история и прогноз</span>
          <span class="badge badge-blue">интервал 5 мин</span>
        </div>
        <div class="chart-wrap">
          <LineChart :data="cpuChartData" :options="cpuChartOptions" />
        </div>
      </div>
    </div>

    <!-- ── Масштабирование ────────────────────────────────────────────── -->
    <div class="mt-4">
      <div class="card">
        <div class="card-title">Управление репликами</div>

        <div class="replicas-visual">
          <div v-for="i in 20" :key="i"
               :class="['replica-pod', i <= status.replicas.current ? 'replica-pod--active' : 'replica-pod--idle']">
            <span>Pod</span><span class="pod-num">{{ i }}</span>
          </div>
        </div>

        <div class="stats-row mt-4">
          <div class="stat-item">
            <div class="stat-label">Текущих реплик</div>
            <div class="stat-val">{{ status.replicas.current }} / 20</div>
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
          <div class="card-title" style="margin-bottom:8px">Прогноз нагрузки (3 шага вперёд)</div>
          <div class="forecast-grid">
            <div v-for="(v, k) in status.forecast.cpu_hat" :key="k" class="forecast-step">
              <div class="fs-label">t+{{ k+1 }}</div>
              <div class="fs-val">{{ (v*100).toFixed(1) }}%</div>
              <div class="fs-ci">
                [{{ (status.forecast.q_lower[k]*100).toFixed(1) }},
                 {{ (status.forecast.q_upper[k]*100).toFixed(1) }}]
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
import { ref, computed, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
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

const classColors = ['#3182ce', '#38a169', '#805ad5']

const pct = v => ((v || 0) * 100).toFixed(1) + '%'
const cpuColor = computed(() => {
  const c = status.value.metrics.cpu_t
  if (c > 0.85) return '#e53e3e'
  if (c > 0.70) return '#dd6b20'
  return '#38a169'
})
const latColor = computed(() => status.value.metrics.lat_t > 200 ? '#e53e3e' : '#1a202c')
const rReq = q => Math.max(2, Math.ceil((q || 0) / 0.70))

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
  const labels = history.value.map((_, i) => i)
  const cpuHist = history.value.map(h => +(h.cpu * 100).toFixed(2))
  const lastTs  = labels.length
  const fLabels = [lastTs, lastTs+1, lastTs+2, lastTs+3]
  const fc      = status.value.forecast

  return {
    labels: [...labels, ...fLabels],
    datasets: [
      {
        label: 'CPU факт., %',
        data: [...cpuHist, ...new Array(fLabels.length).fill(null)],
        borderColor: '#3182ce', backgroundColor: 'rgba(49,130,206,.12)',
        borderWidth: 2, fill: true, tension: .3, pointRadius: 0,
      },
      {
        label: 'Прогноз (медиана)',
        data: [...new Array(labels.length).fill(null),
               cpuHist[cpuHist.length-1] || 0,
               ...(fc.cpu_hat || []).map(v => +(v*100).toFixed(2))],
        borderColor: '#dd6b20', borderWidth: 2,
        borderDash: [5,3], pointRadius: 3, fill: false, tension: .3,
      },
      {
        label: 'Верхний квантиль (q0.975)',
        data: [...new Array(labels.length).fill(null),
               cpuHist[cpuHist.length-1] || 0,
               ...(fc.q_upper || []).map(v => +(v*100).toFixed(2))],
        borderColor: '#e53e3e', borderWidth: 1,
        borderDash: [3,3], pointRadius: 0, fill: false,
      },
      {
        label: 'Нижний квантиль (q0.025)',
        data: [...new Array(labels.length).fill(null),
               cpuHist[cpuHist.length-1] || 0,
               ...(fc.q_lower || []).map(v => +(v*100).toFixed(2))],
        borderColor: '#38a169', borderWidth: 1,
        borderDash: [3,3], pointRadius: 0, fill: '-1',
        backgroundColor: 'rgba(56,161,105,.12)',
      },
      {
        label: 'Цель 70%',
        data: [...labels, ...fLabels].map(() => 70),
        borderColor: '#a0aec0', borderWidth: 1,
        borderDash: [6,4], pointRadius: 0, fill: false,
      }
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
    y: { ticks: { color: '#4a5568' }, grid: { color: '#edf2f7' } }
  },
  plugins: { legend: { display: false } }
}

let timer = null
async function poll() {
  try {
    const [s, h] = await Promise.all([
      axios.get('/api/status'),
      axios.get('/api/history?n=80'),
    ])
    status.value  = s.data
    history.value = h.data.history || []
  } catch {}
}
onMounted(() => { poll(); timer = setInterval(poll, 5000) })
onUnmounted(() => clearInterval(timer))
</script>

<style scoped>
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
