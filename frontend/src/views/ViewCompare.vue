<template>
  <div>
    <div class="page-header">
      <h2 class="page-title">Сравнение методов прогнозирования</h2>
    </div>

    <!-- Панель управления -->
    <div class="toolbar mt-4">
      <label>Набор данных:</label>
      <select v-model="selectedDataset" @change="loadAll">
        <option v-for="d in datasets" :key="d.id" :value="d.id">
          {{ d.name }}
        </option>
      </select>
      <button class="btn btn-ghost" @click="loadAll" :disabled="runStatus === 'running'">
        Обновить
      </button>
      <span v-if="dataSource === 'db'" class="badge badge-green">данные из БД</span>
      <span v-else-if="dataSource === 'default'" class="badge badge-blue">эталонные значения</span>
    </div>

    <!-- Запуск реальных тестов -->
    <div class="card mt-4">
      <div class="toolbar">
        <button class="btn btn-primary"
                :disabled="runStatus === 'running'"
                @click="triggerRun('TestCompareAlibaba')">
          {{ runStatus === 'running' ? 'Тесты выполняются…' : 'Сравнение на Alibaba' }}
        </button>
        <button class="btn btn-ghost"
                :disabled="runStatus === 'running'"
                @click="triggerRun('TestComputationTime')">
          Только время итерации
        </button>
        <button class="btn btn-ghost"
                :disabled="runStatus === 'running'"
                @click="triggerRun('TestHorizonDependence')">
          Зависимость от горизонта
        </button>
        <button class="btn btn-ghost"
                :disabled="runStatus === 'running'"
                @click="triggerRun('')">
          Все тесты
        </button>
        <span v-if="runInfo.started_at" class="badge badge-gray">
          старт: {{ runInfo.started_at }}<span v-if="runInfo.filter"> · фильтр: {{ runInfo.filter }}</span>
        </span>
        <span v-if="runStatus === 'running'" class="badge badge-orange">выполняется…</span>
        <span v-else-if="runStatus === 'done'" class="badge badge-green">
          готово · записей: {{ runInfo.count ?? '—' }}
        </span>
      </div>
    </div>

    <!-- ── Таблица сравнения методов ─────────────────────────────────────── -->
    <div class="card mt-4" v-if="comparison.length">
      <div class="card-title">
        Точность прогнозирования и эффективность управления ресурсами
      </div>
      <div class="table-scroll">
        <table class="table">
          <thead>
            <tr>
              <th>Метод</th>
              <th class="num">MAE</th>
              <th class="num">RMSE</th>
              <th class="num">MAPE, %</th>
              <th class="num">Покрытие ДИ, %</th>
              <th class="num">Нарушения SLA, %</th>
              <th class="num">Средний CPU, %</th>
              <th class="num">Операции масштабирования</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(r, i) in comparison" :key="r.method" :class="{ best: i === 0 }">
              <td>
                <span v-if="i === 0" class="badge badge-green">*</span>
                {{ r.method }}
              </td>
              <td class="num">
                <span :style="{ color: i===0 ? '#276749' : '' }">
                  {{ fmt(r.mae, 3) }}
                </span>
              </td>
              <td class="num">{{ fmt(r.rmse, 3) }}</td>
              <td class="num">{{ fmt(r.mape, 1) }}</td>
              <td class="num">
                <span v-if="r.coverage" :class="coverageBadge(r.coverage)">
                  {{ r.coverage?.toFixed(1) }}%
                </span>
                <span v-else class="text-muted">—</span>
              </td>
              <td class="num">
                <span v-if="r.sla_pct != null" :style="{ color: slaColor(r.sla_pct) }">
                  {{ r.sla_pct?.toFixed(1) }}%
                </span>
                <span v-else class="text-muted">—</span>
              </td>
              <td class="num">{{ r.avg_util != null ? r.avg_util.toFixed(1) + '%' : '—' }}</td>
              <td class="num">{{ r.scale_ops ?? '—' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="grid-2 mt-4" v-if="comparison.length">
      <!-- ── Гистограмма MAE ─────────────────────────────────────────────── -->
      <div class="card">
        <div class="card-title">Сравнение MAE по методам</div>
        <div class="chart-wrap">
          <BarChart :data="maeChartData" :options="maeOptions" />
        </div>
      </div>

      <!-- ── Scatter: нарушения SLA vs CPU ──────────────────────────  -->
      <div class="card">
        <div class="card-title">Нарушения SLA и CPU</div>
        <div class="chart-wrap">
          <ScatterChart :data="scatterData" :options="scatterOptions" />
        </div>
      </div>
    </div>

    <!-- ── Зависимость от горизонта h ─────────────────────────────────────── -->
    <div class="card mt-4" v-if="horizonRows.length">
      <div class="card-title">Зависимость MAE от горизонта прогнозирования</div>
      <table class="table">
        <thead>
          <tr>
            <th>Горизонт, шагов (мин)</th>
            <th class="num">MAE</th>
            <th class="num">± std</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in horizonRows" :key="row.h">
            <td><strong>{{ row.h }} ({{ row.h*5 }} мин)</strong></td>
            <td class="num" style="color:#276749">{{ fmt(row.mae, 4) }}</td>
            <td class="num" style="color:#718096">{{ fmt(row.mae_std, 4) }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- ── Смешанный сценарий φ_t ─────────────────────────────────────── -->
    <div class="card mt-4" v-if="phiRows.length">
      <div class="card-title">Эффект признаков состава классов φ_t</div>
      <table class="table">
        <thead>
          <tr>
            <th>MAE без φ_t</th>
            <th>MAE с φ_t</th>
            <th>Снижение MAE, %</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in phiRows" :key="row.mae_with_phi">
            <td class="num" style="color:#e53e3e">{{ fmt(row.mae_without_phi, 4) }}</td>
            <td class="num" style="color:#276749">{{ fmt(row.mae_with_phi, 4) }}</td>
            <td class="num" style="color:#276749">{{ fmt(row.improvement, 1) }}%</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- ── Вычислительное время ───────────────────────────────────────────── -->
    <div class="card mt-4" v-if="timeRows.length">
      <div class="card-title">Вычислительное время итерации</div>
      <table class="table">
        <thead>
          <tr>
            <th>Модуль</th>
            <th class="num">Среднее время, мс</th>
            <th class="num">Стандартное отклонение, мс</th>
            <th class="num">Доля от интервала</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in timeRows" :key="row.module" :class="{ best: row.module === 'Полный цикл' }">
            <td>{{ row.module }}</td>
            <td class="num">{{ row.mean_ms }}</td>
            <td class="num">{{ row.std_ms }}</td>
            <td class="num" style="color:#276749">{{ row.pct }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- ── Устойчивость к всплескам ──────────────────────────────────────── -->
    <div class="card mt-4" v-if="spikeRows.length">
      <div class="card-title">Устойчивость к всплескам разной амплитуды</div>
      <table class="table">
        <thead>
          <tr>
            <th>Амплитуда (σ)</th>
            <th class="num">MAE</th>
            <th class="num">± std</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in spikeRows" :key="row.amplitude_sigma">
            <td><strong>{{ row.amplitude_sigma }}σ</strong></td>
            <td class="num">{{ fmt(row.mae, 4) }}</td>
            <td class="num" style="color:#718096">{{ fmt(row.mae_std, 4) }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- ── Влияние дообучения ─────────────────────────────────────────── -->
    <div class="card mt-4" v-if="retrainRows.length">
      <div class="card-title">Влияние дообучения модели</div>
      <table class="table">
        <thead>
          <tr>
            <th>MAE с дообучением</th>
            <th>MAE без дообучения</th>
            <th>Улучшение, %</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in retrainRows" :key="row.mae_with">
            <td class="num" style="color:#276749">{{ fmt(row.mae_with, 4) }}</td>
            <td class="num" style="color:#e53e3e">{{ fmt(row.mae_without, 4) }}</td>
            <td class="num" style="color:#276749">{{ fmt(row.improvement, 1) }}%</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement,
  PointElement, LineElement, Tooltip, Legend
} from 'chart.js'
import { Bar as BarChart, Scatter as ScatterChart } from 'vue-chartjs'
ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, Tooltip, Legend)

const selectedDataset = ref('alibaba')
const datasets     = ref([])
const comparison   = ref([])
const horizonRows  = ref([])
const phiRows      = ref([])
const timeRows     = ref([])
const spikeRows    = ref([])
const retrainRows  = ref([])
const dataSource   = ref('')

// Состояние запущенного pytest-прогона: 'idle' | 'running' | 'done'
const runStatus = ref('idle')
const runInfo   = ref({ started_at: null, filter: '', count: null })
let runPoll = null

const fmt = (v, d) => v != null ? Number(v).toFixed(d) : '—'
const slaColor  = v => v > 5 ? '#e53e3e' : v > 3 ? '#dd6b20' : '#276749'
const coverageBadge = v => v >= 93 && v <= 97 ? 'badge badge-green' : 'badge badge-orange'

const methodColors = ['#38a169','#3182ce','#805ad5','#dd6b20','#e53e3e','#718096']

const maeChartData = computed(() => ({
  labels: comparison.value.map(r => r.method?.replace('Разработанный метод', '* Разработанный') || ''),
  datasets: [{
    label: 'MAE',
    data: comparison.value.map(r => r.mae ?? null),
    backgroundColor: comparison.value.map((_, i) => methodColors[i] || '#a0aec0'),
    borderRadius: 4,
  }]
}))
const maeOptions = {
  responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
  indexAxis: 'y',
  scales: {
    x: { ticks: { color: '#4a5568' }, grid: { color: '#edf2f7' },
         title: { display:true, text:'MAE', color:'#4a5568' } },
    y: { ticks: { color: '#4a5568', font:{size:11} }, grid: { color: '#edf2f7' } }
  },
  plugins: { legend: { display: false } }
}

const scatterData = computed(() => ({
  datasets: comparison.value
    .filter(r => r.sla_pct != null && r.avg_util != null)
    .map((r, i) => ({
      label: r.method,
      data: [{ x: r.sla_pct, y: r.avg_util }],
      backgroundColor: methodColors[comparison.value.indexOf(r)] || '#a0aec0',
      pointRadius: r.method?.includes('Разработанный') ? 10 : 6,
      pointStyle: r.method?.includes('Разработанный') ? 'star' : 'circle',
    }))
}))
const scatterOptions = {
  responsive: true, maintainAspectRatio: false, animation: { duration: 0 },
  scales: {
    x: { ticks: { color:'#4a5568', callback: v => v+'%' }, grid: { color:'#edf2f7' },
         title: { display:true, text:'Нарушения SLA, %', color:'#4a5568' }, min:0 },
    y: { ticks: { color:'#4a5568', callback: v => v+'%' }, grid: { color:'#edf2f7' },
         title: { display:true, text:'Средний CPU, %', color:'#4a5568' } }
  },
  plugins: { legend: { labels: { color:'#4a5568', font:{size:10} } } }
}

async function loadComparison() {
  try {
    const { data } = await axios.get(`/api/compare?dataset=${selectedDataset.value}`)
    comparison.value = data.results || []
    dataSource.value = data.source || ''
  } catch {}
}

async function loadHorizon() {
  try {
    const { data } = await axios.get('/api/horizon')
    horizonRows.value = data.results || []
  } catch {}
}

async function loadPhi() {
  try {
    const { data } = await axios.get('/api/phi')
    phiRows.value = data.results || []
  } catch {}
}

async function loadTiming() {
  try {
    const { data } = await axios.get('/api/timing')
    timeRows.value = data.results || []
  } catch {}
}

async function loadSpike() {
  try {
    const { data } = await axios.get('/api/spike')
    spikeRows.value = data.results || []
  } catch {}
}

async function loadRetrain() {
  try {
    const { data } = await axios.get('/api/retrain')
    retrainRows.value = data.results || []
  } catch {}
}

function loadAll() {
  loadComparison(); loadHorizon()
  loadPhi(); loadTiming(); loadSpike(); loadRetrain()
}

async function fetchDatasets() {
  try {
    const { data } = await axios.get('/api/datasets')
    datasets.value = data.datasets || []
  } catch {}
}

async function checkRunStatus() {
  try {
    const { data } = await axios.get('/api/experiments/status')
    if (data.started_at) {
      runInfo.value = {
        started_at: data.started_at,
        filter:     data.filter || '',
        count:      data.count,
      }
    }
    if (data.running) {
      runStatus.value = 'running'
    } else if (runStatus.value === 'running') {
      // прогон завершился — обновим все таблицы
      runStatus.value = 'done'
      runInfo.value.count = data.count
      loadAll()
    }
  } catch {}
}

async function triggerRun(filter) {
  // Защита от повторного нажатия
  if (runStatus.value === 'running') return
  try {
    await axios.post('/api/experiments/run', { filter })
    runStatus.value = 'running'
    runInfo.value = { started_at: new Date().toLocaleTimeString('ru-RU'),
                      filter, count: null }
    // Поллим статус каждые 5 секунд
    if (runPoll) clearInterval(runPoll)
    runPoll = setInterval(checkRunStatus, 5000)
  } catch (e) {
    runStatus.value = 'idle'
  }
}

onMounted(() => {
  fetchDatasets()
  loadAll()
  // Подхватываем «висящий» прогон, если страница перезагружена во время теста
  checkRunStatus()
})
onUnmounted(() => { if (runPoll) clearInterval(runPoll) })
</script>

<style scoped>
.page-header  { margin-bottom: 4px }
.page-title   { font-size: 20px; font-weight: 700; color: #1a202c }
.page-desc    { font-size: 13px; color: #718096; margin-top: 4px }
.toolbar      { display: flex; align-items: center; gap: 12px; flex-wrap: wrap }
.table-scroll { overflow-x: auto }
.text-muted   { color: #a0aec0 }
.chart-wrap   { position: relative; height: 40vh; max-height: 360px }
</style>
