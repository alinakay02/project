<template>
  <div>
    <div class="page-header">
      <h2 class="page-title">Сравнение методов и анализ абляции</h2>
      <p class="page-desc">
        Сводные результаты экспериментов.
      </p>
    </div>

    <!-- Панель управления -->
    <div class="toolbar mt-4">
      <label>Набор данных:</label>
      <select v-model="selectedDataset" @change="loadComparison">
        <option value="alibaba">Alibaba Cluster Trace 2018</option>
        <option value="stationary">Стационарный</option>
        <option value="trend">Трендовый</option>
        <option value="spike">Всплесковый</option>
        <option value="mixed">Смешанный</option>
      </select>
      <button class="btn btn-primary" :disabled="runStatus === 'running'" @click="triggerRun('')">
        {{ runStatus === 'running' ? 'Тесты выполняются...' : 'Запустить все тесты' }}
      </button>
      <button class="btn btn-ghost" :disabled="runStatus === 'running'" @click="triggerRun('TestComputationTime')">
        Только время
      </button>
      <button class="btn btn-ghost" :disabled="runStatus === 'running'" @click="triggerRun('TestAblation')">
        Только абляция
      </button>
      <span v-if="dataSource === 'db'" class="badge badge-green">данные из БД</span>
      <span v-else-if="dataSource === 'empty'" class="badge badge-orange">нет данных — запустите тесты</span>
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
              <th class="num">Средняя утилизация, %</th>
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

      <!-- ── Scatter: нарушения SLA vs утилизация ──────────────────────────  -->
      <div class="card">
        <div class="card-title">Нарушения SLA и утилизация</div>
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

    <!-- ── Анализ абляции ─────────────────────────────────────────────────── -->
    <div class="card mt-4" v-if="ablation.length">
      <div class="card-title">Анализ вклада компонентов метода</div>
      <table class="table">
        <thead>
          <tr>
            <th>Конфигурация</th>
            <th class="num">MAE</th>
            <th class="num">Детали</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, i) in ablation" :key="row.config" :class="{ best: i === 0 }">
            <td>
              <span v-if="i === 0" class="badge badge-green">*</span>
              {{ row.config }}
            </td>
            <td class="num" :style="{ color: i===0 ? '#276749' : '' }">
              {{ fmt(row.mae, 3) }}
            </td>
            <td class="num" style="font-size:11px;color:#718096">
              {{ ablationDetails(row) }}
            </td>
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
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement,
  PointElement, LineElement, Tooltip, Legend
} from 'chart.js'
import { Bar as BarChart, Scatter as ScatterChart } from 'vue-chartjs'
ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, Tooltip, Legend)

const selectedDataset = ref('alibaba')
const comparison   = ref([])
const ablation     = ref([])
const horizonRows  = ref([])
const phiRows      = ref([])
const timeRows     = ref([])
const spikeRows    = ref([])
const retrainRows  = ref([])
const runStatus    = ref('idle')
const dataSource   = ref('')

const fmt = (v, d) => v != null ? Number(v).toFixed(d) : '—'
const slaColor  = v => v > 5 ? '#e53e3e' : v > 3 ? '#dd6b20' : '#276749'
const coverageBadge = v => v >= 93 && v <= 97 ? 'badge badge-green' : 'badge badge-orange'

function ablationDetails(row) {
  const m = row.metrics || {}
  if (m.worsening_pct) return `Ухудшение MAE: +${Number(m.worsening_pct).toFixed(1)}%`
  if (m.sla_without) return `SLA: ${m.sla_with}% → ${m.sla_without}%`
  if (m.scale_ops_without) return `Ops: ${m.scale_ops_with} → ${m.scale_ops_without}`
  if (m.util_fixed) return `Util: ${m.util_adaptive}% → ${m.util_fixed}%`
  return ''
}

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
         title: { display:true, text:'Средняя утилизация, %', color:'#4a5568' } }
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

async function loadAblation() {
  try {
    const { data } = await axios.get('/api/ablation')
    ablation.value = data.results || []
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
  loadComparison(); loadAblation(); loadHorizon()
  loadPhi(); loadTiming(); loadSpike(); loadRetrain()
}

async function triggerRun(filter) {
  runStatus.value = 'running'
  try {
    await axios.post('/api/experiments/run', { filter })
    // Поллинг статуса
    const poll = setInterval(async () => {
      try {
        const { data } = await axios.get('/api/experiments/status')
        if (!data.running) {
          clearInterval(poll)
          runStatus.value = 'done'
          loadAll()  // Обновляем все данные
        }
      } catch {}
    }, 5000)
  } catch {
    runStatus.value = 'idle'
  }
}

onMounted(loadAll)
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
