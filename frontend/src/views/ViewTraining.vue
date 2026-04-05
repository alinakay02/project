<template>
  <div>
    <div class="page-header">
      <h2 class="page-title">Обучение нейросетевой модели</h2>
      <p class="page-desc">
        Визуализация процесса обучения GRU-компоненты гибридного прогнозировщика.
      </p>
    </div>

    <div class="grid-2 mt-4">
      <!-- ── Левая колонка: управление ───────────────────────────────────── -->
      <div>
        <div class="card">
          <div class="card-title">Запуск обучения</div>
          <div class="form-col">
            <label>Набор данных:</label>
            <select v-model="selectedDataset" class="select-wide">
              <option v-for="d in datasets" :key="d.id" :value="d.id"
                      :disabled="!d.available">
                {{ d.name }} {{ !d.available ? '(не загружен)' : '' }}
              </option>
            </select>
          </div>
          <div class="btn-row mt-4">
            <button class="btn btn-primary" :disabled="training.running" @click="startTraining">
              {{ training.running ? 'Обучение...' : 'Запустить обучение' }}
            </button>
          </div>
          <div v-if="training.running" class="progress-wrap mt-4">
            <div class="progress-label">
              Эпоха {{ training.epoch }} / {{ maxEpochs }}
              <span v-if="training.best_val">
                | best val_loss = {{ training.best_val.toFixed(4) }}
              </span>
            </div>
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: epochPct + '%' }"></div>
            </div>
          </div>
          <div v-if="!training.running && training.epoch > 0" class="result-box mt-4">
            <span class="badge badge-green">Обучение завершено</span>
            на эпохе {{ training.epoch }} (ранняя остановка, patience = 10)
            <div class="mt-8" style="font-size:12px;color:#718096">
              Итоговый val_loss = {{ training.best_val?.toFixed(4) || '—' }}
            </div>
          </div>
        </div>
      </div>

      <!-- ── Правая колонка: график потерь ─────────────────────────────── -->
      <div>
        <div class="card">
          <div class="card-title" style="display:flex;justify-content:space-between">
            <span>Динамика квантильных потерь по эпохам</span>
            <span v-if="training.epoch > 0" class="badge badge-blue">
              {{ training.epoch }} эп.
            </span>
          </div>
          <div v-if="training.train_loss.length === 0" class="empty-chart">
            Запустите обучение, чтобы увидеть динамику потерь
          </div>
          <div v-else class="chart-wrap">
            <LineChart :data="lossChartData" :options="lossOptions" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import { Chart as ChartJS, CategoryScale, LinearScale,
         PointElement, LineElement, Tooltip, Legend } from 'chart.js'
import { Line as LineChart } from 'vue-chartjs'
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend)

const training = ref({ running:false, epoch:0, train_loss:[], val_loss:[], best_val: null })
const datasets = ref([])
const selectedDataset = ref('mixed')
const maxEpochs = 100

const epochPct = computed(() => (training.value.epoch / maxEpochs) * 100)

const lossChartData = computed(() => ({
  labels: training.value.train_loss.map((_, i) => i + 1),
  datasets: [
    {
      label: 'Train loss',
      data: training.value.train_loss,
      borderColor: '#3182ce', borderWidth: 2,
      pointRadius: 1, fill: false, tension: .3,
    },
    {
      label: 'Val loss',
      data: training.value.val_loss,
      borderColor: '#e53e3e', borderWidth: 2,
      pointRadius: 1, fill: false, tension: .3,
    },
  ]
}))

const lossOptions = {
  responsive: true, maintainAspectRatio: false, animation: { duration: 0 },
  scales: {
    x: { ticks: { color: '#4a5568' }, grid: { color: '#edf2f7' },
         title: { display: true, text: 'Эпоха', color: '#4a5568' } },
    y: { ticks: { color: '#4a5568' }, grid: { color: '#edf2f7' },
         title: { display: true, text: 'Квантильная потеря', color: '#4a5568' } }
  },
  plugins: { legend: { labels: { color: '#4a5568', font: { size: 11 } } } }
}

async function startTraining() {
  try {
    await axios.post('/api/training/start', { dataset: selectedDataset.value })
  } catch {}
}

let timer = null
async function pollTraining() {
  try {
    const { data } = await axios.get('/api/training/status')
    training.value = data
  } catch {}
}

onMounted(async () => {
  try {
    const { data } = await axios.get('/api/datasets')
    datasets.value = data.datasets
  } catch {}
  pollTraining()
  timer = setInterval(pollTraining, 1000)
})
onUnmounted(() => clearInterval(timer))
</script>

<style scoped>
.page-header { margin-bottom: 4px }
.page-title  { font-size: 20px; font-weight: 700; color: #1a202c }
.page-desc   { font-size: 13px; color: #718096; margin-top: 4px }

.form-col { display: flex; flex-direction: column; gap: 6px }
.select-wide { width: 100%; min-width: 200px; padding: 8px 12px; font-size: 14px }
.btn-row  { display: flex; gap: 8px }

.progress-wrap  { }
.progress-label { font-size: 12px; color: #4a5568; margin-bottom: 6px }
.progress-bar   { height: 6px; background: #edf2f7; border-radius: 3px; overflow: hidden }
.progress-fill  { height: 100%; background: #3182ce; border-radius: 3px; transition: width .3s }

.result-box { background: #f0fff4; border: 1px solid #c6f6d5; border-radius: 6px; padding: 10px 12px;
              font-size: 13px; color: #276749 }

.chart-wrap { position: relative; height: 45vh; max-height: 400px }

.empty-chart { height: 45vh; max-height: 400px; display: flex; align-items: center; justify-content: center;
               color: #a0aec0; font-size: 13px; border: 1px dashed #e2e8f0; border-radius: 8px }
</style>
