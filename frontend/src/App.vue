<template>
  <div class="app">
    <!-- ── Шапка ─────────────────────────────────────────────────────────── -->
    <header class="header">
      <div class="header-brand">
        <span class="brand-title">Прогнозирование нагрузки ИС</span>
        <span class="brand-sub">Экспериментальный стенд</span>
      </div>

      <nav class="nav">
        <router-link to="/monitor"  class="nav-item" active-class="nav-item--active">
          Мониторинг
        </router-link>
        <router-link to="/load"     class="nav-item" active-class="nav-item--active">
          Нагрузка
        </router-link>
        <router-link to="/training" class="nav-item" active-class="nav-item--active">
          Обучение
        </router-link>
        <router-link to="/compare"  class="nav-item" active-class="nav-item--active">
          Сравнение
        </router-link>
      </nav>

      <!-- Статус системы -->
      <div class="header-status">
        <span :class="['status-dot', statusClass]"></span>
        <span class="status-text">{{ statusText }}</span>
        <span class="status-iter">итерация {{ iterations }}</span>
      </div>
    </header>

    <!-- ── Контент ────────────────────────────────────────────────────────── -->
    <main class="main">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { computed, watch } from 'vue'
import { useApiStream } from './composables/useApiStream'

// Подписываемся на push-поток обновлений (SSE). Это заменяет прежний
// setInterval(poll, 5000): фронтенд получает новый снапшот состояния
// сразу, как только на сервере появляется новый сэмпл от поллера.
const { status } = useApiStream('/api/stream')

const iterations = computed(() => status.value?.iterations ?? 0)
const replicas   = computed(() => status.value?.replicas?.current ?? 2)
const rMax       = computed(() => status.value?.replicas?.r_max ?? 20)
const saturation = computed(() => status.value?.replicas?.saturation ?? false)

const statusClass = computed(() => {
  if (saturation.value) return 'status-dot--warn'
  if (replicas.value >= rMax.value - 1) return 'status-dot--alert'
  return 'status-dot--ok'
})
const statusText = computed(() => {
  if (saturation.value) return 'Насыщение ресурсов'
  return `${replicas.value} / ${rMax.value} реплик`
})
</script>

<style>
/* ── Сброс и базовые стили ──────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0 }
html, body { height: 100%; font-family: 'Segoe UI', system-ui, sans-serif;
             background: #f7fafc; color: #1a202c }
#app { height: 100% }

/* ── Шапка ──────────────────────────────────────────────────────────────── */
.app   { display: flex; flex-direction: column; min-height: 100vh }
.header {
  display: flex; align-items: center; gap: 24px;
  padding: 0 24px; height: 56px;
  background: #ffffff; border-bottom: 1px solid #e2e8f0;
  position: sticky; top: 0; z-index: 100;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.header-brand { display: flex; align-items: center; gap: 8px; min-width: 220px }
.brand-title  { font-size: 15px; font-weight: 700; color: #2b6cb0 }
.brand-sub    { font-size: 11px; color: #718096; margin-left: 4px }

/* Навигация */
.nav { display: flex; gap: 4px; flex: 1 }
.nav-item {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 14px; border-radius: 6px;
  text-decoration: none; color: #4a5568; font-size: 13px; font-weight: 500;
  transition: background .15s, color .15s;
}
.nav-item:hover { background: #edf2f7; color: #1a202c }
.nav-item--active { background: #ebf8ff; color: #2b6cb0 }

/* Статус */
.header-status { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #718096 }
.status-dot    { width: 8px; height: 8px; border-radius: 50% }
.status-dot--ok    { background: #48bb78; box-shadow: 0 0 6px #48bb78 }
.status-dot--warn  { background: #ed8936; box-shadow: 0 0 6px #ed8936 }
.status-dot--alert { background: #fc8181; box-shadow: 0 0 6px #fc8181 }
.status-text  { color: #1a202c; font-weight: 500 }
.status-iter  { color: #a0aec0; font-size: 11px }

/* ── Главный контент ─────────────────────────────────────────────────────── */
.main { flex: 1; padding: 20px 24px; overflow-y: auto }

/* ── Утилитарные классы ──────────────────────────────────────────────────── */
.card {
  background: #ffffff; border: 1px solid #e2e8f0;
  border-radius: 10px; padding: 18px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.card-title { font-size: 13px; font-weight: 600; color: #4a5568;
              text-transform: uppercase; letter-spacing: .06em; margin-bottom: 14px }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px }
.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px }
.mt-4  { margin-top: 16px }
.mt-8  { margin-top: 8px }

/* ── Кнопки ──────────────────────────────────────────────────────────────── */
.btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 16px; border-radius: 6px; border: none;
  font-size: 13px; font-weight: 600; cursor: pointer; transition: .15s;
}
.btn-primary { background: #3182ce; color: #fff }
.btn-primary:hover { background: #2b6cb0 }
.btn-success { background: #38a169; color: #fff }
.btn-success:hover { background: #2f855a }
.btn-danger  { background: #e53e3e; color: #fff }
.btn-danger:hover  { background: #c53030 }
.btn-ghost   { background: transparent; color: #4a5568; border: 1px solid #e2e8f0 }
.btn-ghost:hover { background: #edf2f7; color: #1a202c }
.btn:disabled { opacity: .45; cursor: not-allowed }

/* ── Бейджи ──────────────────────────────────────────────────────────────── */
.badge { display: inline-flex; padding: 2px 8px; border-radius: 99px; font-size: 11px; font-weight: 600 }
.badge-blue   { background: #ebf8ff; color: #2b6cb0 }
.badge-green  { background: #f0fff4; color: #276749 }
.badge-orange { background: #fffaf0; color: #c05621 }
.badge-red    { background: #fff5f5; color: #c53030 }
.badge-gray   { background: #edf2f7; color: #4a5568 }

/* ── Метрика-карточка ────────────────────────────────────────────────────── */
.metric-card { display: flex; flex-direction: column; gap: 4px }
.metric-label { font-size: 11px; color: #718096; text-transform: uppercase; letter-spacing: .05em }
.metric-value { font-size: 26px; font-weight: 700; color: #1a202c }
.metric-sub   { font-size: 11px; color: #a0aec0 }

/* ── Таблица ─────────────────────────────────────────────────────────────── */
.table { width: 100%; border-collapse: collapse; font-size: 13px }
.table th { padding: 10px 12px; background: #f7fafc; color: #4a5568;
             font-weight: 600; text-align: left; border-bottom: 2px solid #e2e8f0 }
.table td { padding: 10px 12px; border-bottom: 1px solid #edf2f7; color: #2d3748 }
.table tr:last-child td { border-bottom: none }
.table tr.best td { color: #276749 }
.table th.num, .table td.num { text-align: right; font-variant-numeric: tabular-nums }

/* ── Ползунок ────────────────────────────────────────────────────────────── */
input[type=range] { accent-color: #3182ce; width: 100% }
input[type=number] {
  background: #ffffff; border: 1px solid #e2e8f0; color: #1a202c;
  border-radius: 6px; padding: 6px 10px; font-size: 13px; width: 80px;
}
label { font-size: 12px; color: #4a5568 }
select {
  background: #ffffff; border: 1px solid #e2e8f0; color: #1a202c;
  border-radius: 6px; padding: 6px 10px; font-size: 13px;
}
</style>
