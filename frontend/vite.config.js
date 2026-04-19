import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5001',
        changeOrigin: true,
        // SSE (/api/stream) — долгоживущее соединение. Отключаем WS-upgrade
        // и любые таймауты, чтобы proxy не рвал поток событий.
        ws: false,
        timeout: 0,
        proxyTimeout: 0,
      },
    },
  },
})
