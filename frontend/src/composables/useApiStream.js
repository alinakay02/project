/**
 * useApiStream — композабл для push-подписки на обновления состояния
 * через Server-Sent Events (эндпоинт /api/stream).
 *
 * Использование:
 *   import { useApiStream } from '@/composables/useApiStream'
 *   const { status, connected } = useApiStream()
 *   // status.value — последний снапшот { mode, metrics, replicas, forecast, ... }
 *   // connected.value — true если соединение с SSE активно
 *
 * Автоматически подключается в onMounted, закрывается в onUnmounted.
 * При разрыве соединения браузер сам переподключается (EventSource native).
 */
import { ref, onMounted, onUnmounted } from 'vue'

export function useApiStream(url = '/api/stream') {
  const status = ref(null)       // последний снапшот (null до первого события)
  const connected = ref(false)   // состояние соединения
  const lastError = ref(null)    // последняя ошибка (для отладки)
  let es = null

  function connect() {
    if (es) return
    try {
      es = new EventSource(url)
    } catch (e) {
      lastError.value = String(e)
      return
    }

    es.addEventListener('open', () => {
      connected.value = true
      lastError.value = null
    })

    es.addEventListener('status', (event) => {
      try {
        status.value = JSON.parse(event.data)
      } catch (e) {
        lastError.value = `parse error: ${e}`
      }
    })

    es.addEventListener('error', (e) => {
      // EventSource сам переподключается через retry.
      connected.value = false
      lastError.value = 'connection lost, retrying...'
    })
  }

  function disconnect() {
    if (es) {
      es.close()
      es = null
      connected.value = false
    }
  }

  onMounted(connect)
  onUnmounted(disconnect)

  return { status, connected, lastError, reconnect: () => { disconnect(); connect() } }
}
