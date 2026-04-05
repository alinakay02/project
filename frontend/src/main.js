import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'

import ViewMonitor   from './views/ViewMonitor.vue'
import ViewLoad      from './views/ViewLoad.vue'
import ViewTraining  from './views/ViewTraining.vue'
import ViewCompare   from './views/ViewCompare.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/',          redirect: '/monitor' },
    { path: '/monitor',   component: ViewMonitor,  name: 'monitor'  },
    { path: '/load',      component: ViewLoad,     name: 'load'     },
    { path: '/training',  component: ViewTraining, name: 'training' },
    { path: '/compare',   component: ViewCompare,  name: 'compare'  },
  ]
})

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
