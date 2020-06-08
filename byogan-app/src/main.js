import Vue from 'vue'
import Vuetify from 'vuetify'
import App from './App'
import router from './router'
import VueResource from 'vue-resource'
import '@mdi/font/css/materialdesignicons.css'
import 'font-awesome/css/font-awesome.min.css'
import { MdTooltip } from 'vue-material/dist/components'
import 'vue-material/dist/vue-material.min.css'
import 'vue-material/dist/theme/default.css'

Vue.use(Vuetify, {
  iconfont: 'md' || 'mdi' || 'fa' || 'fa4',
  theme: {
    primary: '#FFA726',
    secondary: '#29B6F6',
    error: '#D32F2F'
  }
})
Vue.use(MdTooltip)
Vue.use(VueResource)
Vue.config.productionTip = false
Vue.config.silent = true

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  render: h => h(App)
})
