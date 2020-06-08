import Vue from 'vue'
import Router from 'vue-router'
import Home from '@/components/Home'

import GetStarted from '@/components/GetStarted/GetStarted'
import ByoGanLab from '@/components/ByoGanLab/ByoGanLab'
import Experimentations from '@/components/Experimentations/Experimentations'
import Blogs from '@/components/Blogs/Blogs'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'Home',
      component: Home
    },
    {
      path: '/getstarted',
      name: 'GetStarted',
      component: GetStarted
    },
    {
      path: '/byoganlab',
      name: 'ByoGanLab',
      component: ByoGanLab
    },
    {
      path: '/experimentations',
      name: 'Experimentations',
      component: Experimentations
    },
    {
      path: '/blogs',
      name: 'Blogs',
      component: Blogs
    }
  ],
  mode: 'history'
})
