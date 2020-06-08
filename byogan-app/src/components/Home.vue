<template>
  <div>
    <h2>Overview</h2>
    <p>This is the Home Page of the Gan-Lab tool.</p>
    <p>Our tool is composed of four parts: </p>
    <ul>
      <li><a href="/getstarted">Get Started</a></li>
      <li><a href="/ganlab">Gan Lab</a></li>
      <li><a href="/experimentations">Experimentations</a></li>
      <li><a href="/blogs">Gan Blogs</a></li>
    </ul>

    <h2>Tool Purpose</h2>
    <p>Our tool provides an extended configuration of GANs and interactive visualization of their learning process</p>
  </div>
</template>

<script>
  export default {
    data () {
      return {
        item: 1,
        items: [
        { icon: 'home', title: 'Get Started', link: '/getstarted' },
        { icon: 'build', title: 'GAN LAB', link: '/ganlab' },
        { icon: 'assessment', title: 'Experimentations', link: '/experimentations' },
        { icon: 'forum', title: 'Gan Blogs', link: '/blogs' }
        ]
      }
    }
  }
</script>
