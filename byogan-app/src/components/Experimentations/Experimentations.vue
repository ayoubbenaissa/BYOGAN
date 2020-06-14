<template>
  <v-app>
    <p>Experimentations</p>
    <v-list three-line>
      <template v-for="(item, index) in items">
        <v-subheader
          v-if="item.header"
          :key="item.header"
          v-text="item.header"
        ></v-subheader>

        <v-divider
          v-else-if="item.divider"
          :key="index"
          :inset="item.inset"
        ></v-divider>

        <v-list-item
          v-else
          :key="item.title"
        >
          <v-list-item-avatar>
            <div align="center" justify="center" style="width: 100%;">
            <img
              :src="item.avatar"
              />
            </div>
          </v-list-item-avatar>
          <v-spacer></v-spacer>

          <v-list-item-content>
            <v-list-item-title v-html="item.title"></v-list-item-title>
            <v-spacer></v-spacer>
            <v-list-item-subtitle v-html="item.subtitle"></v-list-item-subtitle>
          </v-list-item-content>
          <v-spacer></v-spacer>
          <v-divider
          v-if="item.divider"
          :key="index"
          :inset="item.inset"
        ></v-divider>
        <v-spacer></v-spacer>
        </v-list-item>
      </template>
    </v-list>
  </v-app>
</template>

<script>
export default {
  data: () => ({
    items: [
      { header: 'GAN main issues' },
      {
        avatar: '../../../static/D_outperforming_G.png',
        title: '<b>D outperforming G:</b>',
        subtitle: 'happens when D learns necessary patterns to differentiate between real and fake data, while G is not producing persuading data.'
      },
        { divider: true, inset: true },
      {
        avatar: '../../../static/g_fooling_d.png',
        title: '<b>G fooling D:</b>',
        subtitle: 'happens when D does learns necessary patterns to differentiate between real and fake data, so G can fool it with garbage.'
      },
      { divider: true, inset: true },
      {
        avatar: '../../../static/mode_collapse_example.png',
        title: '<b>Mode Collapse:</b>',
        subtitle: 'G produces realistci looking data, but it covers only a sub-branch from real data classes.'
      }
    ]
  })
}
</script>
