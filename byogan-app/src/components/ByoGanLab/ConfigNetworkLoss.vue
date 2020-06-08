<template>
    <v-app>
        <form @submit.prevent="onNetworkLoss">
            <v-flex class="d-inline-flex pa-2">
              <v-card-title
                class="headline grey lighten-2"
                primary-title
                >{{this.network}} Loss Configuration
                </v-card-title>
                <v-select
                  :items="networkLosses"
                  v-model="networkLoss"
                  v-bind:label="`${this.network} Loss function:`"
                  outlined
                  required
                  style="height: 45px; margin-bottom: 10px"
                ></v-select>
                <v-btn
                v-bind:id="`submitBtnD_loss_${this.network}`" dark class="green darken-1" type="submit">
                  {{ this.networkLossState }}</v-btn>
            </v-flex>
          </form>
          <div>
                <v-snackbar 
                    v-model="createdNetworkLoss"
                    :multi-line="true"
                    :timeout="2500"
                    class="green darken-2"
                    >
                    {{ this.createdNetworkLossMessageInfo }}
                <span> ... </span>
                <v-btn icon @click="showNetworkLossInfo = true"><v-icon x-small>info</v-icon></v-btn>
                <v-btn icon @click="createdNetworkLoss = false"> <v-icon x-small right>mdi-close</v-icon> </v-btn>  
                </v-snackbar>
            </div>
            <v-dialog v-model="showNetworkLossInfo" persistent width="700px">
                <v-card>
                        <v-card-title
                        class="headline grey lighten-2"
                        primary-title
                        >{{this.network}} Loss Info
                        </v-card-title>
                        <v-text-field
                        v-model="createdNetworkLossMessageInfo"
                        label="Overview"
                        outlined
                        disabled
                        prepend-icon="info"
                        ></v-text-field>
                        <v-text-field
                        v-bind:label="`${this.network} Loss Info:`"
                        outlined
                        disabled
                        prepend-icon="info"
                        v-model="createdNetworkLossDetailInfo"
                        ></v-text-field>
                        <v-btn @click="showNetworkLossInfo = false" class="deep-orange accent-3">Close</v-btn>
                    </v-card>
            </v-dialog>
    </v-app>
</template>
<script>
import { EventBus } from './event-bus.js'

export default {
  // Network loss, used for both networks (Generator & Discriminator)
  name: 'NetworkLossConfig',
  props: {
    network: String
  },
  data: () => ({
    createdNetworkLoss: false,
    createdNetworkLossDetailInfo: '',
    createdNetworkLossMessageInfo: '',
    showNetworkLossInfo: false,
    networkLossStateREQUEST: 'created',
    networkLosses: ['BCE', 'MSE'],
    networkLossState: (this.network === 'Discriminator') ? 'Instantiate D_Loss' : 'Instantiate G_Loss'
  }),
  created () {
    if (this.network === 'Discriminator') {
    // Listen for the i-got-clicked event and its payload.
      EventBus.$on('d-tanh-activation', outDiscriminator => {
        let allLosses = ['BCE', 'MSE']
        if (outDiscriminator) {
          allLosses = ['MSE']
        }
        this.networkLosses = allLosses
      })
    }
  },
  methods: {
    onNetworkLoss () {
      if (this.network === 'Discriminator') {
        this.Url = this.$parent.apiDockerContainerIp + ':5000/discriminatorloss'
      } else if (this.network === 'Generator') {
        this.Url = this.$parent.apiDockerContainerIp + ':5000/generatorloss'
      }
      const networkLossElement = {
        loss: this.networkLoss,
        state: this.networkLossStateREQUEST
      }
      this.$http.post(this.Url, networkLossElement)
        .then(data => {
          document.getElementById('submitBtnD_loss_' + this.network).className = 'yellow darken-2 btn btn--raised theme--dark'
          this.networkLossState = (this.network === 'Discriminator') ? 'Update D_Loss' : 'Update G_Loss'
          this.createdNetworkLoss = true
          this.createdNetworkLossMessageInfo = data.body.response
          this.createdNetworkLossDetailInfo = data.body.loss_function
          this.networkLossStateREQUEST = 'updated'
          if (this.createdNetworkLossDetailInfo === 'BCELoss()') {
            this.createdNetworkLossDetailInfo = 'Binary Cross Entropy Loss'
          } else if (this.createdNetworkLossDetailInfo === 'MSELoss()') {
            this.createdNetworkLossDetailInfo = 'Mean Squarred Error Loss'
          }
          var mseG = (this.network === 'Generator' && this.networkLoss === 'MSE')
          EventBus.$emit('mse-g', mseG)
        })
        .catch(e => {
          this.$parent.errorConfig = e.statusText
          console.log('error: ', e)
        })
    }
  }

}
</script>
