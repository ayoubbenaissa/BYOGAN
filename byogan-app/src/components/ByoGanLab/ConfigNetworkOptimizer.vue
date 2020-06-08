<template>
    <v-app>
        <form @submit.prevent="onNetworkOptimizer">
            <v-flex class="d-inline-flex pa-2">
              <v-card-title
                class="headline grey lighten-2"
                primary-title
                >{{this.network}} Optim Configuration
                </v-card-title>
                <v-select
                  :items="this.$parent.OptimizerNames"
                  v-model="networkOptimizerName"
                  v-bind:label="`Optimizer of the ${this.network} :`"
                  outlined
                  required
                  style="height: 45px; margin-bottom: 10px"
                ></v-select>
                <v-text-field
                  v-model="networkLR"
                  v-bind:label="`Learning rate for ${this.network} optimizer:`"
                  outlined
                  required
                  style="height: 45px; margin-bottom: 10px"
                ></v-text-field>
                <v-text-field
                  v-model="networkBetaOne"
                  label="Beta1 value for Adam : "
                  v-show="nAdam"
                  outlined
                  required
                  style="height: 45px; margin-bottom: 10px"
                ></v-text-field>
                <v-text-field
                  v-model="networkBetaTwo"
                  label="Beta2 value for Adam : "
                  v-show="nAdam"
                  outlined
                  required
                  style="height: 45px; margin-bottom: 10px"
                ></v-text-field>
                <v-text-field
                  label="Epsilon (Numerical Stability) :"
                  outlined
                  v-model="nOptimEps"
                  v-show="nRMS || nAdam"
                  dense
                  style="margin-right: 10px"
                ></v-text-field>
                <v-text-field
                  label="weight decay (L2 penalty) :"
                  outlined
                  v-model="nOptimWeightDecay"
                  dense
                  style="margin-right: 10px"
                ></v-text-field>
                <v-text-field
                  label="momentum factor (default: 0) :"
                  outlined
                  v-show="nRMS || nSGD"
                  v-model="nOptimMomentum"
                  dense
                  style="margin-right: 10px"
                ></v-text-field>
                <v-text-field
                  label="smoothing constant (default: 0.99) :"
                  outlined
                  v-show="nRMS"
                  v-model="nOptimAlpha"
                  dense
                  style="margin-right: 10px"
                ></v-text-field>
                <div style="display: flex;" v-show="dSGD">
                  <v-switch 
                  v-model="nOptimNosterov" 
                  inset 
                  :label="`Nosterov Momentum`"></v-switch>
                  <v-btn icon @click="showoutOptimNosterov = !showoutOptimNosterov" style="margin-top: 16px; margin-right: 10px;">
                    <v-icon x-small>info</v-icon>
                    <md-tooltip :md-active.sync="showoutOptimNosterov">enables Nesterov momentum (default: False)</md-tooltip>
                  </v-btn>
                </div>
                <div style="display: flex;" v-show="nAdam">
                  <v-switch 
                  v-model="nOptimAMSGrad" 
                  inset 
                  :label="`AMSGrad`"></v-switch>
                  <v-btn icon @click="showoutOptimAMSGrad = !showoutOptimAMSGrad" style="margin-top: 16px; margin-right: 10px;">
                    <v-icon x-small>info</v-icon>
                    <md-tooltip :md-active.sync="showoutOptimAMSGrad">whether to use the AMSGrad variant of the algorithm from the paper `On the Convergence of Adam and Beyond`</md-tooltip>
                  </v-btn>
                </div>
                <div style="display: flex;" v-show="dRMS">
                  <v-switch 
                  v-model="nOptimCentered"
                  inset 
                  :label="`centered RMSProp`"></v-switch>
                  <v-btn icon @click="showoutOptimCentered = !showoutOptimCentered" style="margin-top: 16px; margin-right: 10px;">
                    <v-icon x-small>info</v-icon>
                    <md-tooltip :md-active.sync="showoutOptimCentered">compute the centered RMSProp, the gradient is normalized by an estimation of its variance</md-tooltip>
                  </v-btn>
                </div>
                <v-btn v-bind:id="`submitBtnD_optim_${this.network}`" dark class="green darken-1" type="submit">
                  {{ this.networkOptimizerState }}</v-btn>
            </v-flex>
          </form>
          <div>
                <v-snackbar 
                    v-model="createdNetworkOptimizer"
                    :multi-line="true"
                    :timeout="2500"
                    class="green darken-2"
                    >
                    {{ this.createdNetworkOptimizerMessageInfo }}
                <span> ... </span>
                <v-btn icon @click="showNetworkOptimizerInfo = true"><v-icon x-small>info</v-icon></v-btn>
                <v-btn icon @click="createdNetworkOptimizer = false"> <v-icon x-small right>mdi-close</v-icon> </v-btn>  
                </v-snackbar>
            </div>
            <v-dialog v-model="showNetworkOptimizerInfo" persistent width="700px">
                <v-card>
                        <v-card-title
                        class="headline grey lighten-2"
                        primary-title
                        >{{this.network}} Optimizer Info
                        </v-card-title>
                        <v-text-field
                        v-model="createdNetworkOptimizerMessageInfo"
                        label="Overview"
                        outlined
                        disabled
                        prepend-icon="info"
                        ></v-text-field>
                        <v-text-field
                        v-bind:label="`created ${this.network} Optimizer Info`"
                        outlined
                        disabled
                        multi-line
                        prepend-icon="info"
                        v-model="createdNetworkOptimizerDetailInfo"
                        ></v-text-field>
                        <v-btn @click="showNetworkOptimizerInfo = false" class="deep-orange accent-3">Close</v-btn>
                    </v-card>
            </v-dialog>
    </v-app>
</template>
<script>
export default {
  name: 'NetworkOptimizerConfig',
  props: {
    network: String
  },
  data: () => ({
    createdNetworkOptimizer: false,
    showNetworkOptimizerInfo: false,
    createdNetworkOptimizerMessageInfo: '',
    createdNetworkOptimizerDetailInfo: '',
    networkOptimizerName: '',
    networkOptimizerStateREQUEST: 'created',
    networkBetaOne: 0.5,
    networkBetaTwo: 0.999,
    nOptimEps: 0.00000001,
    networkLR: 0.0002,
    nOptimWeightDecay: 0,
    showoutOptimAMSGrad: false,
    nOptimAMSGrad: false,
    nOptimMomentum: 0,
    nOptimAlpha: 0.99,
    showoutOptimCentered: false,
    nOptimCentered: false,
    showoutOptimNosterov: false,
    nOptimNosterov: false,
    networkOptimizerState: (this.network === 'Discriminator') ? 'Instantiate D_Optim' : 'Instantiate G_Optim'
  }),
  computed: {
    nAdam () {
      return this.networkOptimizerName === 'Adam'
    },
    nRMS () {
      return this.networkOptimizerName === 'RMS'
    },
    nSGD () {
      return this.networkOptimizerName === 'SGD'
    }
  },
  methods: {
    onNetworkOptimizer () {
      if (this.network === 'Discriminator') {
        this.Url = this.$parent.apiDockerContainerIp + ':5000/discriminatoroptimizer'
      } else if (this.network === 'Generator') {
        this.Url = this.$parent.apiDockerContainerIp + ':5000/generatoroptimizer'
      }
      this.createdNetworkOptimizer = true
      const networkOptimizerElement = {
        name: this.networkOptimizerName,
        learning_rate: parseFloat(this.networkLR),
        beta1: parseFloat(this.networkBetaOne),
        beta2: parseFloat(this.networkBetaTwo),
        state: this.networkOptimizerStateREQUEST,
        epsilon: parseFloat(this.nOptimEps),
        weight_decay: parseFloat(this.nOptimWeightDecay),
        ams_grad: this.nOptimAMSGrad,
        momentum: parseFloat(this.nOptimMomentum),
        alpha: parseFloat(this.nOptimAlpha),
        centered_rms: this.nOptimCentered,
        nosterov: this.nOptimNosterov
      }
      this.$http.post(this.Url, networkOptimizerElement)
        .then(data => {
          document.getElementById('submitBtnD_optim_' + this.network).className = 'yellow darken-2 btn btn--raised theme--dark'
          this.networkOptimizerState = (this.network === 'Discriminator') ? 'Update D_Optim' : 'Update G_Optim'
          this.createdNetworkOptimizerMessageInfo = data.body.response
          this.createdNetworkOptimizerDetailInfo = data.body.optimizer
          this.networkOptimizerStateREQUEST = 'updated'
        })
        .catch(e => {
          this.$parent.errorConfig = e.statusText
          console.log('error: ', e)
        })
    }
  }
}
</script>
