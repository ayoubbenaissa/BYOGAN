<template>
    <v-app>
      <v-dialog v-model="configuringDiscriminator" persistent width="500px" scrollable="false">
          <v-card>
            <v-container style="
                  grid-auto-flow: column;
                  display: flex;">
              <v-text-field
                  v-model="waitConfigDiscriminator"
                  style="height: 45px; margin-bottom: 10px"
                  disabled
                ></v-text-field>
              <v-progress-circular
                color="primary"
                indeterminate
                style="margin-left: 10px; margin-top: 15px"
              ></v-progress-circular>
            </v-container>
            </v-card>
        </v-dialog>

          <v-dialog v-model="boolConfigDiscriminator" persistent width="1500px">
            <v-card v-show="!trainedModel">
            <form @submit.prevent="onCreateDiscriminator">
                <v-flex class="d-inline-flex pa-2">
                  <div id="discriminator" style="width: 100%;">
                        <v-card-title
                          class="headline grey lighten-2"
                          primary-title
                        >Discriminator Configuration
                        </v-card-title>
                        <v-select
                          :items="this.$parent.networkModels"
                          v-model="discriminatorModel"
                          label="Discriminator Model: "
                          outlined
                          required
                          style="height: 45px; margin-bottom: 10px"
                          ></v-select>
                        <v-select
                          :items="nbLayersRange"
                          v-model="nbLayersDiscriminator"
                          label="Number Of hidden Layers: "
                          outlined
                          required
                          style="height: 45px; margin-bottom: 10px"
                          ></v-select>
                          <v-select
                          :items="this.$parent.dropOutRange"
                          v-model="negativeSlopeDiscriminator"
                          label="Negative Slope Value in Networks: "
                          outlined
                          required
                          style="height: 45px; margin-bottom: 10px"
                          ></v-select>
                          <div style="display: flex;">
                            <td v-for="(index, pos) in nbLayersDiscriminator" :key="pos" width="100%">
                              <v-select
                                :items="dropOutRange"
                                v-model="dropOutDiscriminator[index-1]"
                                v-bind:label="`Drop Out Value for layer ${index} :`"
                                type="number"
                                outlined
                                required
                                style="height: 45px; margin-bottom: 10px; margin-right: 10px;"
                                ></v-select>
                             </td>
                            </div>
                          <div v-show="v_wArchitecture" style="display: flex;">
                              <td v-for="(index, pos) in nbLayersDiscriminator+1" :key="pos" width="100%">
                                <v-text-field
                                  v-bind:label="`Number Of neurons for layer ${index-1} :`"
                                  type="number"
                                  outlined
                                  v-model="neuronLayers[index-1]"
                                  dense
                                  style="margin-right: 10px"
                                ></v-text-field>
                              </td>
                              </div>
                            <div v-show="dc_wArchitecture" style="display: flex;">
                              <td v-for="(index, pos) in nbLayersDiscriminator" :key="pos" width="100%">
                                <v-switch
                                  v-bind:label="`use BatchNorm for layer ${index} :`"
                                  outlined
                                  v-model="batchNLayers[index-1]"
                                  dense
                                  style="margin-right: 10px"
                                ></v-switch>
                                <v-text-field
                                  v-bind:label="`Epsilon for BatchNorm ${index} :`"
                                  outlined
                                  dense
                                  v-model="epsilonBatchN[index-1]"
                                  v-show="batchNLayers[index-1]"
                                  style="margin-right: 10px"
                                ></v-text-field>
                                <v-text-field
                                  v-bind:label="`Momentum for BatchNorm ${index} :`"
                                  outlined
                                  dense
                                  v-model="momentumBatchN[index-1]"
                                  v-show="batchNLayers[index-1]"
                                  style="margin-right: 10px"
                                ></v-text-field>                                                                
                              </td>
                              </div>
                          <div style="display: flex;">
                            <v-switch 
                            v-model="gpuDiscriminator" 
                            inset 
                            :label="`GPU`"></v-switch>
                            <v-btn icon @click="showoutDiscriminatorLabelInfo = !showoutDiscriminatorLabelInfo" style="margin-top: 16px; margin-right: 10px;">
                              <v-icon x-small>info</v-icon>
                              <md-tooltip :md-active.sync="showoutDiscriminatorLabelInfo">sets the output layer of the discriminator (<strong>Sigmoid</strong> => [0...1]/<strong>Tanh</strong> => [-1...1])</md-tooltip>
                            </v-btn>
                            <v-switch 
                            v-model="outDiscriminator"
                            :label="`${outDiscriminatorLabel}`"></v-switch>
                          </div>
                          <div style="
                                  grid-auto-flow: column;
                                  display: flex;">
                                <v-select
                                  :items="InitializationOptions"
                                  v-model="initD"
                                  label="Initialize Discriminator Network:"
                                  outlined
                                  style="height: 45px; margin-bottom: 10px"
                                  ></v-select>
                                  <v-btn icon @click="showInitDInfo = !showInitDInfo" style="margin-top: 15px; margin-right: 40px;">
                                      <v-icon x-small>info</v-icon>
                                      <md-tooltip :md-active.sync="showInitDInfo">Will initialize the Discriminator weights according to the selected function</md-tooltip>
                                    </v-btn>    
                              </div>
                      </div>
                      <v-card-actions>
                      <div class="text-center" style="display: flex;">
                        <div>
                          <md-tooltip>This imports symmetric configuration based on Generator configuration</md-tooltip>
                        <v-btn 
                          @click="onImportGeneratorConfig"
                          class="light-blue accent-3"> Import Gonfiguration from Generator </v-btn>
                        </div>
                      <v-btn 
                      :disabled="!validDiscriminatorForm"
                      type="submit"
                      @click="onToggleBoolConfigDiscriminator"
                      class="light-green accent-3">Instantiate Discriminator</v-btn>
                      <div>
                        <md-tooltip v-show="ImportTrainedModel">This allows you to import a trained model, but beforehand please make configuration of the trained Discriminator and Keep in mind that it has to be the same as for the trained model</md-tooltip>
                        <v-btn 
                        text 
                        @click="trainedModel = true"
                        :disabled="ImportTrainedModel"
                        class="grey lighten-1">Import trained Model</v-btn>
                      </div>
                      <v-btn class="deep-orange accent-3" text @click="onToggleBoolConfigDiscriminator">Close Dialog</v-btn>
                      </div>
                    </v-card-actions>
                </v-flex>
                <small>*indicates required field</small>
            </form>
            </v-card>
            <v-card v-show="trainedModel">
              <form @submit.prevent="onTrainedDiscriminator">
              <v-flex class="d-inline-flex pa-2">
                      <v-card-title
                          class="headline grey lighten-2"
                          primary-title
                        >Trained Discriminator Configuration
                        </v-card-title>
                        <v-text-field
                          required
                          label="Trained Model Path: "
                          v-model="trainedDiscriminatorPath"
                          style="height: 45px; margin-bottom: 10px; width: 100%">
                        </v-text-field>
                        <v-switch 
                          v-model="gpuTrainedDiscriminator" 
                          inset 
                          :label="`GPU`">
                        </v-switch>
                        <v-card-actions>
                              <div class="text-center" style="display: flex;">
                              <v-btn 
                              :disabled="!validTrainedDiscriminatorForm"
                              type="submit"
                              @click="onToggleBoolConfigDiscriminator"
                              class="light-green accent-3">Import trained Discriminator</v-btn>
                              <v-btn text @click="trainedModel = false" class="grey lighten-1">Return to configuration</v-btn>
                              <v-btn class="deep-orange accent-3" text @click="onToggleBoolConfigDiscriminator">Close Dialog</v-btn>
                              </div>
                        </v-card-actions>
              </v-flex>
              </form>
            </v-card>
          </v-dialog>
          <div>
          <v-snackbar 
            v-model="createdDiscriminator"
            :multi-line="true"
            :timeout="2500"
            class="green darken-2"
              >
               {{ this.createdDiscriminatorMessageInfo }}
          <span> ... </span>
          <v-btn icon @click="showDiscriminatorInfo = true"><v-icon x-small>info</v-icon></v-btn>
          <v-btn icon @click="createdDiscriminator = false"> <v-icon x-small right>mdi-close</v-icon> </v-btn>  
          </v-snackbar>
        </div>
        <v-dialog v-model="showDiscriminatorInfo" persistent width="700px" height="700px">
          <v-card>
            <v-card-title
              class="headline grey lighten-2"
              primary-title
            >Created Discriminator Info
            </v-card-title>
            <v-text-field
              v-model="createdDiscriminatorSummaryInfo"
              label="Created Discriminator Summary"
              multi-line
              outlined
              disabled
              prepend-icon="info"
              style="height: 100px; margin-buttom: 30px;"
            ></v-text-field>
            <v-text-field
              label="created Discriminator Device"
              outlined
              disabled
              prepend-icon="info"
              style="margin-top: 40px;"
              v-model="createdDiscriminatorDeviceInfo"
            ></v-text-field>
            <div style="
                grid-auto-flow: column;
                display: flex;
                margin-left: 20px;"> 
              <v-text-field
                v-model="usedInitDInfo"
                label="used Initialization method:"
                outlined
                disabled
                style="height: 45px; margin-bottom: 10px"
                ></v-text-field>
                <v-btn icon @click="showUsedInitDInfo = !showUsedInitDInfo" style="margin-top: 15px; margin-right: 20px;">
                    <v-icon x-small>help</v-icon>
                    <md-tooltip :md-active.sync="showUsedInitDInfo">{{this.usedInitDDescriptionInfo}}</md-tooltip>
                  </v-btn>
            </div>
            <v-btn @click="showDiscriminatorInfo = false" class="deep-orange accent-3">Close</v-btn>
          </v-card>
        </v-dialog>
    </v-app>
</template>
<script>
import { EventBus } from './event-bus.js'
export default {
  name: 'ConfigDiscriminator',
  props: {
    boolConfigDiscriminator: Boolean
  },
  data: () => ({
    createdDiscriminator: false,
    ImportTrainedModel: true,
    gpuDiscriminator: false,
    discriminatorModel: 'Vanilla GAN architecture',
    dropOutDiscriminator: [],
    nbLayersDiscriminator: 2,
    dropOutRange: [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.33],
    negativeSlopeDiscriminator: 0,
    createdDiscriminatorMessageInfo: '',
    showDiscriminatorInfo: false,
    createdDiscriminatorSummaryInfo: '',
    createdDiscriminatorDeviceInfo: 'cpu',
    configuringDiscriminator: false,
    waitConfigDiscriminator: 'Please wait until configuration of Discriminator is done',
    neuronLayers: [],
    batchNLayers: [],
    trainedModel: false,
    trainedDiscriminatorPath: '',
    gpuTrainedDiscriminator: false,
    outDiscriminator: false,
    showoutDiscriminatorLabelInfo: false,
    importedGeneratorConfig: {},
    showInitDInfo: false,
    InitializationOptions: ['default', 'uniform', 'normal', 'Xavier uniform', 'Xavier normal', 'Kaiming uniform', 'Kaiming normal'],
    showUsedInitDInfo: false,
    usedInitDInfo: '',
    usedInitDDescriptionInfo: ''
  }),
  created () {
    EventBus.$on('created-generator-config', generatorElement => {
      this.importedGeneratorConfig.discriminatorModel = generatorElement.generatorModel
      this.importedGeneratorConfig.nbLayersDiscriminator = generatorElement.n_layers
      this.importedGeneratorConfig.gpuDiscriminator = (generatorElement.device === 'cuda: 0')
      this.importedGeneratorConfig.negativeSlopeDiscriminator = generatorElement.leaky_relu
      const tmpDropOut = Array.from(generatorElement.drop_out, x => parseFloat(x))
      this.importedGeneratorConfig.dropOutDiscriminator = tmpDropOut.reverse()
      this.importedGeneratorConfig.batchNLayers = generatorElement.batch_norm.reverse()
      const tmpEps = Array.from(generatorElement.eps, x => isNaN(x) ? 0.8 : parseFloat(x))
      this.importedGeneratorConfig.epsilonBatchN = tmpEps.reverse()
      const tmpMomentum = Array.from(generatorElement.momentum, x => isNaN(x) ? 0.1 : parseFloat(x))
      this.importedGeneratorConfig.momentumBatchN = tmpMomentum.reverse()
      const tmpNeurons = Array.from(generatorElement.layers, x => parseInt(x, 10))
      this.importedGeneratorConfig.neuronLayers = tmpNeurons.reverse()
      this.importedGeneratorConfig.init = generatorElement.init
    })
  },
  computed: {
    validDiscriminatorForm () {
      return this.discriminatorModel !== '' &&
        this.dropOutDiscriminator !== '' &&
        this.nbLayersDiscriminator >= 2 && this.nbLayersDiscriminator <= 5
    },
    validTrainedDiscriminatorForm () {
      return this.trainedDiscriminatorPath !== ''
    },
    nbLayersRange () {
      return (this.discriminatorModel === 'Deep Convolutional GAN architecture') ? [2, 3, 4] : this.$parent.nbLayersRange
    },
    v_wArchitecture () {
      return (this.discriminatorModel === 'Vanilla GAN architecture' || this.discriminatorModel === 'Wasserstein GAN architecture')
    },
    dc_wArchitecture () {
      return (this.discriminatorModel === 'Deep Convolutional GAN architecture' || this.discriminatorModel === 'Wasserstein GAN architecture')
    },
    epsilonBatchN () {
      return Array(this.nbLayersDiscriminator).fill(0.8)
    },
    momentumBatchN () {
      return Array(this.nbLayersDiscriminator).fill(0.1)
    },
    outDiscriminatorLabel () {
      return (this.outDiscriminator) ? 'Tanh' : 'Sigmoid'
    }
  },
  methods: {
    onImportGeneratorConfig () {
      this.discriminatorModel = this.importedGeneratorConfig.discriminatorModel
      this.nbLayersDiscriminator = this.importedGeneratorConfig.nbLayersDiscriminator
      this.dropOutDiscriminator = this.importedGeneratorConfig.dropOutDiscriminator
      this.gpuDiscriminator = this.importedGeneratorConfig.gpuDiscriminator
      this.negativeSlopeDiscriminator = this.importedGeneratorConfig.negativeSlopeDiscriminator
      this.neuronLayers = this.importedGeneratorConfig.neuronLayers
      this.batchNLayers = this.importedGeneratorConfig.batchNLayers
      this.epsilonBatchN = []
      this.momentumBatchN = []
      this.initD = this.importedGeneratorConfig.init
      for (var i = 0; i < this.nbLayersDiscriminator; i++) {
        this.epsilonBatchN[i] = this.importedGeneratorConfig.epsilonBatchN[i]
        this.momentumBatchN[i] = this.importedGeneratorConfig.momentumBatchN[i]
      }
    },
    onToggleBoolConfigDiscriminator () {
      this.boolConfigDiscriminator = !this.boolConfigDiscriminator
      EventBus.$emit('toogle-bool-config-discriminator', this.boolConfigDiscriminator)
    },
    onCreateDiscriminator () {
      this.configuringDiscriminator = true
      // number of units per hidden layer for ANN-based network:
      this.neuronLayers = Array.from(this.neuronLayers, x => parseInt(x, 10))
      this.dropOutDiscriminator = Array.from(this.dropOutDiscriminator, x => parseFloat(x))
      this.batchNLayers = Array.from(this.batchNLayers, x => (x) ? 1 : 0)
      this.epsilonBatchN = Array.from(this.epsilonBatchN, x => isNaN(x) ? 0.8 : parseFloat(x))
      this.momentumBatchN = Array.from(this.momentumBatchN, x => isNaN(x) ? 0.1 : parseFloat(x))
      const definedDiscriminatorModels = ['VGAN', 'DCGAN', 'WGAN']
      const discriminatorModelREQUEST = definedDiscriminatorModels[this.$parent.networkModels.indexOf(this.discriminatorModel)]
      const deviceDiscriminator = (this.gpuDiscriminator) ? 'cuda: 0' : 'cpu'
      const discriminatorElement = {
        type: discriminatorModelREQUEST,
        drop_out: this.dropOutDiscriminator,
        leaky_relu: this.negativeSlopeDiscriminator,
        n_layers: this.nbLayersDiscriminator,
        device: deviceDiscriminator,
        input_channels: this.inChannelsGenerator,
        batch_norm: this.batchNLayers,
        eps: this.epsilonBatchN,
        momentum: this.momentumBatchN,
        layers: this.neuronLayers,
        out_activation: this.outDiscriminator,
        init: this.initD
      }
      this.$http.post(this.$parent.apiDockerContainerIp + ':5000/discriminator', discriminatorElement)
        .then(data => {
          this.createdDiscriminator = true
          this.ImportTrainedModel = false
          this.createdDiscriminatorMessageInfo = data.body.message
          this.createdDiscriminatorSummaryInfo = data.body.model
          this.createdDiscriminatorDeviceInfo = data.body.device
          this.configuringDiscriminator = false
          this.usedInitDInfo = data.body.init
          this.usedInitDDescriptionInfo = data.body.initDescription
          // Send the event on a channel (d-tanh-activation) with a payload
          EventBus.$emit('d-tanh-activation', this.outDiscriminator)
        })
        .catch(e => {
          this.$parent.errorConfig = e.statusText
          this.configuringDiscriminator = false
          console.log('error: ', e)
        })
    },
    onTrainedDiscriminator () {
      this.configuringDiscriminator = true
      const trainedDiscriminatorElement = {
        path: this.trainedDiscriminatorPath,
        device: this.gpuTrainedDiscriminator
      }
      this.$http.post('http://127.0.0.1:5000/traineddiscriminator', trainedDiscriminatorElement)
      .then(data => {
        this.createdDiscriminator = true
        this.createdDiscriminatorMessageInfo = data.body.message
        this.createdDiscriminatorSummaryInfo = data.body.model
        this.createdDiscriminatorDeviceInfo = data.body.device
        this.configuringDiscriminator = false
      })
      .catch(e => {
        this.configuringDiscriminator = false
        this.$parent.errorConfig = e.statusText
        console.log('error: ', e)
      })
    }
  }
}
</script>
