<template>
    <v-app>
      <v-dialog v-model="configuringGenerator" persistent width="500px" scrollable="false">
          <v-card>
            <v-container style="
                  grid-auto-flow: column;
                  display: flex;">
                <v-text-field
                    v-model="waitConfigGenerator"
                    style="height: 45px; margin-bottom: 10px"
                    disabled
                  ></v-text-field>
                <v-progress-circular
                  color="primary"
                  indeterminate
                  style="margin-left: 10px; margin-top: 15px;"
                ></v-progress-circular>
            </v-container>
          </v-card>
        </v-dialog>

          <v-dialog v-model="boolConfigGenerator" persistent width="1500px">
            <v-card v-show="!trainedModel">
            <form @submit.prevent="onCreateGenerator">
              <v-flex class="d-inline-flex pa-2">
                    <div id="generator">
                      <v-card-title
                          class="headline grey lighten-2"
                          primary-title
                        >Generator Configuration
                        </v-card-title>
                            <v-select
                            :items="this.$parent.networkModels"
                            v-model="generatorModel"
                            label="Generator Model: "
                            outlined
                            required
                            style="height: 45px; margin-bottom: 10px"
                            ></v-select>
                            <v-select
                              :items="nbLayersRange"
                              v-model="nbLayersGenerator"
                              label="Number Of hidden Layers: "
                              outlined
                              required
                              style="height: 45px; margin-bottom: 10px"
                            ></v-select>
                            <v-select
                            :items="this.$parent.inChannelsGeneratorRange"
                            v-model="inChannelsGenerator"
                            label="Input Dimension for Generator: "
                            outlined
                            style="height: 45px; margin-bottom: 10px"
                            ></v-select>
                            <v-select
                            :items="this.$parent.dropOutRange"
                            v-model="negativeSlopeGenerator"
                            label="Negative Slope Value in Networks: "
                            outlined
                            required
                            style="height: 45px; margin-bottom: 10px"
                            ></v-select>
                            <div style="display: flex;">
                            <td v-for="(index, pos) in nbLayersGenerator" :key="pos" width="100%">
                              <v-select
                                :items="dropOutRange"
                                v-model="dropOutGenerator[index-1]"
                                v-bind:label="`Drop Out Value for layer ${index} :`"
                                type="number"
                                outlined
                                required
                                style="height: 45px; margin-bottom: 10px; margin-right: 10px;"
                                ></v-select>
                             </td>
                            </div>
                            <div v-show="v_wArchitecture" style="display: flex;">
                              <td v-for="(index, pos) in nbLayersGenerator+1" :key="pos" width="100%">
                                <v-text-field
                                  v-bind:label="`Number Of neurons for layer ${index-1} :`"
                                  type="number"
                                  outlined
                                  required
                                  v-model="neuronLayers[index-1]"
                                  dense
                                  style="margin-right: 10px"
                                ></v-text-field>
                              </td>
                              </div>
                            <div v-show="dc_wArchitecture" style="display: flex;">
                              <td v-for="(index, pos) in nbLayersGenerator" :key="pos" width="100%">
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
                            <v-switch 
                              v-model="gpuGenerator" 
                              inset 
                              :label="`GPU`"></v-switch>
                              <div style="
                                  grid-auto-flow: column;
                                  display: flex;">
                                <v-select
                                  :items="InitializationOptions"
                                  v-model="initG"
                                  label="Initialize Generator Network:"
                                  outlined
                                  style="height: 45px; margin-bottom: 10px"
                                  ></v-select>
                                  <v-btn icon @click="showInitGInfo = !showInitGInfo" style="margin-top: 15px; margin-right: 40px;">
                                      <v-icon x-small>info</v-icon>
                                      <md-tooltip :md-active.sync="showInitGInfo">Will initialize the Generator weights according to the selected function</md-tooltip>
                                    </v-btn>    
                              </div>                     
                    </div>
                <v-card-actions>
                      <div class="text-center" style="display: flex;">
                      <v-btn 
                      :disabled="!validGeneratorForm"
                      type="submit"
                      @click="onToggleBoolConfigGenerator"
                      class="light-green accent-3">Instantiate Generator</v-btn>
                      <div>
                        <md-tooltip v-show="ImportTrainedModel">This allows you to import a trained model, but beforehand please make configuration of the trained Generator and Keep in mind that it has to be the same as for the trained model</md-tooltip>
                        <v-btn 
                        text 
                        @click="trainedModel = true"
                        :disabled="ImportTrainedModel"
                        class="grey lighten-1">Import trained Model</v-btn>
                      </div>
                      <v-btn class="deep-orange accent-3" text @click="onToggleBoolConfigGenerator">Close Dialog</v-btn>
                      </div>
                </v-card-actions>
              </v-flex>
              <small><b>*indicates required field</b></small>
            </form>
            </v-card>
            <v-card v-show="trainedModel">
              <form @submit.prevent="onTrainedGenerator">
              <v-flex class="d-inline-flex pa-2">
                      <v-card-title
                          class="headline grey lighten-2"
                          primary-title
                        >Trained Generator Configuration
                        </v-card-title>
                        <v-text-field
                          required
                          label="Trained Model Path: "
                          v-model="trainedGeneratorPath"
                          style="height: 45px; margin-bottom: 10px; width: 100%">
                        </v-text-field>
                        <v-switch 
                          v-model="gpuTrainedGenerator" 
                          inset 
                          :label="`GPU`">
                        </v-switch>
                        <v-card-actions>
                              <div class="text-center" style="display: flex;">
                              <v-btn 
                              :disabled="!validTrainedGeneratorForm"
                              type="submit"
                              @click="onToggleBoolConfigGenerator"
                              class="light-green accent-3">Import trained Generator</v-btn>
                              <v-btn text @click="trainedModel = false"
                              class="grey lighten-1">Return to configuration</v-btn>
                              <v-btn class="deep-orange accent-3" text @click="onToggleBoolConfigGenerator">Close Dialog</v-btn>
                              </div>
                        </v-card-actions>
              </v-flex>
              </form>
            </v-card>
          </v-dialog>
          <div>
          <v-snackbar 
            v-model="createdGenerator"
            :multi-line="true"
            :timeout="2500"
            class="green darken-2"
              >
               {{ this.createdGeneratorMessage }}
          <span> ... </span>
          <v-btn icon @click="showGeneratorInfo = true"><v-icon x-small>info</v-icon></v-btn>
          <v-btn icon @click="createdGenerator = false"> <v-icon x-small right>mdi-close</v-icon> </v-btn>  
          </v-snackbar>
        </div>
        <v-dialog v-model="showGeneratorInfo" persistent width="700px">
          <v-card>
            <v-card-title
              class="headline grey lighten-2"
              primary-title
            >Generator Info
            </v-card-title>
            <v-text-field
              v-model="createdGeneratorSummaryInfo"
              label="Generator Summary"
              multi-line
              outlined
              disabled
              prepend-icon="info"
            ></v-text-field>
            <v-text-field
              label="Generator Device"
              outlined
              disabled
              prepend-icon="info"
              v-model="createdGeneratorDeviceInfo"
            ></v-text-field>
            <div style="
                grid-auto-flow: column;
                display: flex;
                margin-left: 20px;"> 
              <v-text-field
                v-model="usedInitGInfo"
                label="used Initialization method:"
                outlined
                disabled
                style="height: 45px; margin-bottom: 10px"
                ></v-text-field>
                <v-btn v-show="usedInitGDescriptionInfo" icon @click="showUsedInitGInfo = !showUsedInitGInfo" style="margin-top: 15px; margin-right: 20px;">
                    <v-icon x-small>help</v-icon>
                    <md-tooltip :md-active.sync="showUsedInitGInfo">{{this.usedInitGDescriptionInfo}}</md-tooltip>
                  </v-btn>
            </div>
            <v-btn @click="showGeneratorInfo = false" class="deep-orange accent-3">Close</v-btn>
          </v-card>
        </v-dialog>
    </v-app>
</template>

<script>
import { EventBus } from './event-bus.js'

export default {
  name: 'ConfigGenerator',
  props: {
    boolConfigGenerator: Boolean
  },
  data: () => ({
    createdGenerator: false,
    ImportTrainedModel: true,
    gpuGenerator: false,
    // default type is Vanilla GAN:
    generatorModel: 'Vanilla GAN architecture',
    dropOutGenerator: [],
    nbLayersGenerator: 2,
    dropOutRange: [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.33],
    inChannelsGenerator: '',
    negativeSlopeGenerator: 0,
    createdGeneratorMessage: '',
    showGeneratorInfo: false,
    createdGeneratorSummaryInfo: '',
    createdGeneratorDeviceInfo: 'cpu',
    configuringGenerator: false,
    waitConfigGenerator: 'Please wait until configuration of Generator is done',
    neuronLayers: [],
    batchNLayers: [],
    trainedModel: false,
    trainedGeneratorPath: '',
    gpuTrainedGenerator: false,
    showInitGInfo: false,
    InitializationOptions: ['default', 'uniform', 'normal', 'Xavier uniform', 'Xavier normal', 'Kaiming uniform', 'Kaiming normal'],
    initG: 'default',
    showUsedInitGInfo: false,
    usedInitGInfo: '',
    usedInitGDescriptionInfo: ''

  }),
  computed: {
    validGeneratorForm () {
      // form is valid when all necessary fields are filled:
      return this.generatorModel !== '' &&
        this.dropOutGenerator !== '' &&
        this.nbLayersGenerator >= 2 && this.nbLayersGenerator <= 5 &&
        this.inChannelsGenerator !== ''
    },
    validTrainedGeneratorForm () {
      return this.trainedGeneratorPath !== ''
    },
    nbLayersRange () {
      return (this.generatorModel === 'Deep Convolutional GAN architecture') ? [2, 3, 4] : this.$parent.nbLayersRange
    },
    v_wArchitecture () {
      // ANN-based networks:
      return (this.generatorModel === 'Vanilla GAN architecture' || this.generatorModel === 'Wasserstein GAN architecture')
    },
    dc_wArchitecture () {
      //  networks using Batch Normalization:
      return (this.generatorModel === 'Deep Convolutional GAN architecture' || this.generatorModel === 'Wasserstein GAN architecture')
    },
    epsilonBatchN () {
      return Array(this.nbLayersGenerator).fill(0.8)
    },
    momentumBatchN () {
      return Array(this.nbLayersGenerator).fill(0.1)
    }
  },
  methods: {
    onToggleBoolConfigGenerator () {
      this.boolConfigGenerator = !this.boolConfigGenerator
      EventBus.$emit('toogle-bool-config-generator', this.boolConfigGenerator)
    },
    onCreateGenerator () {
      this.configuringGenerator = true
      this.neuronLayers = Array.from(this.neuronLayers, x => parseInt(x, 10))
      this.dropOutGenerator = Array.from(this.dropOutGenerator, x => parseFloat(x))
      for (var i = 0; i < this.nbLayersGenerator; i++) {
        if (this.batchNLayers[i] === undefined) {
          this.batchNLayers[i] = 0
        }
      }
      this.epsilonBatchN = Array.from(this.epsilonBatchN, x => isNaN(x) ? 0.8 : parseFloat(x))
      this.momentumBatchN = Array.from(this.momentumBatchN, x => isNaN(x) ? 0.1 : parseFloat(x))
      const definedGeneratorModels = ['VGAN', 'DCGAN', 'WGAN']
      const generatorModelREQUEST = definedGeneratorModels[this.$parent.networkModels.indexOf(this.generatorModel)]
      const deviceGenerator = (this.gpuGenerator) ? 'cuda: 0' : 'cpu'
      const generatorElement = {
        type: generatorModelREQUEST,
        drop_out: this.dropOutGenerator,
        leaky_relu: this.negativeSlopeGenerator,
        n_layers: this.nbLayersGenerator,
        device: deviceGenerator,
        input_channels: this.inChannelsGenerator,
        batch_norm: this.batchNLayers,
        eps: this.epsilonBatchN,
        momentum: this.momentumBatchN,
        layers: this.neuronLayers,
        generatorModel: this.generatorModel,
        init: this.initG
      }
      this.$http.post(this.$parent.apiDockerContainerIp + ':5000/generator', generatorElement)
            .then(data => {
              EventBus.$emit('created-generator-config', generatorElement)
              this.createdGenerator = true
              this.ImportTrainedModel = false
              this.createdGeneratorMessage = data.body.message
              this.createdGeneratorSummaryInfo = data.body.model
              this.createdGeneratorDeviceInfo = data.body.device
              this.usedInitGInfo = data.body.init
              this.usedInitGDescriptionInfo = data.body.initDescription
              this.configuringGenerator = false
            })
            .catch(e => {
              this.configuringGenerator = false
              this.$parent.errorConfig = e.statusText
              console.log('error: ', e)
            })
    },
    onTrainedGenerator () {
      this.configuringGenerator = true
      const trainedGeneratorElement = {
        path: this.trainedGeneratorPath,
        device: this.gpuTrainedGenerator
      }
      this.$http.post('http://127.0.0.1:5000/trainedgenerator', trainedGeneratorElement)
      .then(data => {
        this.createdGenerator = true
        this.createdGeneratorMessage = data.body.message
        this.createdGeneratorSummaryInfo = data.body.model
        this.createdGeneratorDeviceInfo = data.body.device
        this.configuringGenerator = false
      })
      .catch(e => {
        this.configuringGenerator = false
        this.$parent.errorConfig = e.statusText
        console.log('error: ', e)
      })
    }
  }
}
</script>
