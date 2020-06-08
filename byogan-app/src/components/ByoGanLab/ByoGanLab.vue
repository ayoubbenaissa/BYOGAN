<template>
  <v-app>
    <v-dialog v-model="showTrackInfo" persistent width="700px" height="1000px">
      <v-card style="height: 90%;">
      <v-text-field
        multi-line
        disabled
        v-model="trackInfo"></v-text-field>
      <v-btn @click="showTrackInfo = false">close</v-btn>
      </v-card>
    </v-dialog>
    <v-dialog v-model="showSavedTrainedModelInfo" persistent width="700px">
          <v-card>
            <v-card-title
              class="headline grey lighten-2"
              primary-title
            >trained {{ this.savedTrainedModel }} Info
            </v-card-title>
            <v-text-field
              label="Response Message: "
              outlined
              disabled
              prepend-icon="info"
              v-model="savedTrainedModelMessageInfo"
            ></v-text-field>
            <v-text-field
              v-model="savedTrainedModelSummaryInfo"
              label="trained model Summary"
              multi-line
              outlined
              disabled
              prepend-icon="info"
            ></v-text-field>
            <v-btn @click="showSavedTrainedModelInfo = false">Close</v-btn>
          </v-card>
        </v-dialog>
    <v-dialog v-model="SaveModels" persistent width="500px" scrollable="false">
          <v-card v-show="saveGenerator">
            <form @submit.prevent="onSaveGenerator">
              <v-flex class="d-inline-flex pa-2">
                <v-card-title
                  class="headline grey lighten-2"
                  primary-title
                >Save Trained Generator
                </v-card-title>
                <v-container style="
                      grid-auto-flow: column;
                      display: flex;">
                  <v-text-field
                    label="Path"
                    v-model="trainedGPath"
                    style="height: 45px; margin-bottom: 10px; width: 100%">
                  </v-text-field>
                  <v-btn icon @click="showTrainedGPath = !showTrainedGPath" style="margin-top: 15px; margin-right: 40px;">
                    <v-icon x-small>info</v-icon>
                    <md-tooltip :md-active.sync="showTrainedGPath">A common PyTorch convention is to save models using either a <strong>.pt</strong> or <strong>.pth</strong> file extension</md-tooltip>
                  </v-btn>
                </v-container>
                <v-card-actions>
                  <v-btn 
                  :disabled="trainedGPath === ''"
                  type="submit"
                  @click="SaveModels = false">Save Generator</v-btn>
                  <v-btn
                  @click="saveGenerator = false">Go to save Discriminator</v-btn>
                  <v-btn @click="SaveModels = false">Close Dialog</v-btn>
                </v-card-actions>
              </v-flex>
            </form>
          </v-card>
          <v-card v-show="!saveGenerator">
            <form @submit.prevent="onSaveDiscriminator">
              <v-flex class="d-inline-flex pa-2">
                <v-card-title
                  class="headline grey lighten-2"
                  primary-title
                >Save Trained Discriminator
                </v-card-title>
                <v-container style="
                      grid-auto-flow: column;
                      display: flex;">
                  <v-text-field
                    label="Path"
                    v-model="trainedDPath"
                    style="height: 45px; margin-bottom: 10px; width: 100%">
                  </v-text-field>
                  <v-btn icon @click="showTrainedDPath = !showTrainedDPath" style="margin-top: 15px; margin-right: 40px;">
                    <v-icon x-small>info</v-icon>
                    <md-tooltip :md-active.sync="showTrainedDPath">A common PyTorch convention is to save models using either a <strong>.pt</strong> or <strong>.pth</strong> file extension</md-tooltip>
                  </v-btn>
                </v-container>
                <v-card-actions>
                  <v-btn 
                  :disabled="trainedDPath === ''"
                  type="submit"
                  @click="SaveModels = false">Save Discriminator</v-btn>
                  <v-btn
                  @click="saveGenerator = true">Go to save Generator</v-btn>
                  <v-btn @click="SaveModels = false">Close Dialog</v-btn>
                </v-card-actions>
              </v-flex>
            </form>
          </v-card>
        </v-dialog>
      <v-dialog v-model="implementTricks" persistent width="500px">
        <v-card>
          <form @submit.prevent="onTrainingTricks">
            <div style="
                    grid-auto-flow: column;
                    display: flex;">
                  <v-switch 
                    v-model="applyDivideDcost" 
                    inset 
                    :label="`Divide D loss by 2`"
                    style="margin-left: 15px"></v-switch>
                  <v-btn icon @click="showApplyDivideDcost = !showApplyDivideDcost" style="margin-top: 15px; margin-right: 55px;">
                    <v-icon x-small>info</v-icon>
                    <md-tooltip :md-active.sync="showApplyDivideDcost">Since D is updated twice (one for Real data and one for fake), it can be helpful to divide its cost by 2</md-tooltip>
                  </v-btn>
              </div>
            <div style="display: flex;">
              <div style="
                    grid-auto-flow: column;
                    display: flex;
                    width: 45%;">
                      <v-switch 
                        v-model="flip" 
                        inset 
                        :label="`Flip labels`"
                        style="margin-left: 20px"></v-switch>
                      <v-btn icon @click="showFlipLablesInfo = !showFlipLablesInfo" style="margin-top: 15px; margin-right: 40px;">
                        <v-icon x-small>info</v-icon>
                        <md-tooltip :md-active.sync="showFlipLablesInfo">this will flip labels in training process (Real=> 0 & Fake=> 1)</md-tooltip>
                      </v-btn>
              </div>
              <div style="
                    grid-auto-flow: column;
                    display: flex;
                    width: 55%;">
                  <v-switch 
                    v-model="smooth" 
                    inset 
                    :label="`Smooth labels`"
                    style="margin-left: 15px"></v-switch>
                  <v-btn icon @click="showSmoothLablesInfo = !showSmoothLablesInfo" style="margin-top: 15px; margin-right: 55px;">
                    <v-icon x-small>info</v-icon>
                    <md-tooltip :md-active.sync="showSmoothLablesInfo">this will assure smoothing labels (instead of giving value 1 for real, we give value between 0.9 and 1)</md-tooltip>
                  </v-btn>
              </div>
            </div>
            <div style="
                    grid-auto-flow: column;
                    display: flex;">
                  <v-switch 
                    v-model="applyOccasionalFlip" 
                    inset 
                    :label="`Flip labels occasionally`"
                    style="margin-left: 15px"></v-switch>
                  <v-btn icon @click="showApplyOccasionalFlipInfo = !showApplyOccasionalFlipInfo" style="margin-top: 15px; margin-right: 55px;">
                    <v-icon x-small>info</v-icon>
                    <md-tooltip :md-active.sync="showApplyOccasionalFlipInfo">toggling this will make occasional label flipping which can help to stabilise training</md-tooltip>
                  </v-btn>
              </div>
              <v-container style="
                  grid-auto-flow: column;
                  display: flex;"
                  v-show="applyOccasionalFlip">
                  <v-text-field
                    label="Probability of label flipping"
                    type="number"
                    required
                    v-model="occasionalFlip"
                    style="margin-right: 10px; width: 30px"
                  ></v-text-field>
                  <v-btn icon @click="showOccasionalFlipInfo = !showOccasionalFlipInfo" style="margin-top: 15px; margin-right: 40px;">
                      <v-icon x-small>info</v-icon>
                      <md-tooltip :md-active.sync="showOccasionalFlipInfo">The probability with which the label flipping will be applied (please do not exceed 30%)</md-tooltip>
                    </v-btn>    
              </v-container>
            <div style="
                    grid-auto-flow: column;
                    display: flex;"
                  v-show="wGenerator">
                  <v-switch
                    v-model="applyGradientPenalty" 
                    inset 
                    :label="`Apply Gradient Penalty(experimental)`"
                    style="margin-left: 15px"></v-switch>
                  <v-btn icon @click="showApplyGradientPenaltyInfo = !showApplyGradientPenaltyInfo" style="margin-top: 15px; margin-right: 55px;">
                    <v-icon x-small>info</v-icon>
                    <md-tooltip :md-active.sync="showApplyGradientPenaltyInfo">toggling this will add Gradient Penalty in training</md-tooltip>
                  </v-btn>
              </div>
              <v-container style="
                  grid-auto-flow: column;
                  display: flex;"
                  v-show="applyGradientPenalty">
                <v-select
                  :items="lambdaGradientPenaltyRange"
                  v-model="lambdaGradientPenalty"
                  label="Loss weight for gradient penalty ="
                  outlined
                  style="height: 45px; margin-bottom: 10px"
                  ></v-select>
                  <v-btn icon @click="showLossGpInfo = !showLossGpInfo" style="margin-top: 15px; margin-right: 40px;">
                      <v-icon x-small>info</v-icon>
                      <md-tooltip :md-active.sync="showLossGpInfo">Will be multiplied by the Gradient Penalty and taken into account in Discriminator loss</md-tooltip>
                    </v-btn>    
              </v-container>
              <div style="
                    grid-auto-flow: column;
                    display: flex;"
                  v-show="wGenerator">
                  <v-switch
                    v-model="applyDiscriminatorClipping" 
                    inset 
                    :label="`Apply Discriminator weights clipping(experimental)`"
                    style="margin-left: 15px"></v-switch>
                  <v-btn icon @click="showApplyDiscriminatorClippingInfo = !showApplyDiscriminatorClippingInfo" style="margin-top: 15px; margin-right: 55px;">
                    <v-icon x-small>info</v-icon>
                    <md-tooltip :md-active.sync="showApplyDiscriminatorClippingInfo">toggling this will assure Discriminator weights clipping (used in W-GAN)</md-tooltip>
                  </v-btn>
              </div>
              <v-container style="
                  grid-auto-flow: column;
                  display: flex;"
                  v-show="applyDiscriminatorClipping">
                <v-text-field
                        label="lower and upper clip value for D weights"
                        v-model="clipD"
                        style="height: 45px; margin-bottom: 10px"
                    ></v-text-field>
                  <v-btn icon @click="showClipD = !showClipD" style="margin-top: 15px; margin-right: 40px;">
                      <v-icon x-small>info</v-icon>
                      <md-tooltip :md-active.sync="showClipD">lower and upper clip value for Discriminator weights (used in W-GAN training)</md-tooltip>
                    </v-btn>    
              </v-container>
              <div style="
                    grid-auto-flow: column;
                    display: flex;"
                    v-show="showApplyFeatureMatching">
                  <v-switch
                    v-model="applyFeatureMatching" 
                    inset 
                    :label="`Apply Feature Matching`"
                    style="margin-left: 15px"></v-switch>
                  <v-btn icon @click="showApplyFeatureMatchingInfo = !showApplyFeatureMatchingInfo" style="margin-top: 15px; margin-right: 55px;">
                    <v-icon x-small>info</v-icon>
                    <md-tooltip :md-active.sync="showApplyFeatureMatchingInfo">toggling this will apply featue matching technique (from improved GAN training paper)</md-tooltip>
                  </v-btn>
              </div>
            <v-container style="
                  grid-auto-flow: column;
                  display: flex;">
                <v-select
                  :items="trainingRepetition"
                  v-model="trainingDiscriminatorMore"
                  label="Train Discriminator more times (recommended at the beginning of the training or with W-GAN)"
                  outlined
                  style="height: 45px; margin-bottom: 10px"
                  ></v-select>
                  <v-btn icon @click="showTrainingProbabilityDiscriminator = !showTrainingProbabilityDiscriminator" style="margin-top: 15px; margin-right: 40px;">
                      <v-icon x-small>info</v-icon>
                      <md-tooltip :md-active.sync="showTrainingProbabilityDiscriminator">this will train D more with the chosen probability (exp: if chosing 1/3 => when G is trained 3 times, D will have been trained 4 times)</md-tooltip>
                    </v-btn>    
              </v-container>
              <v-container style="
                  grid-auto-flow: column;
                  display: flex;">
                <v-select
                  :items="trainingRepetition"
                  v-model="trainingGeneratorMore"
                  label="Train Generator more times (recommended after some advanced steps of the training)"
                  outlined
                  style="height: 45px; margin-bottom: 10px"
                  ></v-select>
                  <v-btn icon @click="showTrainingProbabilityGenerator = !showTrainingProbabilityGenerator" style="margin-top: 15px; margin-right: 40px;">
                      <v-icon x-small>info</v-icon>
                      <md-tooltip :md-active.sync="showTrainingProbabilityGenerator">this will train G more with the chosen probability (exp: if chosing 1/3 => when D is trained 3 times, G will have been trained 4 times)</md-tooltip>
                    </v-btn>    
              </v-container>
              <v-container style="
                  grid-auto-flow: column;
                  display: flex;">
                <v-select
                  :items="unrollingSteps"
                  v-model="unrollingStep"
                  label="Unrolling the training with number of steps (experimental):"
                  outlined
                  style="height: 45px; margin-bottom: 10px"
                  ></v-select>
                  <v-btn icon @click="showUnrollingGANInfo = !showUnrollingGANInfo" style="margin-top: 15px; margin-right: 40px;">
                      <v-icon x-small>info</v-icon>
                      <md-tooltip :md-active.sync="showUnrollingGANInfo">this will assure defining the generator objective with respect to an unrolled optimization of the discriminator</md-tooltip>
                    </v-btn>    
              </v-container>
            <v-card-actions>
                <div class="text-center" style="display: flex;">
                <v-btn
                type="submit"
                @click="implementTricks = false"
                class="light-green accent-3">Apply Training Tricks</v-btn>
                <v-btn class="deep-orange accent-3" text @click="implementTricks = false">Close Dialog</v-btn>
                </div>
             </v-card-actions>
          </form>
        </v-card>
      </v-dialog>
      <v-layout row wrap>
        <div style="{
            grid-auto-flow: row;
            display: flex;
            height: 100px;
            margin: 0 auto;
        }">
        <div>
          <v-btn dark class="blue darken-2" @click="boolConfigDataSet = true">Configure DataSet</v-btn>
        </div>
        <div>
          <v-btn dark class="blue darken-2" @click="boolConfigGenerator = true">Configure Generator</v-btn>
        </div>
        <div>
          <v-btn dark class="blue darken-2" @click="boolConfigDiscriminator = true">Configure Discriminator</v-btn>
        </div> 
          <DataSetConfig :boolConfigDataSet="boolConfigDataSet" style="display: inline-block;"/>
          <GeneratorConfig :boolConfigGenerator="boolConfigGenerator" style="display: inline-block;"/>
          <DiscriminatorConfig :boolConfigDiscriminator="boolConfigDiscriminator" style="display: inline-block;"/>
        </div>
        <div align="center" justify="center" style="width: 100%;">
            <img
              src="../../../static/GANFinal.png"
              />
        </div>
        <div style="{
            grid-auto-flow: row;
            display: flex;
        }">  
          <NetworkLossConfig :network="GeneratorStr"/>
          <NetworkOptimizerConfig :network="GeneratorStr"/>
          <NetworkLossConfig :network="DiscriminatorStr"/>
          <NetworkOptimizerConfig :network="DiscriminatorStr"/>
          <LatentVectorConfig style="display: inline-block;"/>
        </div>
        <div style="width: 100%;">
          <form @submit.prevent="onGetGeneratedData">
            <v-flex class="d-inline-flex pa-2">
              <v-card-title
                class="headline grey lighten-2"
                primary-title
                >Generated Data
                </v-card-title>
                <v-text-field
                  label="Visualization Size: "
                  type="number"
                  required
                  v-model="visualizationSize"
                  style="height: 45px; margin-bottom: 10px"
                 ></v-text-field>
                <v-text-field
                  label="Number Of batches: "
                  type="number"
                  required
                  v-model="nbBatchesViz"
                  style="height: 45px; margin-bottom: 10px"
                  ></v-text-field>
                <v-btn @click="implementTricks = true">Custom Training Hacks</v-btn>
                <v-btn type="submit">
                  Get generated data: </v-btn>
                  <v-btn @click="SaveModels = true">Save Trained Models</v-btn>
                  <v-btn @click="onStopTraining">Stop Training</v-btn>
                  <v-btn @click="onResetTraining">Reset Training</v-btn>
                  <v-card v-show="waitingGeneratedData">
                    <v-text-field
                        v-model="waitGeneratedDataText"
                        style="height: 45px; margin-bottom: 10px; margin-left: 20px; width: 80%"
                        disabled
                      ></v-text-field>
                    <v-progress-circular
                      color="primary"
                      indeterminate
                      style="margin-left: 50%;"
                    ></v-progress-circular>
                  </v-card>
                  <v-card v-show="elapsedTime" style="display: flex;">
                      <v-card-title style="width: 25%;" class="headline" primary-title>
                        epch number: {{this.epochNumber}}</v-card-title>
                      <v-card-title style="width: 25%;" class="headline" primary-title>
                        batch index: {{this.batchIndex}}</v-card-title>
                      <v-card-title style="width: 50%;" class="headline" primary-title>
                        elapsed time: {{this.elapsedTime}} s</v-card-title>
                  </v-card>
              </v-flex>
            </form>
                <v-layout column wrap>
                  <div style="
                    grid-auto-flow: row;
                    display: flex;
                    margin: 0 auto;">
                    <img id="generatedBytes" src="" style="margin-right: 10px;">
                    <img id="worstGeneratedBytes" src="" style="margin-right: 10px;">
                    <img id="realBytes" src="">
                  </div>
                  <div style="{
                        grid-auto-flow: row;
                        display: flex;
                        margin: 0 auto;
                    }"
                    v-show="showResults">
                    <div id="lossData"></div>
                    <div id="KL_JS_div"></div>
                  </div>
                  <div style="{
                        grid-auto-flow: row;
                        display: flex;
                        margin: 0 auto;
                    }"
                    v-show="showResults">
                    <div id="viz_reduced_2d"></div>
                    <div id="viz_reduced_3d"></div>
                  </div>
                  <div id="D_metrics"
                  style="{
                        grid-auto-flow: row;
                        display: flex;
                        margin: 0 auto;
                    }"
                    v-show="showResults"></div>
                  <div>
                    <v-data-table
                      :headers="headers"
                      :items="items"
                      :items-per-page="6"
                      class="elevation-1"
                    >
                    <template slot="items" slot-scope="props">
                      <td>{{ props.item.name }}</td>
                      <td class="text-xs-right">{{ props.item.realData }}</td>
                      <td class="text-xs-right">{{ props.item.generatedData }}</td>
                    </template>
                    </v-data-table>
                  </div>
                    <div id="statistics_real" style="width: 95%"></div>
                    <div id="statistics_generated" style="width: 95%"></div>
                </v-layout>
        </div>
      </v-layout>
      <v-snackbar 
          v-model="errorConfig"
          :timeout="2500"
          class="red darken-2"
          >
          Something went wrong, Please Try Again
      </v-snackbar>
      <v-snackbar 
            v-model="implementedTricks"
            :multi-line="true"
            :timeout="2500"
            class="green darken-2"
              >
               {{ this.implementedTricksMessage }}
          <span> ... </span>
          <v-btn icon @click="implementedTricks = false"> <v-icon x-small right>mdi-close</v-icon> </v-btn>  
        </v-snackbar>
        <v-snackbar
        v-model="successSavedTrainedModel"
        :timeout="2500"
        class="green darken-2"
        >trained {{ this.savedTrainedModel }} saved successfully
        <v-btn icon @click="showSavedTrainedModelInfo = true"><v-icon x-small>info</v-icon></v-btn>
        <v-btn icon @click="successSavedTrainedModel = false"> <v-icon x-small right>mdi-close</v-icon> </v-btn>
        </v-snackbar>  
  </v-app>
</template>

<script>
  import GeneratorConfig from './ConfigGenerator'
  import DataSetConfig from './ConfigDataSet'
  import DiscriminatorConfig from './ConfigDiscriminator'
  import NetworkLossConfig from './ConfigNetworkLoss'
  import NetworkOptimizerConfig from './ConfigNetworkOptimizer'
  import LatentVectorConfig from './ConfigLatentVector'
  import Plotly from 'plotly.js-dist'
  import visSettings from './visualization.settings'
  import { EventBus } from './event-bus.js'

  export default {
    components: {
      DataSetConfig,
      DiscriminatorConfig,
      GeneratorConfig,
      NetworkLossConfig,
      NetworkOptimizerConfig,
      LatentVectorConfig
    },
    data: () => ({
      waitGeneratedDataText: 'Please wait until data is generated',
      waitingGeneratedData: false,
      errorConfig: '',
      networkModels: [
        'Vanilla GAN architecture',
        'Deep Convolutional GAN architecture',
        'Wasserstein GAN architecture'
      ],
      dropOutRange: [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.33],
      inChannelsGeneratorRange: [50, 75, 80, 100, 120],
      trainingRepetition: [0, 1, 2, 3, 4, 5],
      nbLayersRange: [2, 3, 4, 5],
      OptimizerNames: ['Adam', 'RMS', 'SGD'],
      LearningRates: [0.0002, 0.0005, 0.0001, 0.0003, 0.0004, 0.00009, 0.001, 0.002],
      BetaOneValues: [0.5, 0.45, 0.55, 0.6, 0.4],
      BetaTwoValues: [0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.993],
      LVtypes: ['gaussian', 'multimodal gaussian', 'uniform'],
      generatedData: '',
      nbBatchesViz: '',
      visualizationSize: '',
      implementTricks: false,
      flip: false,
      smooth: false,
      training: true,
      implementedTricks: false,
      showFlipLablesInfo: false,
      showSmoothLablesInfo: false,
      showTrainingProbabilityDiscriminator: false,
      trainingDiscriminatorMore: 0,
      trainingGeneratorMore: 0,
      unrollingSteps: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      unrollingStep: 0,
      showUnrollingGANInfo: false,
      showResults: false,
      SaveModels: false,
      saveGenerator: true,
      trainedGPath: '',
      showTrainedGPath: false,
      trainedDPath: '',
      showTrainedDPath: false,
      successSavedTrainedModel: false,
      savedTrainedModel: '',
      showSavedTrainedModelInfo: false,
      savedTrainedModelMessageInfo: '',
      savedTrainedModelSummaryInfo: '',
      continueTraining: true,
      boolConfigDataSet: false,
      boolConfigGenerator: false,
      boolConfigDiscriminator: false,
      DiscriminatorStr: 'Discriminator',
      GeneratorStr: 'Generator',
      showApplyGradientPenaltyInfo: false,
      applyGradientPenalty: false,
      lambdaGradientPenaltyRange: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      lambdaGradientPenalty: 10,
      showLossGpInfo: false,
      epochNumber: 0,
      batchIndex: 0,
      elapsedTime: 0,
      showTrackInfo: false,
      showApplyOccasionalFlipInfo: false,
      applyOccasionalFlip: false,
      showOccasionalFlipInfo: false,
      applyFeatureMatching: false,
      showApplyFeatureMatchingInfo: false,
      occasionalFlip: 0,
      wGenerator: false,
      clipD: 0.01,
      showClipD: false,
      applyDiscriminatorClipping: false,
      showApplyFeatureMatching: false,
      showApplyDiscriminatorClipping: false,
      applyDivideDcost: false,
      showApplyDivideDcost: false,
      trackInfo: '',
      apiDockerContainerIp: 'http://127.0.0.1',
      headers: [
        {
          text: 'Statistics of Batches',
          align: 'left',
          sortable: false,
          value: 'name'
        },
          { text: 'Real Data', value: 'realData' },
          { text: 'Generated Data', value: 'generatedData' }],
      items: [],
      AllStatistics: false
    }),
    created () {
      EventBus.$on('toogle-bool-config-dataset', dataSetDialog => {
        this.boolConfigDataSet = dataSetDialog
      })
      // 'toogle-bool-config-generator'
      EventBus.$on('toogle-bool-config-generator', generatorDialog => {
        this.boolConfigGenerator = generatorDialog
      })
      EventBus.$on('toogle-bool-config-discriminator', discriminatorDialog => {
        this.boolConfigDiscriminator = discriminatorDialog
      })
      EventBus.$on('created-generator-config', generatorElement => {
        this.wGenerator = generatorElement.generatorModel === 'Wasserstein GAN architecture'
      })
      EventBus.$on('mse-g', mseG => {
        this.showApplyFeatureMatching = mseG
      })
    },
    computed: {
      error: '#ff00ff'
    },
    methods: {
      onTrainingIteration (trainingBool) {
        trainingBool = false
        this.waitingGeneratedData = true
        const trainElement = {
          visualization_size: parseInt(this.visualizationSize),
          nb_batches: parseInt(this.nbBatchesViz),
          train_more_Discriminator: this.trainingDiscriminatorMore,
          train_more_Generator: this.trainingGeneratorMore,
          unrolling_step: this.unrollingStep
        }
        this.$http.post(this.apiDockerContainerIp + ':5000/train', trainElement)
          .then(data => {
            console.log('generated data: ', data)
            trainingBool = data.body.training
            this.showResults = true
            this.generatedData = data.body.generated_images
            this.epochNumber = data.body.epoch_number
            this.batchIndex = data.body.index_batch
            this.elapsedTime = data.body.elapsed_time
            this.trackInfo = data.body.track_convergence_DS
            console.log('track convergence: ', this.trackInfo)
            // this.showTrackInfo = true
            document.getElementById('generatedBytes').src = 'data:image/png;base64,' + data.body.generated_bytes
            document.getElementById('generatedBytes').title = 'Generated Data'
            document.getElementById('worstGeneratedBytes').src = 'data:image/png;base64,' + data.body.worst_generated_bytes
            document.getElementById('realBytes').src = 'data:image/png;base64,' + data.body.real_bytes

            var batchesX = Array.from({length: 40}, (v, k) => k + 1)
            // Losses Visualization :
            visSettings.losses.dLossViz.x = batchesX
            visSettings.losses.dLossViz.y = data.body.d_error
            visSettings.losses.gLossViz.x = batchesX
            visSettings.losses.gLossViz.y = data.body.g_error
            var lossData = [visSettings.losses.dLossViz, visSettings.losses.gLossViz]
            var layoutlossData = visSettings.losses.layoutlossData
            Plotly.newPlot('lossData', lossData, layoutlossData)
            // KL & JS visualization:
            visSettings.kl_js.klDiv.x = batchesX
            visSettings.kl_js.klDiv.y = data.body.kl_div
            visSettings.kl_js.jsDiv.x = batchesX
            visSettings.kl_js.jsDiv.y = data.body.js_div
            var KLJSData = [visSettings.kl_js.klDiv, visSettings.kl_js.jsDiv]
            Plotly.newPlot('KL_JS_div', KLJSData, visSettings.kl_js.layoutKLJSdiv)
            // D metrics (precision, recall, f1_score):
            visSettings.d_metrics.precision.x = batchesX
            visSettings.d_metrics.precision.y = data.body.precision
            visSettings.d_metrics.recall.x = batchesX
            visSettings.d_metrics.recall.y = data.body.recall
            visSettings.d_metrics.f1_score.x = batchesX
            visSettings.d_metrics.f1_score.y = data.body.f1_score
            var DMetricsPresentation = [visSettings.d_metrics.precision, visSettings.d_metrics.recall, visSettings.d_metrics.f1_score]
            Plotly.newPlot('D_metrics', DMetricsPresentation, visSettings.d_metrics.layoutDmetrics)
            // Presentation of reduced Data:
            visSettings.reduced2D.real2D.x = data.body.real_2d[0]
            visSettings.reduced2D.real2D.y = data.body.real_2d[1]
            visSettings.reduced2D.generated2D.x = data.body.fake_2d[0]
            visSettings.reduced2D.generated2D.y = data.body.fake_2d[1]
            var reduced2D = [visSettings.reduced2D.real2D, visSettings.reduced2D.generated2D]
            Plotly.newPlot('viz_reduced_2d', reduced2D, visSettings.reduced2D.layoutReduced2D)
            visSettings.reduced3D.real3D.x = data.body.real_3d[0]
            visSettings.reduced3D.real3D.y = data.body.real_3d[1]
            visSettings.reduced3D.real3D.z = data.body.real_3d[2]
            visSettings.reduced3D.generated3D.x = data.body.fake_3d[0]
            visSettings.reduced3D.generated3D.y = data.body.fake_3d[1]
            visSettings.reduced3D.generated3D.z = data.body.fake_3d[2]
            var reduced3D = [visSettings.reduced3D.real3D, visSettings.reduced3D.generated3D]
            Plotly.newPlot('viz_reduced_3d', reduced3D, visSettings.reduced3D.layoutReduced3D)
        // var barStatisticsGenerated = []
            var barStatisticsX = [] // batches 1->10*nbBacthes
            this.items = []
            for (let index = 0; index < this.nbBatchesViz; index++) {
              this.items.push({
                name: 'min of batch ' + String(index + 1),
                realData: data.body.d_Loss_real_min[index],
                generatedData: data.body.d_Loss_fake_min[index]
              })
              this.items.push({
                name: 'mean of batch ' + String(index + 1),
                realData: data.body.d_Loss_real_mean[index],
                generatedData: data.body.d_Loss_fake_mean[index]
              })
              this.items.push({
                name: 'max of batch ' + String(index + 1),
                realData: data.body.d_Loss_real_max[index],
                generatedData: data.body.d_Loss_fake_max[index]
              })
              barStatisticsX.push('batch ' + String(index + 1))
            }
            var barStatisticsRealMin = {
              x: barStatisticsX,
              y: data.body.d_Loss_real_min,
              name: '<b>Min</b>',
              type: 'bar'
            }
            var barStatisticsRealMean = {
              x: barStatisticsX,
              y: data.body.d_Loss_real_mean,
              name: '<b>Mean</b>',
              type: 'bar'
            }
            var barStatisticsRealMax = {
              x: barStatisticsX,
              y: data.body.d_Loss_real_max,
              name: '<b>Max</b>',
              type: 'bar'
            }
            var barStatisticsReal = [barStatisticsRealMin, barStatisticsRealMean, barStatisticsRealMax]
            Plotly.newPlot('statistics_real', barStatisticsReal, {barmode: 'stack', title: '<b>Statistics for real data</b>'})
            var barStatisticsGeneratedMin = {
              x: barStatisticsX,
              y: data.body.d_Loss_fake_min,
              name: '<b>Min</b>',
              type: 'bar'
            }
            var barStatisticsGeneratedMean = {
              x: barStatisticsX,
              y: data.body.d_Loss_fake_mean,
              name: '<b>Mean</b>',
              type: 'bar'
            }
            var barStatisticsGeneratedMax = {
              x: barStatisticsX,
              y: data.body.d_Loss_fake_max,
              name: '<b>Max</b>',
              type: 'bar'
            }
            var barStatisticsGenerated = [barStatisticsGeneratedMin, barStatisticsGeneratedMean, barStatisticsGeneratedMax]
            Plotly.newPlot('statistics_generated', barStatisticsGenerated, {barmode: 'stack', title: '<b>Statistics for generated data</b>'})
            this.AllStatistics = true
            this.waitingGeneratedData = false
          })
          .then(() => {
            if (trainingBool) {
              this.onTrainingIteration()
            }
          })
          .catch(e => {
            this.errorConfig = e.statusText
            this.waitingGeneratedData = false
            trainingBool = false
            console.log('error: ', e)
          })
      },
      async onAsyncIteration (trainingBool) {
        const trainingQuery = await this.onTrainingIteration(trainingBool)
        Promise.resolve(trainingQuery)
        return trainingQuery
      },
      onGetGeneratedData () {
        this.continueTraining = true
        if (this.continueTraining) {
          this.onAsyncIteration(this.training)
        }
        this.continueTraining = !this.continueTraining
      },
      onTrainingTricks () {
        const trainTricks = {
          flip: this.flip,
          smooth: this.smooth,
          apply_gp: this.applyGradientPenalty,
          lambda_gp: this.lambdaGradientPenalty,
          feature_matching: this.applyFeatureMatching,
          apply_occasional_flip: this.applyOccasionalFlip,
          occasional_flip: (this.occasionalFlip !== 0) ? parseInt(100 / this.occasionalFlip) : 0,
          clip_d: parseFloat(this.clipD),
          apply_clip_d: this.applyDiscriminatorClipping,
          apply_divide_d_cost: this.applyDivideDcost
        }
        this.$http.post(this.apiDockerContainerIp + ':5000/traintricks', trainTricks)
        .then(data => {
          this.implementedTricks = true
          this.implementedTricksMessage = data.body.message
          console.log('tricks: ', data)
        })
        .catch(e => {
          this.errorConfig = e.statusText
          console.log('error: ', e)
        })
      },
      onStopTraining () {
        this.training = false
        this.$http.post(this.apiDockerContainerIp + ':5000/stoptraining')
        .then(data => {
          console.log('stopped training')
        })
        .catch(e => {
          this.errorConfig = e.statusText
          console.log('error: ', e)
        })
      },
      onResetTraining () {
        this.$http.post(this.apiDockerContainerIp + ':5000/reset', {reset: 'reset'})
        .then(data => {
          console.log('reset training')
        })
        .catch(e => {
          this.errorConfig = e.statusText
          console.log('error: ', e)
        })
      },
      onSaveGenerator () {
        this.$http.post(this.apiDockerContainerIp + ':5000/savegenerator', {path: this.trainedGPath})
        .then(data => {
          console.log('model saved: ', data)
          this.successSavedTrainedModel = true
          this.savedTrainedModel = 'Generator'
          this.savedTrainedModelMessageInfo = data.body.message
          this.savedTrainedModelSummaryInfo = data.body.model
        })
        .catch(e => {
          this.errorConfig = e.statusText
          console.log('error: ', e)
        })
      },
      onSaveDiscriminator () {
        this.$http.post(this.$parent.apiDockerContainerIp + ':5000/savediscriminator', {path: this.trainedDPath})
        .then(data => {
          console.log('model saved: ', data)
          this.successSavedTrainedModel = true
          this.savedTrainedModel = 'Discriminator'
          this.savedTrainedModelMessageInfo = data.body.message
          this.savedTrainedModelSummaryInfo = data.body.model
        })
        .catch(e => {
          this.errorConfig = e.statusText
          console.log('error: ', e)
        })
      }
    }
  }
</script>
