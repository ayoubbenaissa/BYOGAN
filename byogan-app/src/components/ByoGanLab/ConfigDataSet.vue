<template>
    <v-app>
      <v-dialog v-model="configuringDataSet" persistent width="500px" scrollable="false">
        <v-card>
          <v-container style="
                  grid-auto-flow: column;
                  display: flex;">
              <v-text-field
                  v-model="waitConfigDataSet"
                  style="height: 45px; margin-bottom: 10px;"
                  disabled
                ></v-text-field>
              <v-progress-circular
                color="primary"
                indeterminate
                style="margin-top: 15px; margin-left: 10px;"
              ></v-progress-circular>
          </v-container>
        </v-card>
        </v-dialog>

          <v-dialog v-model="boolConfigDataSet" persistent width="800px">
            <v-card>
              <form @submit.prevent="onCreateDataSet">
                <v-flex class="d-inline-flex pa-2">
                  <v-card-title
                    class="headline grey lighten-2"
                    primary-title
                  >
                    DataSet
                  </v-card-title>
                  <div style="
                  grid-auto-flow: column;
                  display: flex;">
                      <v-select
                        :items="datasetsList"
                        v-model="datasetName"
                        label="Dataset:"
                        outlined
                        style="height: 45px; margin-bottom: 10px; width: 100%"
                        ></v-select>
                      <v-btn icon @click="showDataSetTypeInfo = !showDataSetTypeInfo" style="margin-top: 20px; margin-right: 40px;">
                        <v-icon x-small>info</v-icon>
                        <md-tooltip :md-active.sync="showDataSetTypeInfo">chose a defined dataset (i.e: MNIST, FashionMNIST) or a local dataset ImageFolder/CSV</md-tooltip>
                      </v-btn>
                  </div>
                  <div style="
                  grid-auto-flow: column;
                  display: flex;">
                    <v-text-field
                      label="Folder Path"
                      name="path"
                      id="folderPath"
                      required
                      v-model="datasetPath"
                      style="height: 45px; margin-bottom: 10px"
                    ></v-text-field>
                    <v-btn icon @click="showDataSetPathInfo = !showDataSetPathInfo" style="margin-top: 20px; margin-right: 40px;">
                        <v-icon x-small>info</v-icon>
                        <md-tooltip :md-active.sync="showDataSetPathInfo">path where dataset should be retrieved/stored. For more info please visit: <url>https://pytorch.org/docs/stable/torchvision/datasets.html</url></md-tooltip>
                      </v-btn>
                  </div>
                  <div style="display: flex;">
                      <v-text-field
                        label="Batch Size"
                        name="batch"
                        id="batch"
                        type="number"
                        required
                        v-model="batchSize"
                        style="margin-right: 10px; width: 30px"
                      ></v-text-field>
                      <v-text-field
                        label="Image Size"
                        name="image"
                        id="imageSize"
                        type="number"
                        required
                        v-model="imageSize"
                        style="margin-right: 10px; width: 30px"
                      ></v-text-field>
                      <v-text-field
                        label="Number Of Channels"
                        name="channels"
                        id="nbChannels"
                        type="number"
                        required
                        hint= 'if you are working with grey images, choose 1 if with RGB choose 3'
                        v-model="nbChannels"
                        style="height: 45px; margin-bottom: 10px"
                      ></v-text-field>
                  </div>
                  <v-card-actions>
                    <div class="text-center" style="display: flex;">
                    <v-btn 
                    :disabled="!validDatasetForm"
                    type="submit"
                    @click="onToggleBoolConfigDataSet"
                    class="light-green accent-3">Create DataSet</v-btn>
                    <v-btn class="deep-orange accent-3" text @click="onToggleBoolConfigDataSet">Close Dialog</v-btn>
                    </div>
                  </v-card-actions>
                </v-flex>
              <small><b>*indicates required field</b></small>
              </form>
            </v-card>
          </v-dialog>
          <div>
                <v-snackbar 
                    v-model="createdDataSet"
                    :multi-line="true"
                    :timeout="2500"
                    class="green darken-2"
                    >
                    {{ this.createdDataSetMessage }}
                <span> ... </span>
                <v-btn icon @click="showDataSetInfo = true"><v-icon x-small>info</v-icon></v-btn>
                <v-btn icon @click="createdDataSet = false"> <v-icon x-small right>mdi-close</v-icon> </v-btn>  
                </v-snackbar>
            </div>
        <v-dialog v-model="showDataSetInfo" persistent width="700px">
          <v-card>
            <v-card-title
              class="headline grey lighten-2"
              primary-title
            >Created DataSet Info
            </v-card-title>
            <v-text-field
              label="Created Dataset Path Info"
              outlined
              disabled
              prepend-icon="place"
              v-model="createdDataSetPathInfo"
            ></v-text-field>
            <v-text-field
              label="created DataSet Length Info"
              outlined
              disabled
              prepend-icon="info"
              v-model="createdDataSetLengthInfo"
            ></v-text-field>
            <v-btn @click="showDataSetInfo = false" class="deep-orange accent-3">Close</v-btn>
          </v-card>
        </v-dialog>
    </v-app>
</template>
<script>
import { EventBus } from './event-bus.js'
export default {
  name: 'DataSetConfig',
  props: {
    boolConfigDataSet: Boolean
  },
  data: () => ({
    createdDataSet: false,
    datasetName: '',
    datasetsList: ['MNIST', 'FashionMNIST', 'ImageFolder', 'CSV'],
    datasetPath: '',
    batchSize: 128,
    imageSize: '',
    nbChannels: '',
    createdDataSetMessage: '',
    visualizationSize: '',
    createdDataSetPathInfo: '',
    createdDataSetLengthInfo: '',
    showDataSetInfo: false,
    configuringDataSet: false,
    waitConfigDataSet: 'Please wait until configuration of Data loader is done',
    showDataSetTypeInfo: false,
    showDataSetPathInfo: false
  }),
  computed: {
    validDatasetForm () {
      // the form is valid when necessary fields are filled:
      return this.datasetPath !== '' &&
        this.batchSize !== '' &&
        this.imageSize !== '' &&
        this.nbChannels !== ''
    }
  },
  methods: {
    onToggleBoolConfigDataSet () {
      this.boolConfigDataSet = !this.boolConfigDataSet
      EventBus.$emit('toogle-bool-config-dataset', this.boolConfigDataSet)
    },
    onCreateDataSet () {
      this.configuringDataSet = true
      const dataSetElement = {
        name: this.datasetName,
        path: this.datasetPath,
        batch_size: this.batchSize,
        img_size: this.imageSize,
        channels: this.nbChannels
      }
      this.$http.post(this.$parent.apiDockerContainerIp + ':5000/dataset', dataSetElement)
        .then(data => {
          this.waitConfigDataSet = 'Please wait until configuration of Data loader is done'
          this.createdDataSet = true
          this.createdDataSetMessage = data.body.message
          this.createdDataSetPathInfo = data.body.path
          this.createdDataSetLengthInfo = data.body.length
          this.configuringDataSet = false
          console.log('dataset path: ', this.createdDataSetPath)
        })
        .catch(e => {
          this.$parent.errorConfig = e.statusText
          console.log(e)
        })
    }
  }
}
</script>
