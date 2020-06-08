<template>
    <v-app>
      <v-dialog v-model="configuringLatentVector" persistent width="500px" scrollable="false">
          <v-card>
            <v-text-field
                v-model="waitConfigLatentVector"
                style="height: 45px; margin-bottom: 10px"
              ></v-text-field>
            <v-progress-circular
              color="primary"
              indeterminate
              style="margin-left: 50%;"
            ></v-progress-circular>
          </v-card>
        </v-dialog>
        <form @submit.prevent="onLatentVector">
                <v-flex class="d-inline-flex pa-2">
                  <v-card-title
                    class="headline grey lighten-2"
                    primary-title
                    >Latent Vector Configuration
                    </v-card-title>
                    <v-select
                      :items="this.$parent.LVtypes"
                      v-model="LVtype"
                      label="Data Distribution in Latent Vector: "
                      outlined
                      required
                      style="height: 45px; margin-bottom: 10px"
                    ></v-select>
                    <v-btn id="submitBtnLV" dark class="green darken-1" type="submit">
                      {{ this.latentVectorState }}</v-btn>
                </v-flex>
            </form>
            <div>
                <v-snackbar 
                    v-model="createdLatentVector"
                    :multi-line="true"
                    :timeout="2500"
                    class="green darken-2"
                    >
                    {{ this.createdLatentVectorMessageInfo }}
                <span> ... </span>
                <v-btn icon @click="showLatentVectorInfo = true"><v-icon x-small>info</v-icon></v-btn>
                <v-btn icon @click="createdLatentVector = false"> <v-icon x-small right>mdi-close</v-icon> </v-btn>  
                </v-snackbar>
            </div>
            <v-dialog v-model="showLatentVectorInfo" persistent width="700px">
                <v-card>
                        <v-card-title
                        class="headline grey lighten-2"
                        primary-title
                        >Latent Vector Info
                        </v-card-title>
                        <v-text-field
                        v-model="createdLatentVectorMessageInfo"
                        label="Overview"
                        outlined
                        disabled
                        prepend-icon="info"
                        ></v-text-field>
                        <v-text-field
                        label="created Latent Vector Shape"
                        outlined
                        disabled
                        prepend-icon="info"
                        v-model="createdLatentVectorShapeInfo"
                        ></v-text-field>
                        <v-text-field
                        label="created Latent Vector Distribution"
                        outlined
                        disabled
                        prepend-icon="info"
                        v-model="createdLatentVectorDistributionInfo"
                        ></v-text-field>
                        <v-text-field 
                        outlined
                        disabled
                        prepend-icon="info"
                        v-model="createdLatentVectorGpuInfo"
                        label="Device"
                        suffix="(Same device as Generator)">
                        </v-text-field> 
                        <v-btn @click="showLatentVectorInfo = false" class="deep-orange accent-3">Close</v-btn>
                    </v-card>
            </v-dialog>
    </v-app>
</template>
<script>
export default {
  name: 'LatentVectorConfig',
  data: () => ({
    LVtype: '',
    latentVectorState: 'initialise Latent Vector',
    createdLatentVector: false,
    createdLatentVectorMessageInfo: '',
    showLatentVectorInfo: false,
    createdLatentVectorShapeInfo: '',
    createdLatentVectorDistributionInfo: '',
    createdLatentVectorGpuInfo: false,
    latentVectorStateREQUEST: 'created',
    configuringLatentVector: false,
    waitConfigLatentVector: 'Please wait until configuration of Latent Vector is done'
  }),
  methods: {
    onLatentVector () {
      this.configuringLatentVector = true
      const latentVectorElement = {
        noise_type: this.LVtype,
        state: this.latentVectorStateREQUEST
      }
      this.$http.post(this.$parent.apiDockerContainerIp + ':5000/latentvector', latentVectorElement)
        .then(data => {
          this.latentVectorState = 'Update Latent Vector'
          // when the Latent Vector is created, the button color is changed into yellow
          // (warn and inform user)
          document.getElementById('submitBtnLV').className = 'yellow darken-2 btn btn--raised theme--dark'
          this.latentVectorStateREQUEST = 'updated'
          this.createdLatentVector = true
          this.createdLatentVectorMessageInfo = data.body.message
          this.createdLatentVectorShapeInfo = data.body.shape
          this.createdLatentVectorDistributionInfo = data.body.type
          this.createdLatentVectorGpuInfo = data.body.device
          this.configuringLatentVector = false
          console.log('latent vector: ', data)
        })
        .catch(e => {
          this.$parent.errorConfig = e.statusText
          console.log('error: ', e)
        })
    }
  }
}
</script>
