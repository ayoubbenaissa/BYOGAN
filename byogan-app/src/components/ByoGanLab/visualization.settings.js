export default {
  losses: {
    dLossViz: {
      mode: 'lines+markers',
      type: 'scatter',
      name: '<b>Discriminator Loss</b>'
    },
    gLossViz: {
      mode: 'lines+markers',
      type: 'scatter',
      name: '<b>Generator Loss</b>'
    },
    layoutlossData: {
      title: '<b>Generator & Discriminator Losses</b>',
      autosize: false,
      width: 550,
      height: 500 }
  },
  kl_js: {
    klDiv: {
      mode: 'lines+markers',
      type: 'scatter',
      name: '<b>KL divergence</b>'
    },
    jsDiv: {
      mode: 'lines+markers',
      type: 'scatter',
      name: '<b>JS divergence</b>'
    },
    layoutKLJSdiv: {
      title: '<b>KL & JS divergences</b>',
      autosize: false,
      width: 550,
      height: 500 }
  },
  d_metrics: {
    precision: {
      mode: 'lines+markers',
      type: 'scatter',
      name: '<b>Precision</b>'
    },
    recall: {
      mode: 'lines+markers',
      type: 'scatter',
      name: '<b>Recall</b>',
      xaxis: 'x2',
      yaxis: 'y2'
    },
    f1_score: {
      mode: 'lines+markers',
      type: 'scatter',
      name: '<b>F1_score</b>',
      xaxis: 'x3',
      yaxis: 'y3'
    },
    layoutDmetrics: {
      title: '<b>Discriminator metrics: Precision, Recall and F1_score</b>',
      grid: {
        rows: 1,
        columns: 3,
        pattern: 'independent',
        autosize: false,
        width: 1300,
        height: 500
      } }
  },
  reduced2D: {
    real2D: {
      mode: 'markers',
      type: 'scatter',
      name: '<b>real data</b>',
      font: {
        family: 'sans-serif',
        size: 12,
        color: '#000'
      }
    },
    generated2D: {
      mode: 'markers',
      type: 'scatter',
      name: '<b>generated data</b>',
      font: {
        family: 'sans-serif',
        size: 14,
        color: '#000'
      }
    },
    layoutReduced2D: {
      title: '<b>Reduced data to 2D</b>',
      autosize: false,
      width: 550,
      height: 500
    }
  },
  reduced3D: {
    real3D: {
      mode: 'markers',
      type: 'scatter3d',
      name: '<b>real data</b>'
    },
    generated3D: {
      mode: 'markers',
      type: 'scatter3d',
      name: '<b>generated data</b>'
    },
    layoutReduced3D: {
      title: '<b>Reduced data to 3D</b>',
      autosize: false,
      width: 550,
      height: 500
    }
  }
}
