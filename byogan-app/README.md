# byogan

byo-gan product for configuring, training and interacting with GANs.
install npm dependencies

# byogan-app
FrontEnd part of BYOGan: developed mainly using [VueJS](https://vuejs.org/) and [JavaScript](https://www.javascript.com/).


``` bash
# necessary JavaScript(FrontEnd) modules can be found in package.json

## to install VueCLI:
npm install -g @vue/cli

## install dependencies
npm install

## start:
npm start

## serve with hot reload at localhost:8080
npm run dev

## build for production with minification
npm run build

## build for production and view the bundle analyzer report
npm run build --report
```

# Tool UIs:

The tool offers extended configuration of most of the GAN framework components.
This can help beginners get started with this complex framework, and it can also help practitioners apply custom configuration and track the learning process.

### + Dataset:
This is the first component to configure as it will link to the Real Data of interest which we want our model to learn. <br />
The tool offers between selecting a defined Dataset (MNIST, FashionMNIST) or a local one (image folder, CSV file). <br />
The configuration tool looks as follows:
![DataSet UI](./ToolUIs/DatasetUI.png)

