<template>
  <div>
    <h1>Get Started</h1>

    <h2> Generative Adversarial Networks</h2>

    <p>Generative Adversarial Networks (<a href="https://arxiv.org/abs/1406.2661">GANs</a> for short) is a framework proposed by <a href="https://github.com/goodfeli">Dr.Ian Goodfellow</a> on 2014.
      The purpose is to present a framework that can learn Data Distribution from real Data with the goal to finally produce realistic artificial Data.
    </p>
    <p>The framework is composed of two networks that are adversary they compete with each other in a zero-sum game framework.
       The models are the generative model, which generates candidates, and in the other side there is the discriminator, which evaluates candidate.And the contest operates in terms of data distribution.</p>

    <p>
      The workflow of the model is as follows: given several samples from real-world data, the Discriminator tries to learn the data distribution and characteristic patterns of "real data".
    </p>

    <p>The Generator takes as input a vector of random values (often referred as "noise input" or "Latent Vector") and outputs data in same size and form as real data.</p>

    <p>The Discriminator (having a good idea of how real-data looks like) takes the output of the Generator and evaluates it as being real or fake with a probability (1 meaning data is real, 0 data is fake)
       This process is repeated until the Discriminator can NOT differentiate real data from fake-generated data</p>
    
    <h2> Our Tool</h2>

    <p>Understanding GANs is not straightforwad, and their training process is complex as it relies on an adversarial <a href="https://brilliant.org/wiki/minimax/">mini-max game</a>.
    Our tool offers the possbilities to configure GANs with ease: configure the Data-Loader, configure competing networks (Generator & Discriminator) and configure all necessary components for the training.
    the interface of our tool looks as follows:
    </p>

    <div align="center" justify="center" style="width: 100%;">
            <img
              src="../../../static/Tool_Interface.png"
              />
        </div>
  </div>
</template>
