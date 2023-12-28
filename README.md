# Correcting-Image-Exposure-using-DNN
A supervised convolutional autoencoder network inspired by Mao et al's paper (https://arxiv.org/abs/1606.08921) 'Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections' explores a deep convolutional autoencoder for efficiently learning data codings. This network compresses input into a lower-dimensional representation and decodes it back to closely match the original input, reconstructing data from the learned representation. In practice, this translates to restoring lost details and eliminating corruptions from original images, transforming them into clean versions. The paper also discusses the use of skip connections for aiding gradient propagation and addressing the vanishing gradient issue in deep networks, crucial for preserving image details.

For the projects aim, image enhancement tasks like correcting light in low-light conditions or improper exposure, a simpler version of this network was implemented. Compared to traditional image processing techniques, this approach was able to generalise better across different scenarios, additionally autoencoders are able to reconstruct the lost details in underexposed or low-light images without requiring separate processing stages, showcasing their versatility. The project aims to automate post-production tasks, with an end-to-end learning design it eliminates the separate image enhancement stages. Once built as standalone software or integrated into existing platforms, it could automatically enhance multiple images, requiring no user input.

Overview of project: 

Created a paired dataset from the MIT-Adobe Fivek dataset
-preprocessed the images; normalised and rescaled the image
-degraded the quality of the image by reducing the contrast and brightness to simulate 
-Split dataset into train, test and predict 

Implemented the baseline model as a prototype
Experimented with different layers e.g depth and height
Fine-tuned paramaters and optimised hyperparamaters

Dataset Preparation:

    Utilised the MIT-Adobe FiveK dataset to create a paired dataset.
    Preprocessing Steps: Normalized and rescaled images for consistency.
    Degradation Process: Artificially reduced image contrast and brightness to simulate lower quality images, mimicking real-world scenarios like low-light conditions.

Model Implementation:

    Established a baseline model to serve as a prototype for initial experiments.
    Layer Experimentation: Varied the depth and height of layers in the neural network architecture to explore their impact on performance.
    Fine-Tuning and Optimisation: Conducted rigorous fine-tuning of model parameters. This included optimising hyperparameters such as learning rate, batch size, and number of epochs to enhance model efficacy.

Spiltting dataset:

    Divided the dataset into training, testing, and prediction sets.

In comparison to Mao et al.'s network, this network has been adapted for a slightly different purpose and was trained on color images, which increases computational costs. Currently, the results show that the network performs as expected. However, planned work includes increasing the complexity of the network to recover (reconstruct) details from images severely degraded due to low-light conditions. As of now, the data is consistent with artifical corruption, futher work is to increase the variation of the corruption. 

Further work in the future is to leverage the networks versatility to experiment with both enhancement and restoration tasks simulatenously such as denosing. [planned after my graduation].

