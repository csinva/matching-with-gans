# matching + benchmarking

This folder contains code for matching using GAN latent space and then subsequently benchmarking facial recognition systems. This section does not require `StyleGAN2` dependencies, instead requiring the precomputed distances between images in GAN space.

The `eda` notebooks give exploratory analysis into different dataset and aspects of the matching. The non-eda notebooks are used to reproduce the results in the paper.

In the scripts folder, scripts starting with `00` predict attributes for each image in the dataset and scripts starting with `01` calculate pairwise distances between all images (pairwise distances for GAN latents is relatively very quick and does not have its own script).