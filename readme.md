<h1 align="center">Matching in GAN-Space</h1>

<p align="center">Code for using GANs to aid in matching, accompanying the paper "Overcoming confounding in face datasets via GAN-based matching" [arXiv]().
</p>

<p align="center">
  <a href="#projection-and-manipulation">Projection and manipulation</a> •
  <a href="#matching-and-benchmarking">Matching and benchmarking</a>
  <br>
  <a href="#reproducibility">Reproducibility</a> •
  <a href="#reference">Reference</a> 
  <br>
  <a href="https://colab.research.google.com/drive/1zevDVuqXc_ARcbirJAfEzsk1SClzBqXf">Quickstart demo</a> (manipulate and interpolate your own face images!): <a href="https://colab.research.google.com/drive/1zevDVuqXc_ARcbirJAfEzsk1SClzBqXf"> <img src="https://colab.research.google.com/assets/colab-badge.svg"> </a>
</p>


# Projection and manipulation
This code allows one to project images into the GAN latent space, after which they can be modified for certain attributes (e.g. age, gender, hair-length) and mixed with other faces (e.g. other people, older/younger versions of the same person). All this code is handled by the `projection_manipulation/project_and_manipulate.sh` script - the easiest way to get started is to use the [Colab notebook](https://colab.research.google.com/drive/1zevDVuqXc_ARcbirJAfEzsk1SClzBqXf), where you can upload your own images, and they will be automatically cropped, aligned projected, manipulated, and interpolated


Start with 2 real images (higher-res photos work better, as well as photos where the face is front-facing and not obstructed by things like hats, scarves, etc.):
<p align="center">
    <img src="projection_manipulation/samples/chandan.jpg" width="15%">
    <img src="projection_manipulation/samples/pietro.jpg" width="15%">
</p>


Interpolating between the images:
<p align="center">
    <img src="projection_manipulation/samples/interpolated/pietro_01_chandan_01.png" width="100%">
</p>

Manipulating an image along pre-specified attributes:
<p align="center">
    <img src="projection_manipulation/samples/manipulated/chandan_01.png" width="100%">
</p>

Can do a lot more, like blending together many faces or interpolating between different faces of the same person!
    
# Matching and benchmarking
The matching code [here](matching_benchmarking) finds images that match across a certain attribute (e.g. gender). This is useful for removing confounding factors when doing downstream analyses of things like bias in facial recognition. Similarly, we can perform matching using other methods, such as propensity scores, using the GAN latent space as covariates. Some example matches:

<p align="center">
    <img src="matching_benchmarking/samples/matches_big_gender_0.jpg" width="100%">
    <img src="matching_benchmarking/samples/matches_big_gender_1.jpg" width="100%">
</p>


After performing matching, confounding is much lower on CelebA-HQ. This is illustrated by the fact that the mean values of several key (binary) attributes become much closer after matching:

<p align="center">
    <img src="matching_benchmarking/samples/matching_covariates.jpg" width="100%">
</p>   

# Reproducibility

## Dependencies
- tested with python 3.6 and tensorflow-gpu 1.14.0 (the gpu dependencies are only required for the projection / manipulation code which uses  [StyleGAN2](https://github.com/NVlabs/stylegan2))
- the required dependencies can be set up on AWS by selecting a deep learning AMI, running `source activate python3`, and then running `pip install tensorflow-gpu==1.14.0`

## Data/cached outputs for reproducing pipeline [this gdrive folder](https://drive.google.com/drive/folders/1YO_GZ48o30jTnME-z7d8LlcZoJejcNsk?usp=sharing)

- `data/celeba-hq/ims` folder
  - unzip the images in  celeba-hq dataset at 1024 x 1024 resolution into this folder
- `data/processed` folder
  - distances: `dists_pairwise_gan.npy`, `dists_pairwise_vgg.npy`, `dists_pairwise_facial.npy`, `dists_pairwise_facial_facenet.npy`, `dists_pairwise_facial_facenet_casia.npy`, `dists_pairwise_facial_vgg2.npy` - (30k x 30k) matrices storing the pairwise distances between all the images in celeba-hq using different distance measures
- `data/processed/gen/generated_images_0.1`
  - latents `celeba_hq_latents_stylegan2.zip` - these are used in downstream analysis and are required for the propensity score analysis
- (already present) - annotations (e.g. gender, smiling, eyeglasses) + predicted metrics (e.g. predicted yaw, roll, pitch, quality, race) for each image + latent StyleGAN2 directions for different attributes + precomputed match numbers
- (optional) can download the raw annotations and annotated images as well
- (optional) all these paths can be changed in the `config.py` file

## Scripts

Both the [matching_benchmarking](matching_benchmarking) folder and the [projection_manipulation](projection_manipulation) folder contain two types of files:
- `.py` files in the `scripts` subdirectories - these scripts are used to calculate the cached outputs in the gdrive folder. They do not need to be rerun, but show how the cached outputs were generated and can be rerun on new datasets.
- `.ipynb` notebooks - these are used to reproduce the results from the cached outputs in the gdrive folde. Noteboks beginning with `eda` are for exploratory analysis, which can be useful but are note required to generate the final results in the paper

# Reference
- this project builds on many wonderful open-source projects (see the readmes in the [lib](lib) subfolders for more details) including
- stylegan: [stylegan2](https://github.com/NVlabs/stylegan2) and [stylegan2 encoder](https://github.com/rolux/stylegan2encoder)
- facial recogntion: [dlib](https://github.com/davisking/dlib), python [face_recognition](https://face-recognition.readthedocs.io/en/latest/face_recognition.html), [facenet](https://github.com/davidsandberg/facenet)
- gender/race prediction: [fairface](https://github.com/joojs/fairface)
- pose/background prediction: [deep_head_pose](https://github.com/shahroudy/deep-head-pose), [face_segmentation](https://github.com/nasir6/face-segmentation), and [faceQnet](https://github.com/uam-biometrics/FaceQnet)
