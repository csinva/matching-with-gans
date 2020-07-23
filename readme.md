<h1 align="center">Matching in GAN-Space</h1>

<p align="center">Code for using GANs to aid in matching, accompanying the paper "Causal matching with GANS" [arXiv]().
</p>

<p align="center">
  <a href="#projection-and-manipulation">Projection and manipulation</a> •
  <a href="#matching-and-benchmarking">Matching and benchmarking</a>
  <br>
  <a href="#reproducibility">Reproducibility</a> •
  <a href="#reference">Reference</a> 
</p>


# Projection and manipulation
This code allows one to project images into the GAN latent space, after which they can be modified for certain attributes (e.g. age, gender, hair-length) and mixed with other faces (e.g. other people, older/younger versions of the same person).

- all this code is handled by the `projection_manipulation/project_and_manipulate.sh` script
- to start, put images into a directory (e.g. projection_manipulation/sample_projection/test) - no need to crop / align, this will be done automatically
    - higher-res photos work better, as well as photos where the face is front-facing and not obstructed by things like hats, scarves, etc.


<p align="center">
    <img src="projection_manipulation/samples/chandan.jpg" width="200px" align="center">
</p>
<p align="center">
    <img src="projection_manipulation/samples/manipulated/chandan_01.png" width="100%" align="center">
</p>
    
    
# Matching and benchmarking
- [matching_benchmarking](matching_benchmarking) folder contains code for reproducing the matching and benchmarking results obtained here
    
# Disentangling latent space
- experiments to disentangle the latent space of stylegan2
- annotations are available (`Z.npy`, `W.npy`) from gdrive folder (place them in the data/annotation-dataset-stylegan2 folder)


# Reproducibility

## Setup
- running the code here requires installing the dependencies for [StyleGAN2](https://github.com/NVlabs/stylegan2)
    - on AWS, this can be done by selecting a deep learning AMI, running `source activate python3`, and then running `pip install tensorflow-gpu==1.14.0`

## Rerunning the pipeline: download all files available in [this gdrive folder](https://drive.google.com/drive/folders/1YO_GZ48o30jTnME-z7d8LlcZoJejcNsk?usp=sharing)

- `data/celeba-hq/ims` folder
  - unzip the images in  celeba-hq dataset at 1024 x 1024 resolution into this folder
- `data/processed` folder
  - distances: `dists_pairwise_gan.npy`, `dists_pairwise_vgg.npy`, `dists_pairwise_facial.npy`, `dists_pairwise_facial_facenet.npy`, `dists_pairwise_facial_facenet_casia.npy`, `dists_pairwise_facial_vgg2.npy` - (30k x 30k) matrices storing the pairwise distances between all the images in celeba-hq using different distance measures
  - already present in the data folder are annotations (e.g. gender, smiling, eyeglasses) + predicted metrics (e.g. predicted yaw, roll, pitch, quality, race) for each image + latent directions corresponding to different attributes for StyleGAN2
- `gen_latents` - these are used in downstream analysis and are required for the propensity score analysis
- (optional) can download the raw annotations and annotated images as well
- all these paths can be changed in the `config.py` file


# Reference
- this project builds on many wonderful open-source projects (see the readmes in the [lib](lib) subfolders for more details) including
- [stylegan2](https://github.com/NVlabs/stylegan2) and [stylegan2 encoder](https://github.com/rolux/stylegan2encoder)
- facial recogntion: [dlib](https://github.com/davisking/dlib), python [face_recognition](https://face-recognition.readthedocs.io/en/latest/face_recognition.html), [facenet](https://github.com/davidsandberg/facenet)
- [fairface](https://github.com/joojs/fairface)
- [deep_head_pose](https://github.com/shahroudy/deep-head-pose), [face_segmentation](https://github.com/nasir6/face-segmentation), and [faceQnet](https://github.com/uam-biometrics/FaceQnet)

