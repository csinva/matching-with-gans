<p align="center">
  <a href="#projection-and-manipulation">Projection and manipulation</a> •
  <a href="#matching-and-benchmarking">Matching and benchmarking</a> •
  <a href="#disentangling-latent-space">Disentangling latent space</a> 
  <br>
  <a href="#reproducibility">Reproducibility</a> •
  <a href="#reference">Reference</a> 
</p>


# projection and manipulation
This code allows one to project images into the GAN latent space, after which they can be modified for certain attributes (e.g. age, gender, hair-length) and mixed with other faces (e.g. other people, older/younger versions of the same person).

- all this code is handled by the `projection_manipulation/project_and_manipulate.sh` script
- to start, put images into a directory (e.g. projection_manipulation/sample_projection/test) - no need to crop / align, this will be done automatically
    - higher-res photos work better, as well as photos where the face is front-facing and not obstructed by things like hats, scarves, etc.


<p align="center">
    <img src="projection_manipulation/sample_projection/chandan.jpg" width="200px" align="center">
</p>

![](projection_manipulation/sample_projection/manipulated/chandan_01.png)
    
    
# matching and benchmarking
- [matching_benchmarking](matching_benchmarking) folder contains code for reproducing the matching and benchmarking results obtained here
    
# disentangling latent space
- experiments to disentangle the latent space of stylegan2
- annotations are available (`Z.npy`, `W.npy`) from gdrive folder (place them in the data/annotation-dataset-stylegan2 folder)


# reproducibility

## setup
- run `source activate python3`
- `pip install tensorflow-gpu==1.14.0`
- start the lab running in the background: `screen jupyter lab --certfile=~/ssl/mycert.pem --keyfile ~/ssl/mykey.key`

## processed data
- `df.csv` in `data_processed/celeba-hq` contains labels along with different metrics for each image

## rerunning pipeline: download all files available in [this gdrive folder](https://drive.google.com/drive/folders/1YO_GZ48o30jTnME-z7d8LlcZoJejcNsk?usp=sharing)
- requires downloading the celeba-hq dataset at 1024 x 1024 resolution (zip file)
    - images should be place at data/celeba-hq/ims
    - annotations are provided in the data/celeba-hq/Anno folder
- different distances can be downloaded as `.npy` files (30k x 30k matrices)
    - `dists_pairwise_facial.npy` - pairwise distances measure by dlib face-rec encodings
    - `dists_pairwise_vgg.npy` - pairwise distances measure by vgg16 perceptual distance (only first 4 layers)


## distances
- analysis here requires the pairwise distance between all 30k images of 3 types: gan dist, facial-rec dist, vgg dist
    - we have calculated each of these matrices store in the gdrive folder
    - they should be placed in the data_processed/ folder


# reference
- this project builds on many wonderful open-source projects (see the readmes in the [lib](lib) subfolders for more details) including
- [stylegan2](https://github.com/NVlabs/stylegan2) and [stylegan2 encoder](https://github.com/rolux/stylegan2encoder)
- facial recogntion: [dlib](https://github.com/davisking/dlib), python [face_recognition](https://face-recognition.readthedocs.io/en/latest/face_recognition.html), [facenet](https://github.com/davidsandberg/facenet)
- [fairface](https://github.com/joojs/fairface)
- [deep_head_pose](https://github.com/shahroudy/deep-head-pose), [face_segmentation](https://github.com/nasir6/face-segmentation), and [faceQnet](https://github.com/uam-biometrics/FaceQnet)

