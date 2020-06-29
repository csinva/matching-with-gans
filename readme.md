# installation / setup from scratch

- run `source activate python3`
- `pip install tensorflow-gpu==1.14.0`
- start the lab running in the background: `screen jupyter lab --certfile=~/ssl/mycert.pem --keyfile ~/ssl/mykey.key`


# data

- requires downloading the celeba-hq dataset at 1024 x 1024 resolution (zip file can be downloaded from [here](https://drive.google.com/drive/folders/1YO_GZ48o30jTnME-z7d8LlcZoJejcNsk?usp=sharing))
    - images should be place at data/celeba-hq/ims
    - annotations are provided in the data/celeba-hq/Anno folder


# projection / manipulation
- put image into a directory (e.g. projection_manipulation/test)
    - run ./project_new and you will get an altered image
    - sample original is here: ![](projection_manipulation/sample_projection/chandan.jpg)
    - result is here: ![](projection_manipulation/sample_projection/manipulated/chandan_01.png)