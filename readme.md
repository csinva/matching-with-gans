# installation / setup from scratch

- pick AMI that supported tensorflow 1.15.2
  - just run source activate tensorflow_p36
  - stylegan2 code works here
- pip install tensorflow-gpu==1.14.0


```
# initial running
python run_generator.py generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
  --seeds=6600-6625 --truncation-psi=0.5
- notes
	- had to make minor changes to stylegan code to get it to compile properly

```


- start the lab running in the background
`screen jupyter lab --certfile=~/ssl/mycert.pem --keyfile ~/ssl/mykey.key`
