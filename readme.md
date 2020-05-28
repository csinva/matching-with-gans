# installation / setup from scratch
- simplest: pick AMI that supported tensorflow 1.15.2
	- just run source activate tensorflow_p36
	- stylegan2 code works here
- installing from scratch
	- requires an installation of python 3.6 (followed instructions from here: https://realpython.com/installing-python/)
		- update python command to run python 3.6 with an alias
		- install pip
	- install pipenv
	`pipenv shell --python 3.6`

	- installing things
	```
	pipenv install tensorflow-gpu==1.15.0
	pip install scipy==1.3.3
	pip install requests==2.22.0
	pip install Pillow==6.2.1
```

# initial running
python run_generator.py generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
  --seeds=6600-6625 --truncation-psi=0.5
- notes
	- had to make minor changes to stylegan code to get it to compile properly

