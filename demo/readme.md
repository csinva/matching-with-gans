# installation

- install streamlit from [here](https://docs.streamlit.io/en/latest/troubleshooting/clean-install.html#install-streamlit-on-macos-linux)

```
git clone https://github.com/streamlit/demo-face-gan.git
cd demo-face-gan
pip install -r requirements.txt
streamlit run app.py
```

- when using, use `pipenv shell` from the demo repo before running
- adapted from https://github.com/streamlit/demo-face-gan/

# description

This project highlights Streamlit's new `hash_func` feature with an app that calls on TensorFlow to generate photorealistic faces, using Nvidia's [Progressive Growing of GANs](https://research.nvidia.com/publication/2017-10_Progressive-Growing-of) and Shaobo Guan's [Transparent Latent-space GAN](https://blog.insightdatascience.com/generating-custom-photo-realistic-faces-using-ai-d170b1b59255) method for tuning the output face's characteristics. For more information, check out the [tutorial on Towards Data Science](https://towardsdatascience.com/building-machine-learning-apps-with-streamlit-667cef3ff509). 

The Streamlit app is [implemented in only 150 lines of Python](https://github.com/streamlit/demo-face-gan/blob/master/app.py) and demonstrates the wide new range of objects that can be used safely and efficiently in Streamlit apps with `hash_func`. 

![In-use Animation](https://github.com/streamlit/demo-face-gan/blob/master/GAN-demo.gif?raw=true "In-use Animation")
