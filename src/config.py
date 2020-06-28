from os.path import join as oj
DIR_REPO = '/home/ubuntu/face-disentanglement/'

# for fitting the linear models
BEST_MODEL = '07_relu_retrain_3lay_noise_big'
# REAL_LATENT_ENCODINGS_DIR = oj(REPO_DIR, 'models/stylegan2encoder/generated_images')
DIR_LINEAR_DIRECTIONS = oj(DIR_REPO, 'data/annotation-dataset-stylegan2/linear_models/new') # path to many datasets, includes on directory before the below dirs
DIR_GENERATING_LATENTS = oj(DIR_REPO, 'data/annotation-dataset-stylegan2/data')

# attrs
ATTRS = ['age', 'facial-hair', 'skin-color', 'gender', 'hair-length', 'makeup']
ATTRS_MEASURED = 'HAGCBM'
ALL_ATTRS = 'HAGCBMSEW' # 'HAGCBMSEW'
LABELS = {
    'C': 'skin-color',
    'H': 'hair-length',
    'G': 'gender',
    'A': 'age',
    'M': 'makeup',
    'B': 'facial-hair',    
 
     # these are ones we don't have explicit labels for
    'S': 'smiling',
    'E': 'eyeglasses',
    'W': 'earrings', # wearing earrings
}
ATTR_TO_INDEX = {
    LABELS[ALL_ATTRS[i]]: i for i in range(len(ALL_ATTRS))
}

# running things
RESULTS_DIR = oj(DIR_REPO, 'src', 'results')
PROCESSED_DIR = oj(DIR_REPO, 'src', 'processed')

# data
DIR_IMS = oj(DIR_REPO, 'data/celeba-hq/ims/')
DIR_PROCESSED = oj(DIR_REPO, 'data_processed/celeba-hq/')
DIR_GEN = oj(DIR_REPO, 'data_processed/celeba-hq/generated_images_0.1')

# lib paths
LIB_PATH = oj(DIR_REPO, 'lib')
DIR_STYLEGAN = oj(LIB_PATH, 'stylegan2')
