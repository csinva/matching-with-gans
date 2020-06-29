from os.path import join as oj
DIR_REPO = '/home/ubuntu/face-disentanglement/'


# running and saving #################################
DIR_PROCESSED_MISC = oj(DIR_REPO, 'src', 'processed')
DIR_RESULTS = oj(DIR_REPO, 'src', 'results')
DIR_FIGS = oj(DIR_REPO, 'figs')

# data #################################
DIR_CELEBA = oj(DIR_REPO, 'data/celeba-hq')
DIR_IMS = oj(DIR_CELEBA, 'ims/')
DIR_PROCESSED = oj(DIR_REPO, 'data_processed/celeba-hq/')
DIR_GEN = oj(DIR_PROCESSED, 'generated_images_0.1')

# lib paths #################################
DIR_LIB = oj(DIR_REPO, 'lib')
DIR_STYLEGAN = oj(DIR_LIB, 'stylegan2')

# disentangling latent space #################################
BEST_MODEL = '07_relu_retrain_3lay_noise_big'
# REAL_LATENT_ENCODINGS_DIR = oj(REPO_DIR, 'models/stylegan2encoder/generated_images')
DIR_LINEAR_DIRECTIONS = oj(DIR_REPO, 'data/annotation-dataset-stylegan2/linear_models/new') # path to many datasets, includes on directory before the below dirs
DIR_GENERATING_LATENTS = oj(DIR_REPO, 'data/annotation-dataset-stylegan2/data')

# attrs
ATTRS = ['age', 'facial-hair', 'skin-color', 'gender', 'hair-length', 'makeup']
ATTRS_MEASURED = 'HAGCBM'
ALL_ATTRS = 'HAGCBMSEW' # 'HAGCBMSEW'
ATTR_LABELS = {
    'C': 'skin-color',
    'H': 'hair-length',
    'G': 'gender\n(perceived)',
    'A': 'age',
    'M': 'makeup',
    'B': 'facial-hair',    
 
     # these are ones we don't have explicit labels for
    'S': 'smiling',
    'E': 'eyeglasses',
    'W': 'earrings', # wearing earrings
}
ATTR_TO_INDEX = {
    ATTR_LABELS[ALL_ATTRS[i]]: i for i in range(len(ALL_ATTRS))
}