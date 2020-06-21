from os.path import join as oj

# for fitting the linear models
BEST_MODEL = '07_relu_retrain_3lay_noise_big'
REAL_LATENT_ENCODINGS_DIR = '../models/stylegan2encoder/generated_images'
DIRECTIONS_DIR = '../data/annotation-dataset-stylegan2/linear_models/new' # path to many datasets, includes on directory before the below dirs
GENERATING_LATENTS_DIR = '../data/annotation-dataset-stylegan2/data'

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
RESULTS_DIR = 'results'
PROCESSED_DIR = 'processed'

# data
FILES_DIR = '/home/ubuntu/face-disentanglement/'
DIR_CELEBA_IMS = oj(FILES_DIR, 'data/celeba-hq/ims/')
DIRS_CELEBA_GEN = oj(FILES_DIR, 'data_processed/celeba-hq/')