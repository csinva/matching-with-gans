from os.path import join as oj
import os 
DIR_REPO = os.path.dirname(os.path.realpath(__file__)) # directory of the config file


# running and saving #################################
# DIR_PROCESSED_MISC = oj(DIR_REPO, 'processed')
DIR_RESULTS = oj(DIR_REPO, 'src', 'results')
DIR_FIGS = oj(DIR_REPO, 'figs')

# data #################################
DIR_CELEBA = oj(DIR_REPO, 'data/celeba-hq')
DIR_IMS = oj(DIR_CELEBA, 'ims/')
DIR_PROCESSED = oj(DIR_REPO, 'data/processed/')
DIR_GEN = oj(DIR_PROCESSED, 'gen', 'generated_images_0.1')

# lib paths #################################
DIR_LIB = oj(DIR_REPO, 'lib')
DIR_STYLEGAN = oj(DIR_LIB, 'stylegan2')


# attrs in latent space
DIR_LINEAR_DIRECTIONS = oj(DIR_REPO, 'data/annotations_stylegan2/linear_models') # linear models for attributes on latent space
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


# different directories with source code
DIR_ANNOTATIONS = oj(DIR_REPO, 'data', 'annotations_celeba-hq')
DIR_PROJ_ANNOTATIONS = oj(DIR_ANNOTATIONS)