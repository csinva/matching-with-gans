BEST_MODEL = '07_relu_retrain_3lay_noise_big'
REAL_LATENT_ENCODINGS_DIR = '../models/stylegan2encoder/generated_images'
ATTRS = ['age', 'facial-hair', 'skin-color', 'gender', 'hair-length', 'makeup']
RESULTS_DIR = 'results'
PROCESSED_DIR = 'processed'
LABELS = {
    'C': 'Skin color',
    'H': 'Hair length',
    'G': 'Gender',
    'A': 'Age',
    'M': 'Makeup',
    'B': 'Facial Hair',    
}