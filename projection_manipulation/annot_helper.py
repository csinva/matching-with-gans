import os
from os.path import join as oj

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import pickle as pkl
sys.path.append('..')

import data
import style
import config
from config import *
import util
import glob
import json, jsonlines
import viz
import annot_helper

sys.path.append(oj(DIR_REPO, 'disentangling_latent_space'))
from annotation_dset import annotationDatabase


def get_annotations():
    ANNOTATIONS_DIR = DIR_PROJ_ANNOTATIONS
    LABELS_FILE = '/annotation-tool/data.json'
    OUT_MANIFEST_FILE = '/manifests/output/output.manifest'
    WORKERS_RESPONSE = '/annotations/worker-response'
    OUTPUT_PDF_DIR = './figures' # location of output pdfs
    # Get list of experiments in annotation dir. Alternatively, specify ones you care about.
    annotations_dict = {}
    annotations_names_dict = {}
    experiments = [os.path.split(f)[1] for f in glob.glob(os.path.join(ANNOTATIONS_DIR, '*'))]
    print('experiments', experiments)
    experiments = ['.'] #'perona-causal-faces-uncanny-000']

    # If output directory doesn't exist, make it
    if not os.path.exists(OUTPUT_PDF_DIR):
        os.makedirs(OUTPUT_PDF_DIR)

    # Do analysis figures for each experiment
    for EXPERIMENT_LABEL in experiments:
        ANNOTATIONS_PATH = f'{ANNOTATIONS_DIR}/{EXPERIMENT_LABEL}{WORKERS_RESPONSE}'
        OUT_MANIFEST_PATH = f'{ANNOTATIONS_DIR}/{EXPERIMENT_LABEL}{OUT_MANIFEST_FILE}'
        LABELS_PATH = f'{ANNOTATIONS_DIR}/{EXPERIMENT_LABEL}{LABELS_FILE}'
        OUT_PDF_FILE_NAME = os.path.join(OUTPUT_PDF_DIR, EXPERIMENT_LABEL + '.pdf')

        #Read labels using annotation-tool/data.json and assign integers to labels 
        with open(LABELS_PATH, 'r') as labels_file:
            labels_data = json.load(labels_file)['labels']
            LABELS = [l['label'] for l in labels_data]
        # labelScores = {l:i for (i, l) in enumerate(LABELS)} # generate numerical scores for the labels - useful in regression
        labelScores = {l:l for (i, l) in enumerate(LABELS)} # generate numerical scores for the labels - useful in regression

        # Get ordered list of image names from output manifest
        image_names = []
        with jsonlines.open(OUT_MANIFEST_PATH) as reader:
            for obj in reader:
                _, name = os.path.split(obj['source-ref']) #remove leading path
                image_names.append(name)

        # Make map from annotation index to image index
        idx_map = []
        with jsonlines.open(OUT_MANIFEST_PATH) as reader:
            for obj in reader:
                _, name = os.path.split(obj['source-ref']) #remove leading path
    #             print(name)
                idx = name.split('.')[0]
                idx_map.append(idx)

        # Make annotation file name list in proper order (keep same order)
        annotation_file_names = []
        for i in range(len(idx_map)):
            annotation_file_names += glob.glob(ANNOTATIONS_PATH + '/*/%d/*.json'% i)



        # put together the database of the annotator IDs and their work
        annotations = annotationDatabase(annotation_file_names, labelScores, label_type='crowd-image-classifier-multi-select')
        annotations.startPDFReport(OUT_PDF_FILE_NAME, EXPERIMENT_LABEL)
        fig = annotations.displayAnnotatorsWork()
    #     annotations.displaySequenceAnnotations(image_names, LABELS, SEQUENCE_LENGTH, IMAGE_PATH)
        annotations.endPDFReport()

        print(LABELS, np.array(annotations.imageScores).shape)
        annotations_dict[EXPERIMENT_LABEL] = np.array(annotations.imageScores)
        annotations_names_dict[EXPERIMENT_LABEL] = LABELS

    mat_list = annotations_dict['.']
    return annotations, mat_list

def same_or_not(l):
    if 'Same person' in l:
        return True
    elif 'Not same person' in l:
        return False
    else:
        return np.nan
def knows_well(l):
    if 'Well' in l:
        return 1
    elif 'Moderately well' in l:
        return 2
    elif 'Not at all' in l:
        return 3
    else:
        return np.nan
    
def add_intersections(labs, l1, l2, drop_dup=False):
    '''Add values for intersections of keys in l1, l2
    '''
    ks = []
    for k1 in l1:
        for k2 in l2:
            k_full = f'{k1} ({k2})'
            labs[k_full] = np.array(labs[k1]) & np.array(labs[k2])
            if drop_dup:
                labs[k_full] &= np.array(~labs['dup'])
            ks.append(k_full)
    return labs, ks