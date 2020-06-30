import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sy
# import src.style
import style
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit


class annotationDatabase():

    def __init__(self, annotations_file_names, labelScores):
        '''Parameters:
        annotations_file_names - list of names of jupyter files where the annotations are saved.
        Each file refers to one image.'''
        self.annotators = {}  # dictionary that maps the unique ID strings of annotators to local integer IDs
        self.annotator_names = []  # list of the unique string identifiers that AMT uses sorted by integer ID
        self.annotations = []  # list of the number of annotations per annotator
        self.N_ANNOTATORS = len(self.annotators)
        self.N_IMAGES = len(annotations_file_names)
        self.imageScores = []  # scores this image received -- list of lists (num images x num_annotaters per images)
        self.imageAnnotators = []  # annotators who worked on a given image -- list of lists

        #
        # for each file extract all the useful annotations
        #
        for i, fname in enumerate(annotations_file_names):
            with open(fname, 'r') as read_file:
                data = json.load(read_file)  # open the json file and read its contents into data
                scores = []  # initialize the two lists of scores and IDs for the current image
                annotatorIDs = []
                for a, ans in enumerate(data['answers']):  # read each annotator's annotation
                    ID = self.addAnnotation(ans['workerId'])  # mark annotation and retrieve ID of annotator
                    annotatorIDs.append(ID)  # take note of which annotator it was
                    label = data['answers'][a]['answerContent']['crowd-image-classifier']['label']
                    scores.append(labelScores[label])  # transform label into score

                self.imageScores.append(scores)
                self.imageAnnotators.append(annotatorIDs)

        print(f'Found {self.N_IMAGES} images and {self.N_ANNOTATORS} annotators.')

    def addAnnotation(self, annotator_name):
        '''Keep track of the annotations and of the annotators'''
        try:
            ID = self.annotators[annotator_name]  # the annotator was found, here is her ID
            self.annotations[ID] += 1  # chalk up one more annotation for this annotator
        except:  # the annotator was not on the list
            ID = self.N_ANNOTATORS  # create a new ID
            self.annotators[annotator_name] = ID
            self.N_ANNOTATORS += 1
            self.annotations.append(1)  # add a count of one for the last annotator
            self.annotator_names.append(annotator_name)
        return ID

    def annotatorID(self, annotator_name):
        '''Retrieve the integer ID of an annotator from his/her name string'''
        try:
            ID = self.annotators[annotator_name]  # the annotator was found, here is her ID
        except:  # the annotator was not on the list
            ID = -1
            print(f'Annotator {annotator_name} not found in annotator directory. Something is wrong.')
        return ID

    def displayAnnotatorsWork(self):
        '''plot the number of annotations per annotator'''
        n_annotations = sorted(self.annotations, reverse=True)
        fig = plt.figure(figsize=style.STANDARD_FIG_SIZE)
        plt.plot(range(1, 1 + self.N_ANNOTATORS), n_annotations)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('annotator n.')
        plt.ylabel('n. annotations')
        plt.title('Work of individual annotators')
        self.figList.append(fig)

    def displaySequenceAnnotations(self, image_names, label_names, SEQUENCE_LENGTH, IMAGE_PATH, N_SEQUENCES_TO_SHOW=3):
        # Make sure sequence length and number of images are compatible
        if self.N_IMAGES % SEQUENCE_LENGTH:
            raise ValueError("number of images must be divisible by sequence length")
        n_sequences = self.N_IMAGES // SEQUENCE_LENGTH

        # For each sequence, show images, scatter plot, and boxplots
        for i in range(N_SEQUENCES_TO_SHOW):
            fig = plt.figure(figsize=style.STANDARD_FIG_SIZE)

            # Read in images
            imgs = []
            for j in range(SEQUENCE_LENGTH):
                full_image_path = os.path.join(IMAGE_PATH,
                                               image_names[i * SEQUENCE_LENGTH + j])
                img = np.array(Image.open(full_image_path))
                imgs.append(img)

            # Concatenate images into one image, and show
            imgs = np.concatenate(imgs, axis=1)
            plt.subplot(3, 1, 1)
            plt.imshow(imgs)
            plt.axis('off')

            # Make scatterplot
            plt.subplot(3, 1, 2)
            plt.ylim(-1, len(label_names))
            plt.yticks(np.arange(len(label_names)), label_names)

            scores = []  # scores assigned by the annotators
            attrs = []  # value of the corresponding attribute
            for j in range(SEQUENCE_LENGTH):  # for each image in sequence
                scores_j = np.array(self.imageScores[i * SEQUENCE_LENGTH + j])
                attrs_j = np.array(j * np.ones(
                    scores_j.shape))  # these are the x coordinates, i.e. the value of the attribute parameter

                scores.append(scores_j)  # Store for boxplot
                attrs.append(attrs_j)  # Store for fitting the psycho curve
                '''
                # Median score
                med_score_j = np.median(scores_j)

                DATA_WIDTH = 0.25
                #nx = np.random.uniform(-DATA_WIDTH,DATA_WIDTH,scores_j.shape)
                #ny = np.random.uniform(-DATA_WIDTH,DATA_WIDTH,scores_j.shape)
                #plt.plot([j+1] * len(scores_j)+nx, scores_j+ny, 'ro', markersize=10)
                plt.plot([j+1-DATA_WIDTH, j+1+DATA_WIDTH], [med_score_j, med_score_j], 'k-', linewidth=3) # line to indicate median
                '''

            # fit the psychometric curve
            scores_flat = np.concatenate(scores)
            attrs_flat = np.concatenate(attrs) + 1
            par0 = sy.array([SEQUENCE_LENGTH / 2, 1.0, min(scores_flat), max(scores_flat)])  # starting values
            BOUNDS = [[3, 0.01, min(scores_flat), min(scores_flat)],
                      [SEQUENCE_LENGTH - 4, 1, max(scores_flat), max(scores_flat)]]
            SIGMA = 1.5  # error bars on the annotators' labels
            try:
                ## The psychometric function that is used to fit the data
                def pf(x, alpha=0, beta=1, m=0, M=1):
                    '''Logistic function
                    Parameters:
                    alpha - the horizontal offset (default=0)
                    beta - the slope (default=1)
                    m - minimum of the function (default=0)
                    M = maximum of the function  (default=1)
                    '''
                    return m + (M - m) * (1. / (1 + np.exp(
                        -(x - alpha) * (4 * beta))))  # 4*beta so that the derivative at alpha is beta

                par, mcov = curve_fit(pf, attrs_flat, scores_flat, par0, bounds=BOUNDS,
                                      sigma=SIGMA * np.ones(attrs_flat.shape))
                JJ = np.arange(1, SEQUENCE_LENGTH + 0.1,
                               0.1)  # x coordinates to be used to plot the psychometric function
                plt.plot(JJ, pf(JJ, par[0], par[1], par[2], par[3]), lw=50, c=[0.9, 0.9, 0.9])
                plt.title(f'fit: shift={par[0]:.2}, slope={par[1]:.2}, min={min(par[2:4]):.2}, max={max(par[2:4]):.2}')
            except:
                print('Sigmoid fit did not succeed')

            DATA_WIDTH = 0.25
            nx = np.random.uniform(-DATA_WIDTH, DATA_WIDTH, scores_flat.shape)
            ny = np.random.uniform(-DATA_WIDTH, DATA_WIDTH, scores_flat.shape)
            plt.plot(attrs_flat + nx, scores_flat + ny, 'ko', markersize=5)
            plt.xlim((0, SEQUENCE_LENGTH + 1))

            # Make boxplots
            ax = plt.subplot(3, 1, 3)
            plt.xlim((0, SEQUENCE_LENGTH + 1))
            plt.ylim(-1, len(label_names))
            plt.yticks(np.arange(len(label_names)), label_names)
            bp = ax.boxplot(scores)

            ## change color and linewidth of the boxes
            for box in bp['boxes']:
                box.set(color='#7570b3', linewidth=2)

            ## change color and linewidth of the whiskers
            for whisker in bp['whiskers']:
                whisker.set(color='#7570b3', linewidth=2)

            ## change color and linewidth of the caps
            for cap in bp['caps']:
                cap.set(color='#7570b3', linewidth=2)

            ## change color and linewidth of the medians
            for median in bp['medians']:
                median.set(color='black', linewidth=5)

            self.figList.append(fig)

    def startPDFReport(self, OUT_PDF_FILE_NAME, EXPERIMENT_LABEL):
        '''Open PDF where the data will be dumped.
        Displays a few stats of the dataset on a title screen for the PDF report'''

        # initialize the output file pointer and the list of figures
        self.outPDF = PdfPages(OUT_PDF_FILE_NAME)

        # set up figure that will be used to display the opening banner
        fig = plt.figure(figsize=style.STANDARD_FIG_SIZE)
        plt.axis('off')

        # stuff to be printed out on the first page of the report
        plt.text(0, 1, f'Experiment={EXPERIMENT_LABEL}', fontsize=36)
        plt.text(0, 2, f'Date of analysis: {str(datetime.datetime.now())}', fontsize=36)
        #         plt.text(0,3,f'Jupyter Notebook version: {NOTEBOOK_VERSION}',fontsize=36)
        plt.ylim([-1, 4])
        plt.gca().invert_yaxis()

        self.figList = []  # clear list of figures
        self.figList.append(fig)

    def endPDFReport(self):
        '''endReport: Writes figures to PDF file and closes it.'''
        for fig in self.figList:
            self.outPDF.savefig(fig)
            plt.close(fig)  # this will prevent the figures from being shown in Pyplot and will save them to PDF instead
        self.outPDF.close()
        self.figList = []  # clear list of figures
