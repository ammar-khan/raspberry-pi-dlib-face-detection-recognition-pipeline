##
# Copyright 2018, Ammar Ali Khan
# Licensed under MIT.
##

import face_recognition
import pickle
import cv2
import os
from src.common.package.config import application as config_app
from src.dlib.package.config import application as config_dlib
from src.common.package.io.handler import Handler as io_handler
from src.common.package.progress.handler import progress as progress_bar


##
# Training class
##
class Training:
    def __init__(self):
        # Grab the paths to the input images in data set
        print('[INFO] Quantifying dataset...')
        self.image_paths = list(io_handler.files(directory=config_app.STORAGE_DIRECTORY,
                                                 match='*' + config_app.FILE_EXTENSION))

        # Initialise list
        self.encodings = []
        self.descriptions = []

        return None

    ##
    # Static method get_description()
    # Method to extract description from directory name
    #
    # @param path - path
    #
    # @return string description
    ##
    @staticmethod
    def get_description(path):
        return path.split(os.path.sep)[-2].replace('_', ' ').title()

    ##
    # Method pre_processing()
    # Method to prepared data & train model
    #
    # @return encodings, descriptions
    ##
    def pre_processing(self):
        print('[INFO] Processing {} images...'.format(len(self.image_paths)))
        for (idx, image_path) in enumerate(self.image_paths):
            try:
                # Progress bar
                progress_bar(idx + 1, len(self.image_paths), status='Completed')

                # Load image
                frame = cv2.imread(image_path)

                # Convert image from RGB (OpenCV ordering) to dlib ordering (RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect the (x, y)-coordinates of the bounding boxes,
                # corresponding to each detection in the input image
                detection = face_recognition.face_locations(frame, model=config_dlib.DETECTION_MODEL)

                # Extract description from directory name
                description = self.get_description(path=image_path)

                # Encode detection to recognise
                print('[INFO] Training model...')
                encodings = face_recognition.face_encodings(frame, detection)

                for encoding in encodings:
                    # Insert encoding and description to list
                    self.encodings.append(encoding)
                    self.descriptions.append(description)

            except Exception as e:
                print('[ERROR] Exception %s' % str(e))
                continue

        # Need to print carriage return because of progress bar
        print('\r')

        return self.encodings, self.descriptions

    ##
    # Method train()
    # Method to train model
    ##
    def train(self):
        # Make directory to store model
        io_handler.make_dir(directory=config_dlib.TRAINED_MODEL_DIRECTORY)

        # Pre-process dataset and train
        encodings, descriptions = self.pre_processing()

        print('[INFO] Serializing encodings...')
        data = {'encodings': encodings, 'names': descriptions}

        # Save trained model
        trained_model = config_dlib.TRAINED_MODEL_DIRECTORY + config_dlib.TRAINED_MODEL_FILE
        print('[INFO] Trained model saved to: ', trained_model)
        file = open(trained_model, 'wb')
        file.write(pickle.dumps(data))
        file.close()


##
# Method main()
##
def main():
    try:
        print('[INFO] Initialising Dlib training...')
        training = Training()
        training.train()

    except Exception as e:
        print('[ERROR] Exception: %s' % str(e))


if __name__ == '__main__':
    main()
