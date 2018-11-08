##
# Copyright 2018, Ammar Ali Khan
# Licensed under MIT.
##

import face_recognition
import time
import cv2
import imutils
import pickle
from src.common.package.config import application as config_app
from src.dlib.package.config import application as config_dlib
from src.common.package.http import server as http_server
from src.common.package.http.handler import Handler
from src.common.package.camera.handler import Handler as camera
from src.common.package.io.handler import Handler as io_handler
from src.common.package.file.handler import Handler as file_handler
from src.common.package.frame.handler import Handler as frame_handler
from src.common.package.face.align import Handler as FaceAlign
from src.common.package.tracker.handler import Handler as Tracker
from src.opencv.package.opencv.handler import Handler as OpenCV

# Constant
opencv = OpenCV()
tracker = Tracker()
face_align = FaceAlign  # Not using yet, trying to find efficient and accurate way to align faces


##
# StreamHandler class - inherit Handler
# This class provide handler for HTTP streaming
# Note: this class should override Handler.stream
##
class StreamHandler(Handler):

    ##
    # Override method Handler.stream()
    ##
    def stream(self):
        Handler.stream(self)
        print('[INFO] Overriding stream method...')

        # Initialise capture
        capture = camera(src=config_app.CAPTURING_DEVICE,
                         use_pi_camera=config_app.USE_PI_CAMERA,
                         resolution=config_app.RESOLUTION,
                         frame_rate=config_app.FRAME_RATE)

        if config_app.USE_PI_CAMERA:
            print('[INFO] Warming up pi camera...')
        else:
            print('[INFO] Warming up camera...')

        time.sleep(2.0)

        # Load trained recogniser model
        print('[INFO] Loading model...')
        model = config_dlib.TRAINED_MODEL_DIRECTORY + config_dlib.TRAINED_MODEL_FILE

        # Exit if no model trained
        if not io_handler.file_exist(model):
            print('[ERROR] Please train the model first!')
            exit(0)

        # Initialise recogniser
        print('[INFO] Initialise recogniser...')
        recognizer = pickle.loads(open(model, 'rb').read())

        print('[INFO] Start capturing...')
        while True:
            # Read a frame from capture
            frame = capture.read()

            # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame (speedup processing)
            frame = imutils.resize(frame, width=750)
            resize_ratio = frame.shape[1] / float(frame.shape[1])

            # Detect the (x, y)-coordinates of the bounding boxes,
            # corresponding to each detection in frame
            detections = face_recognition.face_locations(frame, model=config_dlib.DETECTION_MODEL)

            # Encode detections to recognise
            encodings = face_recognition.face_encodings(frame, detections)

            # Initialise list of names
            names = []

            # Iterate over each encodings found in frame
            for encoding in encodings:
                # Compare detected encodings with trained model encodings
                matches = face_recognition.compare_faces(recognizer['encodings'], encoding)

                # Set generic detection name
                name = 'Unknown'

                # Found a match
                if True in matches:
                    # Create dictionary of all matched indexes
                    matched_idx = [i for (i, b) in enumerate(matches) if b]

                    # Iterate over all matches and count the total number of times each face was matched
                    counts = {}
                    for idx in matched_idx:
                        name = recognizer['names'][idx]
                        counts[name] = counts.get(name, 0) + 1

                    # Recognised face based on the largest number of count
                    name = max(counts, key=counts.get)

                # Insert name in list
                names.append(name)

            # Reset trackers
            tracker.reset_trackers()

            # Iterate over detections
            for ((top, right, bottom, left), name) in zip(detections, names):
                # Re-scale the coordinates
                coordinates = {'left': int(left * resize_ratio),
                               'top': int(top * resize_ratio),
                               'right': int(right * resize_ratio),
                               'bottom': int(bottom * resize_ratio)}

                # Create or update tracker
                tracker_id = tracker.track(coordinates)

                # Crop detection from image (frame)
                detection = frame_handler.crop(image=frame, coordinates=coordinates)

                # Convert detection to gray scale
                detection = cv2.cvtColor(detection, cv2.COLOR_RGB2GRAY)

                if name is 'Unknown':
                    # Save unknown detection to storage directory
                    directory = file_handler.folder_name(tracker_id)
                    if not io_handler.dir_exist(directory):
                        io_handler.make_dir(directory=directory)

                    io_handler.save_file(image=detection,
                                         filename=file_handler.file_name(),
                                         directory=directory)

                # Create box with id or description
                frame = frame_handler.rectangle(frame=frame,
                                                coordinates=coordinates,
                                                text=name)

            # Write date time on the frame
            frame = frame_handler.text(frame=frame,
                                       coordinates={'left': config_app.WIDTH - 150, 'top': config_app.HEIGHT - 20},
                                       text=time.strftime('%d/%m/%Y %H:%M:%S', time.localtime()),
                                       font_color=(0, 0, 255))

            # Convert frame into buffer for streaming
            retval, buffer = cv2.imencode('.jpg', frame)

            # Write buffer to HTML Handler
            self.wfile.write(b'--FRAME\r\n')
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', len(buffer))
            self.end_headers()
            self.wfile.write(buffer)
            self.wfile.write(b'\r\n')


##
# Method main()
##
def main():
    try:
        # Create directory to store detections
        print('[INFO] Create directory for storage...')
        io_handler.make_dir(directory=config_app.STORAGE_DIRECTORY)

        # Prepare and start HTTP server
        address = ('', config_app.HTTP_PORT)
        server = http_server.Server(address, StreamHandler)
        print('[INFO] HTTP server started successfully at %s' % str(server.server_address))
        print('[INFO] Waiting for client to connect to port %s' % str(config_app.HTTP_PORT))
        server.serve_forever()
    except Exception as e:
        server.socket.close()
        print('[INFO] HTTP server closed successfully.')
        print('[ERROR] Exception: %s' % str(e))


if __name__ == '__main__':
    main()
