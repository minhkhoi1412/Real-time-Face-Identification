# for taking images from webcam
import time
from mtcnn.mtcnn import MTCNN
from utility import *
import cv2
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']


age_net = cv2.dnn.readNetFromCaffe('models/deploy_age.prototxt', 'models/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('models/deploy_gender.prototxt', 'models/gender_net.caffemodel')


def detect_face(database, model):
    detector = MTCNN()
    save_loc = r'saved_image/1.jpg'
    capture_obj = cv2.VideoCapture(0)
    capture_obj.set(3, 480)  # WIDTH
    capture_obj.set(4, 852)  # HEIGHT

    # face_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')

    # whether there was any face found or not
    face_found = False

    # run the webcam for given seconds
    req_sec = 5
    loop_start = time.time()
    elapsed = 0

    while True:
        curr_time = time.time()
        elapsed = curr_time - loop_start
        if elapsed >= req_sec:
            break

        # capture_object frame-by-frame
        ret, frame = capture_obj.read()
        # mirror the frame
        frame = cv2.flip(frame, 1, 0)

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect face
        faces = detector.detect_faces(frame)
        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Display the resulting frame
        if faces:
            for face in faces:
                x, y, w, h = face['box']
                # required region for the face
                roi_color = frame[y-90:y+h+70, x-50:x+w+50]
                # save the detected face
                cv2.imwrite(save_loc, roi_color)
                # draw a rectangle bounding the face
                cv2.rectangle(frame, (x-10, y-70),
                              (x+w+20, y+h+40), (15, 175, 61), 4)

            # display the frame with bounding rectangle
            cv2.imshow('frame', frame)

            # close the webcam when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # release the capture_object
    capture_obj.release()
    cv2.destroyAllWindows()

    img = cv2.imread(save_loc)
    if img is not None:
        face_found = True
    else:
        face_found = False

    return face_found


# detects faces in realtime from webcam feed
def detect_face_realtime(database, model, emotion_model, threshold=0.7):
    detector = MTCNN()
    text = ''
    font = cv2.FONT_HERSHEY_SIMPLEX
    save_loc = r'saved_image/1.jpg'
    capture_obj = cv2.VideoCapture(0)
    capture_obj.set(3, 1280)  # WIDTH
    capture_obj.set(4, 720)  # HEIGHT

    # face_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')
    print('**************** Enter "q" to quit **********************')
    prev_time = time.time()

    while True:

        # capture_object frame-by-frame
        ret, frame = capture_obj.read()
        # mirror the frame
        # frame = cv2.flip(frame, 1, 0)

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect face
        faces = detector.detect_faces(frame)
        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Display the resulting frame
        if faces:
            for face in faces:
                x, y, w, h = face['box']
                # required region for the face
                roi_color = frame[y-90:y+h+70, x-50:x+w+50]
                if roi_color.any():
                    # save the detected face
                    cv2.imwrite(save_loc, roi_color)
                else:
                    pass

                # keeps track of waiting time for face recognition
                curr_time = time.time()

                if curr_time - prev_time >= 5:
                    img = cv2.imread(save_loc)
                    if img is not None:
                        resize_img(save_loc)
                        min_dist, identity, registered = find_face_realtime(save_loc, database, model, threshold)
                        if min_dist <= threshold and registered:
                            # for putting text overlay on webcam feed
                            text = 'Hello ' + identity
                            print('Hello ' + identity + '!')

                        else:
                            text = 'Unknown user'
                            print('Unknown user' + ' detected !')

                        print('Distance:' + str(min_dist))
                    # save the time when the last face recognition task was done
                    prev_time = time.time()

                # draw a rectangle bounding the face
                cv2.rectangle(frame, (x-10, y-70), (x+w+20, y+h+40), (15, 175, 61), 4)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray is not None:
                    roi_gray = gray[y:y + h, x:x + w]
                    if roi_gray is not None:
                        face_img = frame[y:y + h, h:h + w].copy()
                        blob = cv2.dnn.blobFromImage(face_img, 1, (244, 244), MODEL_MEAN_VALUES, swapRB=True)
                        if blob is not None:
                            # Predict Gender
                            gender_net.setInput(blob)
                            gender_preds = gender_net.forward()
                            gender = gender_list[gender_preds[0].argmax()]
                            # Predict Age
                            age_net.setInput(blob)
                            age_preds = age_net.forward()
                            age = age_list[age_preds[0].argmax()]
                            overlay_text = "%s %s" % (gender, age)
                            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

                            prediction = emotion_model.predict(cropped_img)
                            max_index = int(np.argmax(prediction))
                            cv2.putText(frame, overlay_text, (x + 300, y), font, 1,
                                        (0, 255, 255), 2, cv2.LINE_AA)

                            cv2.putText(frame, emotion_dict[max_index], (x + 300, y - 30), font, 1,
                                        (0, 255, 255), 2, cv2.LINE_AA)

                            cv2.putText(frame, text, (150, 150), font, 1.8, (158, 11, 40), 3)

        # display the frame with bounding rectangle
        cv2.imshow('frame', frame)

        # close the webcam when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture_object
    capture_obj.release()
    cv2.destroyAllWindows()


# checks whether the input face is a registered user or not
def find_face_realtime(image_path, database, model, threshold):
    # find the face encodings for the input image
    encoding = img_to_encoding(image_path, model)
    registered = False
    min_dist = 99999
    identity = 'Unknown Person'
    # loop over all the recorded encodings in database
    for name in database:
        # find the similarity between the input encodings and claimed person's encodings using L2 norm
        dist = np.linalg.norm(np.subtract(database[name], encoding))
        # check if minimum distance or not
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > threshold:
        registered = False
    else:
        registered = True
    return min_dist, identity, registered
