from mtcnn.mtcnn import MTCNN
import cv2
import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


detector = MTCNN()
image = cv2.imread("images/be_iu.jpg")
result = detector.detect_faces(image)

for person in result:
    bounding_box = person['box']
    keypoints = person['keypoints']

    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255),
                  2)
    cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

cv2.imshow("image", image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]])
cv2.waitKey(0)
