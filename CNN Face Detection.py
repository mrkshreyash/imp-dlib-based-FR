# OUTPUT 1 (Upsample == 1):
#     Processing file: 5_faces.jpg
#     Number of faces detected: 5
#     Detection 0: Left: 909 Top: 367 Right: 1113 Bottom: 571 Confidence: 1.0844861268997192
#     Detection 1: Left: 389 Top: 424 Right: 559 Bottom: 594 Confidence: 1.0681785345077515
#     Detection 2: Left: 1383 Top: 450 Right: 1587 Bottom: 654 Confidence: 1.061462163925171
#     Detection 3: Left: 1180 Top: 373 Right: 1350 Bottom: 542 Confidence: 1.0556831359863281
#     Detection 4: Left: 647 Top: 373 Right: 817 Bottom: 542 Confidence: 1.046241044998169
#     Hit enter to continue
#     Total time: 452.67066764831543

# OUTPUT 2 (Upsample == 0):
# Processing file: 5_faces.jpg
# Number of faces detected: 5
# Detection 0: Left: 912 Top: 379 Right: 1108 Bottom: 576 Confidence: 1.085799217224121
# Detection 1: Left: 374 Top: 399 Right: 571 Bottom: 596 Confidence: 1.07722008228302
# Detection 2: Left: 1389 Top: 459 Right: 1586 Bottom: 655 Confidence: 1.070913553237915
# Detection 3: Left: 1187 Top: 380 Right: 1350 Bottom: 544 Confidence: 1.049091100692749
# Detection 4: Left: 671 Top: 398 Right: 807 Bottom: 534 Confidence: 1.042119026184082
# Hit enter to continue
# Total time: 113.53456544876099


import dlib
import time
import cv2
import imutils
initial_time = time.time()

model_path, image_path = r"cnn_by_dlib/mmod_human_face_detector.dat", "different_lights_2.jpg"

cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)

win = dlib.image_window()

f = image_path

print("[Info] Processing file: {}".format(f))

img = cv2.imread(image_path)
img = imutils.resize(img, width=600)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# The 1 in the second argument indicates that we should upsample the image
# 1 time.  This will make everything bigger and allow us to detect more
# faces.
dets = cnn_face_detector(img, 1)
'''
This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
These objects can be accessed by simply iterating over the mmod_rectangles object
The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.

It is also possible to pass a list of images to the detector.
    - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)

In this case it will return a mmod_rectangless object.
This object behaves just like a list of lists and can be iterated over.
'''
print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
        i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

rects = dlib.rectangles()
rects.extend([d.rect for d in dets])

final_time = time.time()

win.clear_overlay()
win.set_image(img)
win.add_overlay(rects)
dlib.hit_enter_to_continue()

print(f"[Info] Total time: {final_time - initial_time}")
