import dlib
import cv2


five_faces = r"5_faces.jpg"
diff_angle_1, diff_angle_2, diff_angle_3 = r"different_angles_1.jpg", r"different_angles_2.jpg", r"different_angles_3.jpg"
diff_lights_1, diff_lights_2 = r"different_lights_1.jpg", r"different_lights_2.jpg"
multi_face_1, multi_face_2, multi_face_3 = r"Multiple_faces.jpg", r"Multiple_faces_2.jpg", r"Multiple_faces_3.jpg"

img_path = five_faces
# img_path = diff_angle_1
# img_path = diff_angle_2

# img_path = diff_angle_3
# img_path = diff_lights_1
# img_path = diff_lights_2

# img_path = multi_face_1
# img_path = multi_face_2
# img_path = multi_face_3

image = cv2.imread(img_path)

# print(f"Image : {image.shape[0]}")
# image = cv2.resize(image, (1581, 934))
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hog_detector = dlib.get_frontal_face_detector()
faces = hog_detector(gray_image, 1)

print(f"# of Faces detected: {len(faces)}")

for (i, rect) in enumerate(faces):
    print(f"i, rect = {i, rect}")

    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    cv2.rectangle(image, (x, y), (x+w, y+w), (0, 255, 0), 2)

cv2.imshow('faces detected', image)
cv2.waitKey(0)

