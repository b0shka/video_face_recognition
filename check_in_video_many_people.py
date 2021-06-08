import os
import face_recognition
import pickle
from cv2 import cv2
from threading import Thread
import time

location = []
encoding = []
image = None
status_work = 0

def get_locations():
    global location
    global encoding
    global image
    global status_work

    while status_work == 1:
        location = []
        encoding = []
        try:
            locations = face_recognition.face_locations(image, model="hog")
            for i in locations:
                location.append(i)
                encoding.append(face_recognition.face_encodings(image, [i]))
            time.sleep(1)
        except IndexError:
            location = []
        except:
            break

def detect_person_in_video():
    global location
    global encoding
    global image
    global status_work

    video = cv2.VideoCapture("videos/video1.mp4")
    ret, image = video.read()
    files_pickle = os.listdir("pickle_files")

    try:
        locations = face_recognition.face_locations(image, model="hog")
        for i in locations:
            location.append(i)
            encoding.append(face_recognition.face_encodings(image, [i]))
    except IndexError:
        location = []

    status_work = 1
    find_face = Thread(target=get_locations)
    find_face.start()
    name = None

    while True:
        try:
            ret, image = video.read()
            if len(location) != 0:
                for i in files_pickle:
                    data = pickle.loads(open(f"pickle_files/{i}", "rb").read())
                    for index, j in enumerate(encoding):
                        result = face_recognition.compare_faces(data["encodings"], j[0])
                        if True in result:
                            if name != data["name"]:
                                name = data["name"]
                            left_top = (location[index][3], location[index][0])
                            right_bottom = (location[index][1], location[index][2])
                            color = [0, 255, 0]
                            cv2.rectangle(image, left_top, right_bottom, color, 4)

                            cv2.putText(
                                image,
                                name,
                                (location[index][3] + 10, location[index][2] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                3
                            )

            cv2.imshow("detect_person_in_video is running", image)

            k = cv2.waitKey(50)
            if k == ord("q"):
                print("[INFO] Exit")
                status_work = 0
                find_face.join()
                break
        except TypeError as error:
            print(f"[ERROR] {error}")
        except:
            break

def main():
    if not os.path.exists("pickle_files"):
        print("[ERROR] Not found dirictory pickle_files")
    else:
        detect_person_in_video()

if __name__ == "__main__":
    main()