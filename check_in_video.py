import os
import face_recognition
import pickle
from cv2 import cv2
from threading import Thread

location = None
encoding = None
image = None
status_work = 0

def get_locations():
    global location
    global encoding
    global image
    global status_work

    while status_work == 1:
        try:
            location = face_recognition.face_locations(image, model="hog")[0]
            encoding = face_recognition.face_encodings(image, [location])[0]
        except IndexError as error:
            print(f"[ERROR] {error}")
        except:
            break

def detect_person_in_video():
    global location
    global encoding
    global image
    global status_work

    video = cv2.VideoCapture("video.mp4")
    ret, image = video.read()
    files_pickle = os.listdir("pickle_files")

    try:
        location = face_recognition.face_locations(image, model="hog")[0]
        encoding = face_recognition.face_encodings(image, [location])[0]
    except IndexError as error:
        print(f"[ERROR] {error}")

    status_work = 1
    find_face = Thread(target=get_locations)
    find_face.start()
    name = None

    while True:
        try:
            ret, image = video.read()

            for i in files_pickle:
                data = pickle.loads(open(f"pickle_files/{i}", "rb").read())
                result = face_recognition.compare_faces(data["encodings"], encoding)

                if True in result:
                    name = data["name"]
                    #print(f"This {name}")
                    left_top = (location[3], location[0])
                    right_bottom = (location[1], location[2])
                    color = [0, 255, 0]
                    cv2.rectangle(image, left_top, right_bottom, color, 4)

                    cv2.putText(
                        image,
                        name,
                        (location[3] + 10, location[2] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        4
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