import os
import face_recognition
import pickle
import time
import h5py
from cv2 import cv2
from threading import Thread

location = None
encoding = None
image = None
status_work = 0
video = cv2.VideoCapture("videos/video1.mp4")
files_pickle = os.listdir("pickle_files")

def get_locations():
    global location
    global encoding
    global image
    global status_work

    while status_work == 1:
        try:
            location = face_recognition.face_locations(image, model="hog")[0]
            encoding = face_recognition.face_encodings(image, [location])[0]
            #time.sleep(1)
        except IndexError:
            location = None
        except Exception as error:
            print(f"[ERROR] {error}")
            break

def detect_person_in_video():
    global location
    global encoding
    global image
    global status_work

    ret, image = video.read()

    try:
        location = face_recognition.face_locations(image, model="hog")[0]
        encoding = face_recognition.face_encodings(image, [location])[0]
    except IndexError:
        location = None
    except Exception as error:
        print(f"[ERROR] {error}")

    status_work = 1
    find_face = Thread(target=get_locations)
    find_face.start()
    name = None
    time.sleep(0.5)
    find_face_1 = Thread(target=get_locations)
    find_face_1.start()

    #file_data = h5py.File('data.h5', 'r')

    while True:
        try:
            ret, image = video.read()

            if location != None:
                # Read from .h5 file
                '''for i in file_data.keys():
                    data = file_data[i]
                    result = face_recognition.compare_faces(data, encoding)

                    if True in result:
                        if name != i:
                            name = i
                            # print(f"This {name}")
                        left_top = (location[3], location[0])
                        right_bottom = (location[1], location[2])
                        color = [0, 255, 0]
                        cv2.rectangle(image, left_top, right_bottom, color, 4)

                        cv2.putText(
                            image,
                            name,
                            (location[3], location[2] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            3
                        )'''

                for i in files_pickle:
                    data = pickle.loads(open(f"pickle_files/{i}", "rb").read())
                    result = face_recognition.compare_faces(data["encodings"], encoding)

                    if True in result:
                        if name != data["name"]:
                            name = data["name"]
                            #print(f"This {name}")
                        left_top = (location[3], location[0])
                        right_bottom = (location[1], location[2])
                        color = [0, 255, 0]
                        cv2.rectangle(image, left_top, right_bottom, color, 4)

                        cv2.putText(
                            image,
                            name,
                            (location[3], location[2] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            3
                        )

            cv2.imshow("Video", image)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                status_work = 0
                find_face.join()
                find_face_1.join()
                break
        except TypeError as error:
            print(f"[ERROR] {error}")
        except Exception as error:
            print(f"[ERROR] {error}")
            status_work = 0
            find_face.join()
            find_face_1.join()
            break
    #file_data.close()

def main():
    if not os.path.exists("pickle_files"):
        print("[ERROR] Not found dirictory pickle_files")
    else:
        detect_person_in_video()

if __name__ == "__main__":
    main()
