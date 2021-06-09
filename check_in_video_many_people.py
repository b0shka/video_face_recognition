import face_recognition
import cv2
import os
import pickle
import h5py
import time
from threading import Thread

locations = None
encoding = None
frame = None
status_work = 0
video_capture = cv2.VideoCapture("videos/video1.mp4")
files_pickle = os.listdir("pickle_files")

def get_locations():
    global locations
    global encoding
    global frame
    global status_work

    while status_work == 1:
        try:
            locations = face_recognition.face_locations(frame, model="hog")
            encoding = face_recognition.face_encodings(frame, locations)
            time.sleep(0.3)
        except IndexError:
            locations = None
        except Exception as error:
            print(f"[ERROR] {error}")
            break

def detect_person_in_video():
    global locations
    global encoding
    global frame
    global status_work

    ret, frame = video_capture.read()

    try:
        locations = face_recognition.face_locations(frame, model="hog")
        encoding = face_recognition.face_encodings(frame, locations)
    except IndexError:
        locations = None
    except Exception as error:
        print(f"[ERROR] {error}")

    status_work = 1
    find_face = Thread(target=get_locations)
    find_face.start()
    name = None

    #file_data = h5py.File('data.h5', 'r')

    while True:
        try:
            ret, frame = video_capture.read()

            if locations != None:
                # Read from .h5 file
                '''for i in file_data.keys():
                    data = file_data[i]
                    for (top, right, bottom, left), face_encoding in zip(locations, encoding):
                        result = face_recognition.compare_faces(data, face_encoding)

                        if True in result:
                            if name != i:
                                name = i
                                
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
                            cv2.putText(frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                        3)'''

                for i in files_pickle:
                    data = pickle.loads(open(f"pickle_files/{i}", "rb").read())
                    for (top, right, bottom, left), face_encoding in zip(locations, encoding):
                        result = face_recognition.compare_faces(data["encodings"], face_encoding)

                        if True in result:
                            if name != data["name"]:
                                name = data["name"]

                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
                            cv2.putText(frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            cv2.imshow('Video', frame)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                status_work = 0
                find_face.join()
                break

        except Exception as error:
            print(f"[ERROR] {error}")
            status_work = 0
            find_face.join()
            break
    # file_data.close()

def main():
    if not os.path.exists("pickle_files"):
        print("[ERROR] Not found dirictory pickle_files")
    else:
        detect_person_in_video()

if __name__ == "__main__":
    main()