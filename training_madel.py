import face_recognition
import pickle
import os
import sys
import h5py

def traning_modal_img():
    if not os.path.exists("data"):
        print("[ERROR] Not found dirictory data")
        sys.exit()

    folders = os.listdir("data")

    if not os.path.exists("pickle_files"):
        os.mkdir("pickle_files")
    users_pickle_list = os.listdir("pickle_files/")

    for i in folders:
        if str(i) + ".pickle" not in users_pickle_list:
            print(f"[+] Training {i}")
            face_encodings_user = []
            images = os.listdir(f"data/{i}")
            name = i
            for (j, image) in enumerate(images):
                try:
                    print(f"[+] Processing image {j+1}/{len(images)}")

                    img = face_recognition.load_image_file(f"data/{i}/{image}")
                    img_param = face_recognition.face_encodings(img)

                    if len(img_param) != 0:
                        for param in img_param:
                            face_encodings_user.append(param)
                except IndexError:
                    pass

            data = {
                "name" : name,
                "encodings" : face_encodings_user
            }

            # Write to .h5 file
            '''with h5py.File('data.h5', 'a') as file:
                file.create_dataset(f'{name}', data=face_encodings_user)'''

            with open(f"pickle_files/{name}.pickle", "wb") as file:
                file.write(pickle.dumps(data))

            print(f"[INFO] File {name}.pickle successfully created")


def main():
    traning_modal_img()

if __name__ == "__main__":
    main()