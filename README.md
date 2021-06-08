# video_face_recognition
Face recognition in videos
____
### Install everything you need and run
#### Installation
```
git clone https://github.com/b0shka/video_face_recognition.git
cd video_face_recognition
pip3 install -r requirements.txt
```
#### Run
##### To get the face parameters and create it .pickle file run the file training_madel.py
```
python3 ./training_madel.py
```
##### To search for a person in a video, run the file check_in_video.py
```
python3 ./check_in_video.py

```
##### To find multiple people in a video, run the file check_in_video_name_people.py
```
python3 ./check_in_video_name_people.py

```
### Additionally
In the `data` folder, you need to put folders with photos of people for training the model. And for these photos, `.pickle` files will be created in the `pickle_files` folder with the face parameters for later determining the faces in the video.
