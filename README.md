# lbph-face-recognition

This is a simple face recognition project using Python OpenCV, made to help some friends at work.

### Requirements

- Python 3.6+ (Anaconda is recommended)
- OpenCV 
- Numpy
- Pandas

### How to Use

*First of all, navigate to the directory where the project is.*

### Photo Shooting

1. Run `python take-photos.py`.
2. Enter an ID and a name.
3. Enter a camera ID (the default is 0), if you only have one camera in your PC, just press `ENTER`.
4. Press the `s` key to take photos of your face (it will only work when your face is detected and there's 
enough light in the room).
5. Press the `q` key when you're done.

- Take at least 25 photos of the face of each person.
- Repeat this with at least one different person, otherwise you will get an error.

### Training

1. Run `python train.py`.
2. The program will generate the file "classifiers/lbphClassifier.yml"

- This may take some time depending on how many photos you took.

### Recognizing

1. Run `python recognize.py`

- The number under your name is the "Trust Distance", the less the distance (closer to 0), the more reliable is the
classification.
