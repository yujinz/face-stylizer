# Artistically Stylizing Portrait Photos using Portrait Paintings
Sample output: see [this report](https://github.com/yujinz/art-face/blob/master/documents/CS543%20Project%20Final%20Report.pdf)

### Usage:
python main.py ./examples/art/01.jpg ./examples/target/01.jpg

### Dependencies:
cv2, dlib (use https://pypi.python.org/simple/dlib/ to install on Windows), skimage, numpy, matplotlib

### Face landmark model:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

### Moving Least Square implementation: 
https://github.com/Jarvis73/Moving-Least-Squares 

### Reference:
Face landmark detection starter code: http://dlib.net/face_landmark_detection.py.html & https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

Workflow inspired by: Jakub Fiser, Ondrej Jamriska, David Simons, Eli Shechtman, Jingwan Lu, Paul Asente, Michal Lukac and Daniel Sykora. “Example-Based Synthesis of Stylized Facial Animations”. ACM Transactions on Graphics 36(4):155, 2017 (SIGGRAPH 2017). July 2017.

