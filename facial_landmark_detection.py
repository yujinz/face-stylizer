#!/usr/bin/python
from __future__ import division
from skimage import io
import sys
import os
import dlib
import cv2
import glob
import numpy as np

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = int(rect.left()/ratio)
    y = int(rect.top()/ratio)
    w = int(rect.right()/ratio) - x
    h = int(rect.bottom()/ratio) - y
 
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def get_facial_landmark(rects):
    for (i, rect) in enumerate(rects):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, rect.left(), rect.top(), rect.right(), rect.bottom()))
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(img_resized, rect)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))

        # Draw the face landmarks on the screen.
        win.add_overlay(shape)
        shape = shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = rect_to_bb(rect)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        # for (x, y) in shape:
        #     cv2.circle(img, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)
        #     print(str(int(x/ratio)) + " " + str(int(y/ratio)))        

    return shape


if len(sys.argv) != 2:
    print(
        "Give the directory containing the facial images.\n"
        "e.g.:\n"
        "    ./face_landmark_detection.py ./examples/faces\n")
    exit()

predictor_path = "shape_predictor_68_face_landmarks.dat"
# You can download a trained facial shape predictor from:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
faces_folder_path = sys.argv[1]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

win = dlib.image_window()

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = resize(img_grey, width=500)

    win.clear_overlay()
    win.set_image(img_resized)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.    
    rects = detector(img_resized, 1)

    landmarks = get_facial_landmark(rects)

    win.add_overlay(rects)

    dlib.hit_enter_to_continue()