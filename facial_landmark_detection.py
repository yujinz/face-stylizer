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
    h, w = img.shape[:2]

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (width, height), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (width, height), interpolation)
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

def draw_and_write_landmark(img_resized, detector, predictor, outfilename, is_disp = False):
    if is_disp:
        win = dlib.image_window()
        win.clear_overlay()
        win.set_image(img_resized)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.    
    rects = detector(img_resized, 1)

    if len(rects) != 1:
        return

    outfile = open("./output/" + outfilename + ".txt", "w+")
    for (i, rect) in enumerate(rects):
        print("Left: {} Top: {} Right: {} Bottom: {}".format(
            rect.left(), rect.top(), rect.right(), rect.bottom()))
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(img_resized, rect)

        # Draw the face landmarks on the screen.
        if is_disp:
            win.add_overlay(shape)
        shape = shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # (x, y, w, h) = rect_to_bb(rect)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            #if i<43 or i>48:
            outfile.write("%d %d\r\n" % (int(x), int(y)))
        #     cv2.circle(img, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)

    outfile.close() 
    if is_disp:
        win.add_overlay(rects)
        dlib.hit_enter_to_continue()


if __name__ == "__main__":
	if len(sys.argv) != 3:
	    print(
	        "Usage:\n"
	        "    ./facial_landmark_detection.py ./examples/art/01.jpg ./examples/target/01.jpg\n")
	    exit()

	predictor_path = "shape_predictor_68_face_landmarks.dat"
	# You can download a trained facial shape predictor from:
	# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
	art_face_path = sys.argv[1]
	target_face_path = sys.argv[2]

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

	#win = dlib.image_window()

	print("Processing file: {}".format(art_face_path))
	img = io.imread(art_face_path)
	h, w, c = img.shape
	img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	draw_and_write_landmark(img_grey, detector, predictor, "art")

	print("Processing file: {}".format(target_face_path))
	img = io.imread(target_face_path)
	img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_resized = resize(img_grey, width=w)
	draw_and_write_landmark(img_resized, detector, predictor, "target")