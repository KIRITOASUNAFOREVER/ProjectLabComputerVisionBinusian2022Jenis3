#!/usr/bin/env python3

import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of each person
    '''
    return sorted(os.listdir(root_path))


def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''
    image_list = []
    image_classes_list = []

    for idx, train_name in enumerate(train_names):
        train_dir = os.path.join(root_path, train_name)

        for image_name in get_path_list(train_dir):
            image_file = os.path.join(train_dir, image_name)
            image = cv2.imread(image_file)

            image_list.append(image)
            image_classes_list.append(idx)

    return image_list, image_classes_list


def detect_train_faces_and_filter(image_list, image_classes_list):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered image classes id
    '''
    train_face_grays = []
    filtered_classes_list = []

    face_cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")

    for image, class_id in zip(image_list, image_classes_list):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            image,
            scaleFactor=1.2,
            minNeighbors=5,
        )

        for face in faces:
            x,y,w,h = face

            train_face_grays.append(image[y:y+h, x:x+w])
            filtered_classes_list.append(class_id)

    return train_face_grays, filtered_classes_list


def detect_test_faces_and_filter(image_list):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
    '''
    test_faces_gray = []
    test_faces_rects = []

    face_cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")

    for image, class_id in zip(image_list, image_classes_list):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            image,
            scaleFactor=1.2,
            minNeighbors=5,
        )

        for face in faces:
            x,y,w,h = face

            test_faces_gray.append(image[y:y+h, x:x+w])
            test_faces_rects.append(face)

    return test_faces_gray, test_faces_rects


def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))
    return recognizer


def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all image in the test directories
    '''
    image_list = []

    for image_name in get_path_list(test_root_path):
        image_file = os.path.join(test_root_path, image_name)

        image = cv2.imread(image_file)
        image_list.append(image)

    return image_list


def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        test_faces_gray : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    result = []
    
    for image in test_faces_gray:
        class_id, confidence = recognizer.predict(image)
        result.append(class_id)

    return result


def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, size):
    '''
        To draw prediction results on the given test images and resize the image

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories
        size : number
            Final size of each test image

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''
    n = len(predict_results)

    for idx in range(n):
        x,y,w,h = test_faces_rects[idx]

        cv2.rectangle(
            test_image_list[idx],
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2,
        )

        test_image_list[idx] = cv2.resize(test_image_list[idx], (size, size))

        cv2.putText(
            test_image_list[idx],
            train_names[predict_results[idx]],
            (15, 20),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            lineType=1
        )

    return test_image_list


def combine_and_show_result(image_list, size):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
        size : number
            Final size of each test image
    '''
    n = len(predict_results)

    for idx in range(n):
        image_list[idx] = cv2.resize(test_image_list[idx], (size, size))

    result = cv2.hconcat(image_list)

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''

if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "./dataset/train/"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, filtered_classes_list = detect_train_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "./dataset/test/"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects = detect_test_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, 200)
    
    combine_and_show_result(predicted_test_image_list, 200)