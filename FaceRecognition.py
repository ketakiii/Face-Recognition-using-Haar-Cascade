import cv2
import numpy as np
import pickle
import csv
from PIL import Image
import os
import random
import pandas as pd

def faces():
    # Get user supplied values
    cascPath = "haarcascade_frontalface_default.xml"
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(
        'C:/Users/ketaki/PycharmProjects/First/cascades/data/haarcascade_frontalface_alt2.xml')

    eye_cascade = cv2.CascadeClassifier(
        'C:/Users/ketaki/PycharmProjects/First/cascades/data/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(
        'C:/Users/ketaki/PycharmProjects/First/cascades/data/haarcascade_smile.xml')

    # recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("C:/Users/ketaki/PycharmProjects/First/recognizer/face-trainer.yml")

    # labels
    labels = {"person_name": 1}
    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}  # reversing the order

        # Read the image
        image = cv2.imread("C:/Users/ketaki/Downloads/o.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # print("Found {0} face!".format(len(faces)))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            #print(x, y, w, h)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]

            # recognize?
            id_, conf = recognizer.predict(roi_gray)
            if conf >= 4:
                #print(id_)
                #print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(image, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(image, "Person not recognized", (x, y), font, 1, color, stroke, cv2.LINE_AA)

            img_item = "my-img.jpg"
            cv2.imwrite(img_item, roi_gray)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)

            img_item = "color.jpg"
            cv2.imwrite(img_item, roi_color)

            subitems = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in subitems:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow("Faces found", image)
        cv2.waitKey(0)

        list = os.listdir("C:/Users/ketaki/PycharmProjects/First/images")  # dir is your directory path
        number_files = len(list) - 1
        #print(number_files)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, "images")

        for root, dirs, files in os.walk(image_dir):
                rand = str(random.randint(0, number_files))
                print(""+rand)

        if conf >= 4:
            name = labels[id_]
            df = pd.read_csv("mycsv.csv", index_col='name')
            if name in open("mycsv.csv").read():
                first = df.loc[name]
                print(first)

            else:
                print(labels[id_], "Enter your details below.")
                name = labels[id_]
                rollno = int(input("Enter you roll no: "))
                grno = input("Enter your GR No: ")
                add = input("Enter you address: ")
                email = input("Enter your E-mail id: ")
                phone = input("Enter you Mobile number :")
                a = len(phone)
                if a > 10 or a < 10:
                    print("Enter only a 10 digit phone no")
                    input("Enter you Mobile number :")
                with open('mycsv.csv', 'a', newline='') as f:
                    thewriter = csv.writer(f)
                    thewriter.writerow([rand, name, rollno, grno, add, phone, email])


        else:
                print("Face not found!")
                name2 = input("Enter the name of the new person: ")
                os.mkdir("C:/Users/ketaki/PycharmProjects/First/images/" + name2)
                path = "C:/Users/ketaki/PycharmProjects/First/images/" + name2
                img_Item = name2 + ".jpg"
                print(img_Item)
                cv2.imwrite(os.path.join(path, img_Item), image)

                rollno = int(input("Enter you roll no: "))
                grno = input("Enter your GR No: ")
                add = input("Enter you address: ")
                email = input("Enter your E-mail id: ")
                phone = input("Enter you Mobile number :")
                a = len(phone)
                if a > 10 or a < 10:
                    print("Enter only a 10 digit phone no")
                    input("Enter you Mobile number :")
                with open('mycsv.csv', 'a', newline='') as f:
                    thewriter = csv.writer(f)
                    thewriter.writerow([name2, rollno, grno, add, phone, email])



