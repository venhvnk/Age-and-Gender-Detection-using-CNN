# Import required modules

import cv2   
import math
import argparse

 #highlightface gets the image from dataset or from the realtime image to detect the facial dimensions 
def highlightFace(net, frame, conf_threshold=0.7):
    #detect using the model opencvfacedetector.pbtx
    frameOpencvDnn=frame.copy()
    #detecting the face height
    frameHeight=frameOpencvDnn.shape[0]
    #detecting the face width
    frameWidth=frameOpencvDnn.shape[1]
    #blob is a dl functions, is presented during image recognition & image preprocessing
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
#face highliting (box)
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        #A copy of the frame is created in order to get the facial dimensions
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            #add this cordinates (x1,y1,x2,y2) to the faceboxes list
            faceBoxes.append([x1,y1,x2,y2])
            #drawing the rectangle arount faces
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

#the argparse library is utilized to construct an argument that can be processed to retreive the picture from the command prompt
parser=argparse.ArgumentParser()
#The parser récupaire the picture file contained in the argument (image path)
parser.add_argument('--image')

args=parser.parse_args()
#calling the trained model by calling their filles
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

#age & gender  liste 
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

#load network
#using the provided readnet function The network is loaded with faceModel and faceProto as arguments 
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

#open a video file or an image file or a camera stream
video=cv2.VideoCapture(args.image if args.image else 0)
padding=20
while cv2.waitKey(1)<0 :
    #read frame , The webcam’s Live Stream is read and then stored in the hasframe and frame structures
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    #there are no face detected 
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")
    #dorwing the face box 
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        #converts the image to this minimal format by removing all variables and additional components
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        #uses this image for detection by including  Principal operations ( subtraction of the mean, scaling, and modification of the model’s )
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]

        #print gender output
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]

        #print age output
        print(f'Age: {age[1:-1]} years')
        
        #executing the results
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
