import cv2
import pyttsx3

textToSpeech = pyttsx3.init()
capture = cv2.VideoCapture(0)
capture.set(3,640)
capture.set(4,480)

objectClassNames = []
objectClassFile = 'coco.names'
with open(objectClassFile,'rt') as f:
    objectClassNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(weightsPath, configPath)
model.setInputSize(320,320)
model.setInputScale(1.0/ 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)
spokenTexts = []
while True:
    success, capturedImage = capture.read()
    classIds, confs, bbox = model.detect(capturedImage, confThreshold=0.5)

    if len(classIds) != 0:

        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):

            cv2.rectangle(capturedImage,box,color=(0,255,0), thickness=2)
            cv2.putText(capturedImage,objectClassNames[classId-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
            text = objectClassNames[classId-1]
            if text not in spokenTexts:
                spokenTexts.append(text)
                textToSpeech.say(text)
                textToSpeech.say(text)
                textToSpeech.runAndWait()

    cv2.imshow("Image output",capturedImage)
    cv2.waitKey(1)