import cv2 # type: ignore
import numpy as np # type: ignore

net = cv2.dnn.readNet("darknet/yolov4.weights", "darknet/cfg/yolov4.cfg")
classes = []
with open("darknet/cfg/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

width = 640
height = 480
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

cv2.namedWindow('Camera 1', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera 2', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Camera 1', 800, 600)
cv2.resizeWindow('Camera 2', 800, 600)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not (ret1 and ret2):
        print("Can't read Frame.")
        break
    
    blob1 = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    blob2 = cv2.dnn.blobFromImage(frame2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob1)
    outs1 = net.forward(output_layers)
    net.setInput(blob2)
    outs2 = net.forward(output_layers)
    
    def draw_boxes(frame, outs):
        height, width, channels = frame.shape
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    draw_boxes(frame1, outs1)
    draw_boxes(frame2, outs2)
    
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()