from ultralytics import YOLO
from ultralytics import NAS
import cv2
import matplotlib.pyplot as plt
import urllib.request
import os
import numpy as np

# Download necessary files if not present
# def download_darknet_file(url, filename):
#     if not os.path.exists(filename):
#         print(f"Downloading {filename}...")
#         urllib.request.urlretrieve(url, filename)

# # URLs for YOLOv3 files
# YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
# YOLO_CFG_URL = "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg"
# COCO_NAMES_URL = "https://github.com/pjreddie/darknet/raw/master/data/coco.names"

# # File names
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# WEIGHTS_FILE = os.path.join(BASE_DIR, "yolov3.weights")
# CFG_FILE = os.path.join(BASE_DIR, "yolov3.cfg")
# NAMES_FILE = os.path.join(BASE_DIR, "coco.names")

# # Download files if they don't exist
# download_darknet_file(YOLO_WEIGHTS_URL, WEIGHTS_FILE)
# download_darknet_file(YOLO_CFG_URL, CFG_FILE)
# download_darknet_file(COCO_NAMES_URL, NAMES_FILE)

# # Load YOLOv3 Model
# def load_yolo3_model():
#     net = cv2.dnn.readNet(WEIGHTS_FILE, CFG_FILE)
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#     return net, output_layers

# # Load COCO Class Names
# def load_yolo3_classes():
#     with open(NAMES_FILE, "r") as f:
#         classes = [line.strip() for line in f.readlines()]
#     return classes

# Perform Object Detection
def detect_yolo3_objects(image, net, output_layers, classes):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.25:
                center_x, center_y, w, h = map(int, detection[:4] * np.array([width, height, width, height]))
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(classes[class_id])
    return boxes, confidences, class_ids

######################################################################

def load_yolo_model(model_path:str = 'yolov8x.pt'):
    model = YOLO(model_path) 
    return model

def run_inference(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image, conf=0.25)
    bounding_boxes = []
    probabilities = []
    entity_names = []

    for result in results:
        for box in result.boxes:
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            score = box.conf[0].item()  # Confidence score
            cls_id = box.cls[0].item()  # Class ID
            class_name = model.names[int(cls_id)]  # Map ID to class name
            bounding_boxes.append(bbox)
            probabilities.append(score)
            entity_names.append(class_name)

    return bounding_boxes, probabilities, entity_names, image

def draw_boxes(image, bounding_boxes, probabilities, entity_names):
    for bbox, score, name in zip(bounding_boxes, probabilities, entity_names):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        label = f"{name} {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

##################################################

def main(image_path, model_path):
    print(f"욜로 : {model_path}")

    if model_path == 'yolov3x.pt': #형식 통일을 위함. 
        # 실제로는 pt가 존재하진 않고 darknet방식 따름
        net, output_layers = load_yolo3_model()
        classes = load_yolo3_classes()
        image = cv2.imread(image_path)
        boxes, confidences, class_ids = detect_yolo3_objects(image, net, output_layers, classes)
        result = []
        for i in range(len(confidences)):
            result.append((class_ids[i],confidences[i]))
        result = list({t[0]: t for t in result}.values())
        return result
    
    elif model_path == 'yolov8x.pt':
        model = load_yolo_model(model_path)
        bounding_boxes, probabilities, entity_names, image = run_inference(model, image_path)
        all = []
        for bbox, prob, name in zip(bounding_boxes, probabilities, entity_names):
            # print(f"Entity: {name}, Probability: {prob:.2f}, Bounding Box: {bbox}")
            all.append((name,prob))
        unique_items = {}
        for item in all:
            entity, probability = item
            if entity not in unique_items or probability > unique_items[entity]:
                unique_items[entity] = probability
        result = [(entity, probability) for entity, probability in unique_items.items()]
        
        return result
    else:
        print(f"Your YOLO path: {model_path}\nValid YOLO path: 'yolov3x.pt', 'yolov8x.pt'")
        return None
    
if __name__ == "__main__":
    # Path to the input image
    number = 575931
    input_image_path = f"/home/onomaai/deeptext_multicaption/jihoon/coco2014/COCO_val2014_{int(number):012d}.jpg"  
    print(main(input_image_path, 'yolov8x.pt'))
