from ultralytics import YOLO
from ultralytics import NAS
import cv2
import matplotlib.pyplot as plt

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

def main(image_path, model_path):
    model = load_yolo_model(model_path)
    print(f"욜로 : {model_path}")
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

if __name__ == "__main__":
    # Path to the input image
    number = 575931
    input_image_path = f"/home/donut2024/coco2014/COCO_val2014_{int(number):012d}.jpg"  
    print(main(input_image_path, 'yolov8x.pt'))
