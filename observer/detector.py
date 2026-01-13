from ultralytics import YOLO

class Observer:
    def __init__(self):
        # Load a small and fast YOLOv8 model
        self.model = YOLO('yolov8n.pt') 

    def get_persons(self, frame):
        # Detect only class 0 (Person) with 50% confidence
        results = self.model(frame, classes=[0], conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        return boxes