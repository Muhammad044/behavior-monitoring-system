import numpy as np
from ultralytics import YOLO

class Analyzer:
    def __init__(self):
        # We use the Pose model to get body "Keypoints" (skeleton)
        self.pose_model = YOLO('yolov8n-pose.pt')

    def analyze_behavior(self, frame):
        results = self.pose_model(frame, verbose=False)
        behavior_label = "Standing" # Default
        
        for r in results:
            if r.keypoints is None or len(r.keypoints.xyn) == 0: 
                continue
            
            # Get keypoints for the first person detected
            # Keypoints: [0]=nose, [5,6]=shoulders, [11,12]=hips, [15,16]=ankles
            points = r.keypoints.xyn[0].cpu().numpy()
            
            if len(points) < 17: continue

            # 1. GET COORDINATES
            nose_y = points[0][1]
            hip_y = (points[11][1] + points[12][1]) / 2
            ankle_y = (points[15][1] + points[16][1]) / 2

            # 2. LOGIC
            height_of_person = ankle_y - nose_y
            hip_to_floor = ankle_y - hip_y

            if height_of_person < 0.3:
                behavior_label = "FALLEN"
            elif hip_to_floor < 0.2:
                behavior_label = "Sitting"
            else:
                behavior_label = "Standing"

        return behavior_label