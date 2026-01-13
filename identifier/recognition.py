import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

class Identifier:
    def __init__(self, folder_path, model_name='buffalo_l'):
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.known_embeddings = {} 
        self.threshold = 0.25 # Distance threshold for recognition
        
        # Load images from your actual folder
        self._load_database(folder_path)

    def _load_database(self, path):
        print(f"Loading Research Database from: {path}")
        if not os.path.exists(path):
            print(f"ERROR: Folder '{path}' not found!")
            return

        for file in os.listdir(path):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(path, file))
                faces = self.app.get(img)
                if faces:
                    name = os.path.splitext(file)[0]
                    self.known_embeddings[name] = faces[0].embedding
                    print(f"Registered: {name}")

    def identify(self, person_crop):
        """Identify a person from a cropped image"""
        faces = self.app.get(person_crop)
        if not faces:
            return "Unknown"
        
        test_embedding = faces[0].embedding
        
        best_match = "Unknown"
        best_distance = float('inf')
        
        for name, known_embedding in self.known_embeddings.items():
            distance = norm(test_embedding - known_embedding)
            # print(f"Distance to {name}: {distance:.4f}")
            if distance < best_distance:
                best_distance = distance
                best_match = name
        
        if best_distance > self.threshold:
            return "Unknown"
        
        return best_match