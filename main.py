import cv2
from observer.detector import Observer
from identifier.recognition import Identifier
from analyzer.action import Analyzer
from decision.logic import DecisionMaker
import time

# 1. Initialize all Research Modules
obs = Observer()
ide = Identifier(folder_path="identifier/database")
ana = Analyzer()
dec = DecisionMaker() # Initialize the timer logic

# Load video from file instead of webcam
cap = cv2.VideoCapture('identifier/database/falling.mp4')

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('final_research_demo.mp4', fourcc, 20.0, (640, 480))

print("System Active. Monitoring for Safety...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        person_boxes = obs.get_persons(frame)

        for box in person_boxes:
            x1, y1, x2, y2 = box
            person_crop = frame[max(0, y1):y2, max(0, x1):x2]
            if person_crop.size == 0: continue

            # Stage 1: Who is it?
            name = ide.identify(person_crop)

            # Stage 2: What is the posture?
            action = ana.analyze_behavior(person_crop)

            # Stage 3: Should we trigger an alert?
            alert_message = dec.process(name, action)

            # Visualization Logic
            display_text = f"{name} | {action}"
            color = (0, 255, 0) # Green for safe

            if action == "FALLEN":
                color = (0, 165, 255) # Orange for "Caution"
                if alert_message:
                    color = (0, 0, 255) # Red for "Confirmed Emergency"
                    display_text = alert_message
                    # This is where you'd trigger a buzzer or SMS in a real system
                    print(f"!!! {alert_message} !!!")

            # Draw the UI
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        cv2.imshow("Behavior Awareness", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by User")
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Demo saved as final_research_demo.mp4")