import cv2
import torch
from ultralytics import YOLO  # Import your YOLO model

class ObjectTracker:
    def __init__(self, video_path, model):
        self.video_path = video_path
        self.model = model
        self.trackers = {}

    def track_objects_by_id(self):
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection using YOLO model
            detections = self.model(frame)

            # Update or create trackers for each detected object ID
            for box, confidence, obj_id in detections:
                if obj_id not in self.trackers:
                    self.trackers[obj_id] = cv2.TrackerKCF_create()
                    self.trackers[obj_id].init(frame, box)
                else:
                    success, box = self.trackers[obj_id].update(frame)
                    if success:
                        x, y, w, h = [int(coord) for coord in box]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {obj_id}, Confidence: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage:
video_path = 1

# Load your custom pre-trained YOLO model
model = YOLO('train11/weights/best.pt') #'yolov5n.pt'   # Load your model architecture and weights

# Create an instance of the ObjectTracker class
object_tracker = ObjectTracker(video_path, model)

# Perform object tracking by ID
object_tracker.track_objects_by_id()