import cv2
import torch
from torch.utils.data import DataLoader
#from your_yolo_model import YOLO  # Import your YOLO model
#from your_custom_dataset import CustomDataset  # Import your custom dataset class

def object_tracking_by_id(video_path):
    # Load your custom-trained YOLO model
    model = YOLO('yolov5n.pt')  # Load your model architecture and weights

    # Load your custom dataset
    custom_dataset = CustomDataset("train11/weights/best.pt")  # Initialize your custom dataset

    # Create DataLoader for iterating over the dataset in batches
    dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

    # Dictionary to store trackers for each object ID
    trackers = {}

    # Process video frames
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform YOLO detection on the frame
        # (Assuming your YOLO model returns bounding boxes, confidence scores, and object IDs)
        detections = model.detect(frame)

        # Update or create trackers for each detected object
        for box, confidence, obj_id in detections:
            if obj_id not in trackers:
                # Initialize tracker for new object ID
                trackers[obj_id] = cv2.TrackerKCF_create()
                x, y, w, h = box
                trackers[obj_id].init(frame, (x, y, w, h))
            else:
                # Update existing tracker
                success, box = trackers[obj_id].update(frame)
                if success:
                    x, y, w, h = [int(coord) for coord in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {obj_id}, Confidence: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
video_path = "1"
object_tracking_by_id(video_path)
