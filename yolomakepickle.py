import pickle

from cv2.ml import LogisticRegression
from ultralytics import YOLO
model = YOLO('train11/weights/best.pt') #'yolov5n.pt'

#model.fit(X_train, Y_train)
# save the model to disk
filename = 'niva_Car_Trained_YOLO.pkl'
pickle.dump(model, open(filename, 'wb'))