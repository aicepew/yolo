import cv2
from ultralytics import YOLO
import random
import socket
import struct
import datetime
import os


current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
save_dir = 'images/'

if not os.path.exists(save_dir):
    # Create the directory
    os.makedirs(save_dir)

    print(f"Directory '{save_dir}' created successfully.")
else:
    print(f"Directory '{save_dir}' already exists.")

path2dir = f'{save_dir}/date_{formatted_datetime}'
os.makedirs(path2dir)

# Время в миллисекундах от начала суток
def current_milli_time():
    current_time = datetime.datetime.now()
    milliseconds_from_midnight = (current_time - current_time.replace(hour=0, minute=0, second=0,
                                                                      microsecond=0)).total_seconds() * 1000
    return round(milliseconds_from_midnight)



# Функция для отправки данных по UDP
def send_udp_data(data):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(data, ("192.168.1.9", 12345))
    sock.close()


def process_video_with_tracking(model, input_video_path, show_video=True, save_photos=False, save_video=False,
                                output_video_path="output_video.mp4"):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    # Get input video frame rate and dimensions
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video writer
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    counterPhoto = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        counterPhoto += 1

        results = model.track(frame, iou=0.4, conf=0.5, persist=True, imgsz=608, verbose=False,
                              tracker="bytetrack.yaml", classes=0)
        
        if save_photos:
            image_filename = f'{path2dir}/image_{counterPhoto}.bmp'
            cv2.imwrite(image_filename, frame)
            print(f"Image saved: {image_filename}")

        if results[0].boxes.id is not None:  # this will ensure that id is not None -> exist tracks
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy().astype(float)

            for box, id, conf in zip(boxes, ids, confidences):
                # Generate a random color for each object based on its ID

                random.seed(int(id))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                section1 = bytes([0xAA])  # Заголовок
                section2 = bytearray(struct.pack("I", current_milli_time() & 0xFFFFFFFF))  # Текущее время
                section3 = bytearray(struct.pack("I", counterPhoto))  # Номер снимка
                section4 = bytearray(struct.pack("h", frame_width))  # Ширина кадра с камеры
                section5 = bytearray(struct.pack("h", frame_height))  # Высота кадра с камеры
                section6 = bytearray(struct.pack("B", 2)) # Количество найденых объектов
                             
                section7 = bytearray(struct.pack("B", int(id)))  # ID объекта
                section8 = bytearray(struct.pack("f", float(conf)))  # Величина отклика трекера
                section9 = bytearray(struct.pack("h", box[0]))  # Положение левого верхнего угла X
                section10 = bytearray(struct.pack("h", box[1]))  # Положение левого верхнего угла Y
                section11 = bytearray(struct.pack("h", box[2] - box[0]))  # Ширина рамки
                section12 = bytearray(struct.pack("h", box[3] - box[1]))  # Высота рамки

                section13 = bytearray(struct.pack("B", int(id)))  # ID объекта
                section14 = bytearray(struct.pack("f", float(conf)))  # Величина отклика трекера
                section15 = bytearray(struct.pack("h", box[0]))  # Положение левого верхнего угла X
                section16 = bytearray(struct.pack("h", box[1]))  # Положение левого верхнего угла Y
                section17 = bytearray(struct.pack("h", box[2] - box[0]))  # Ширина рамки
                section18 = bytearray(struct.pack("h", box[3] - box[1]))  # Высота рамки
                section19 = bytes([0xBB]) # Конец Файла

                packed_data = (section1 + section2 + section3 + section4 + section5 + section6 +
                                section7 + section8 + section9 + section10 + section11 + section12
                                + section13 + section14 + section15 + section16 + section17 + section18
                                + section19)
                send_udp_data(packed_data)

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3],), color, 2)
                cv2.putText(
                    frame,
                    f"Id {id}",
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (0, 255, 255),
                    2,
                )
        else:
            section1 = bytes([0xAA])  #Заголовок

            section2 = bytearray(struct.pack("I", current_milli_time() & 0xFFFFFFFF))  # Текущее время
            section3 = bytearray(struct.pack("I", counterPhoto))  # Номер снимка
            section4 = bytearray(struct.pack("h", frame_width))  # Ширина кадра с камеры
            section5 = bytearray(struct.pack("h", frame_height))  # Высота кадра с камеры
            section6 = bytearray(struct.pack("B", 0)) # Количество найденых объектов
            
            section7 = bytes([0xBB]) # Конец файла
            
            packed_data = (section1 + section2 + section3 + section4 + section5 + section6 + section7)
            
            send_udp_data(packed_data)
            
        if save_video:
            out.write(frame)

        if show_video:
            frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
            cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the input video capture and output video writer
    cap.release()
    if save_video:
        out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    return results

# model_detect = YOLO('train11/weights/best.pt')
save_photo = True
input_video = 1
input_video_mov = '/media/pavel/data/techmash/datasets/nivaCar/IMG_4377.mov'
input_video_mp4 = '/media/pavel/data/techmash/projects/github/yolo/object_tracking/test_742.mp4'
#model = YOLO('model/train11/weights/best.pt')  # 'yolov5n.pt'
model = YOLO('yolov5n.pt')  # 'yolov5n.pt'
model.fuse()
# model_detect.fuse()
process_video_with_tracking(model, input_video_mp4, save_photo, save_photos=False,
                                      save_video=False,
                                      output_video_path="output_video.mp4")
