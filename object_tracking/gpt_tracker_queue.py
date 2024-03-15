import cv2
import numpy as np
import socket
import datetime
import struct
import argparse
import os
import queue
import threading


# Время в миллисекундах от начала суток
def current_milli_time():
    current_time = datetime.datetime.now()
    milliseconds_from_midnight = (current_time - current_time.replace(hour=0, minute=0, second=0,
                                                                      microsecond=0)).total_seconds() * 1000
    return round(milliseconds_from_midnight)

# Функция для отправки данных через UDP
# Функция для отправки данных через UDP
def send_bbox_udp(sock, bbox_data, address, frame_data):
    try:
        #print(int(round(time.time() * 1000))) 
        # Сбор данных для отправки в бинарном формате
        payload = bytearray()
        payload.append(0xAA)  # Заголовок
        payload.extend(struct.pack('<L', current_milli_time() & 0xFFFFFFFF))  # Текущее время в миллисекундах
        payload.extend(struct.pack("<I", frame_count))  # Номер снимка
        payload.extend(struct.pack('<h', frame_data['width']))  # Ширина кадра
        payload.extend(struct.pack('<h', frame_data['height']))  # Высота кадра
         
        # Количество найденных объектов
        num_objects = len(bbox_data)
        payload.append(num_objects)
        #payload.extend(struct.pack('<h', num_objects))

        # Добавление данных bbox для каждого объекта
        for bbox in bbox_data:
            if bbox is not None:
                id, response, x, y, w, h = bbox
                payload.append(id)  # ID объекта
                payload.extend(struct.pack('<f', response))  # Величина отклика
                payload.extend(struct.pack('<h', x))  # Положение левого верхнего угла рамки по горизонтали
                payload.extend(struct.pack('<h', y))  # Положение левого верхнего угла рамки по вертикали
                payload.extend(struct.pack('<h', w))  # Ширина рамки
                payload.extend(struct.pack('<h', h))  # Высота рамки

        payload.append(0xBB)  # Окончание пакета

        # Отправка данных по UDP
        sock.sendto(payload, address)
    except Exception as e:
        print("Ошибка при отправке данных:", e)

# Функция для захвата кадров, отслеживания объектов и отправки данных по UDP
def capture_frames_and_track_objects(video, udp_socket, udp_address, save_queue):
    trackers = [cv2.TrackerMIL_create() for _ in range(2)]
    init_bboxes = [(frame_width // 2 - 64, frame_height // 2 - 64, 128, 128),
                   (frame_width // 4 - 64, frame_height // 4 - 64, 128, 128)]
    
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            print("Ошибка: Не удалось прочитать кадр")
            break

        bbox_data = []
        for i, tracker in enumerate(trackers):
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = list(map(int, bbox))
                bbox_data.append((i+1, 1.0, x, y, w, h))  # Передаем ID, отклик всегда 1.0, и координаты bbox

        frame_data = {'frame_count': frame_count, 'width': frame_width, 'height': frame_height}
        send_bbox_udp(udp_socket, bbox_data, udp_address, frame_data)

        save_queue.put((frame_count, frame))
        frame_count += 1

# Функция для сохранения снимков
def save_frames(save_queue, save_dir):
    while True:
        frame_count, frame = save_queue.get()
        cv2.imwrite(os.path.join(save_dir, f"snapshot_{frame_count}.jpg"), frame)

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Сохранение снимков и передача данных через UDP')
parser.add_argument('--save_dir', type=str, default='snapshots', help='Путь до каталога для сохранения снимков')
args = parser.parse_args()

# Создание каталога для сохранения снимков
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Захват видеопотока с камеры
video = cv2.VideoCapture(1)

# Проверка успешности захвата видеопотока
if not video.isOpened():
    print("Ошибка: Не удалось открыть видеопоток")
    exit()

# Установка размера кадра
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Размер кадра
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создание UDP-сокета
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Адрес и порт для отправки данных
udp_address = ("127.0.0.1", 12345)

# Создание очереди для сохранения кадров
save_queue = queue.Queue()

# Запуск потоков
capture_thread = threading.Thread(target=capture_frames_and_track_objects, args=(video, udp_socket, udp_address, save_queue))
save_thread = threading.Thread(target=save_frames, args=(save_queue, save_dir))

capture_thread.start()
save_thread.start()

# Ожидание завершения потоков
capture_thread.join()
save_thread.join()

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()