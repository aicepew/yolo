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

# Функция для чтения команд из аргументов командной строки и установки параметра сохранения фрагментов
def command_reader(command_queue, save_flag, trackers):
    while True:
        command = input("Введите команду (save_on/save_off): ")
        if command == "save_on":
            command_queue.put(True)
            print("Сохранение фрагментов включено")
            save_flag = True
            # Если включено сохранение, инициализируем трекеры, если они не были инициализированы
            if not any(trackers):
                for i in range(len(trackers)):
                    trackers[i] = cv2.TrackerMIL_create()
                    trackers[i].init(frame, init_bboxes[i])
        elif command == "save_off":
            command_queue.put(False)
            print("Сохранение фрагментов выключено")
            save_flag = False
            # Если выключено сохранение, убеждаемся, что все трекеры пусты
            for i in range(len(trackers)):
                trackers[i] = None
        else:
            print("Неверная команда")


# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Сохранение снимков и передача данных через UDP')
parser.add_argument('--save_dir', type=str, default='snapshots', help='Путь до каталога для сохранения снимков')
parser.add_argument('--save', dest='save', action='store_true', help='Enable saving of image fragments')
args = parser.parse_args()


# Создание каталога для сохранения снимков
save_dir = args.save_dir
images_dir = os.path.join(save_dir, 'images')  # Путь до папки images
#print(save_dir)
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Захват видеопотока с камеры
video = cv2.VideoCapture(1)

# Создание очереди для команд
command_queue = queue.Queue()

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

# Переменная для подсчета кадров
frame_count = 0

# Создание UDP-сокета
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Адрес и порт для отправки данных
udp_address = ("192.168.1.9", 12345)

# Переменная для определения, включено ли сохранение снимков
save_images = args.save

# Создание трекеров MIL
trackers = [cv2.TrackerMIL_create() for _ in range(2)]

# Запуск потока для чтения команд
command_thread = threading.Thread(target=command_reader, args=(command_queue, save_images, trackers))
command_thread.daemon = True
command_thread.start()

# Инициализация трекеров после захвата первого кадра
ret, frame = video.read()
if not ret:
    print("Ошибка: Не удалось прочитать первый кадр")
    exit()

init_bboxes = [
    (frame_width // 4 - 64, frame_height // 4 - 64, 128, 128),  # Центральный объект
    (3 * frame_width // 4 - 64, 3 * frame_height // 4 - 64, 128, 128)   # Второй объект
]

# Переменная для определения, инициализировались ли трекеры
trackers_initialized = False

while True:
   
    try:
        # Получение команды из очереди с ожиданием до 1 секунды
        save_images = command_queue.get_nowait()
        #print()
        if save_images and not trackers_initialized:
            # Инициализация трекеров перед началом сохранения
            for tracker, bbox in zip(trackers, init_bboxes):
                tracker.init(frame, bbox)
            trackers_initialized = True
        else:
            trackers_initialized = False


    except queue.Empty:
        pass


    # Захват нового кадра
    ret, frame = video.read()
    if not ret:
        print("Ошибка: Не удалось прочитать кадр")
        break


    frame_count += 1
    # Сохранение снимка
    if save_images:
        cv2.imwrite(os.path.join(images_dir, f"snapshot_{frame_count}.jpg"), frame)
  
    # Отображение рамок трекеров
    if trackers_initialized:
        for tracker in trackers:
            if tracker is not None:  # Проверяем, что трекер был инициализирован
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Зеленая рамка
            else:
                success, bbox = False, None
    
    if trackers_initialized:
        for tracker, init_bbox in zip(trackers, init_bboxes):
            if tracker is not None:  # Проверяем, что трекер был инициализирован
                # Получение центра инициализационного прямоугольника
                init_center = (init_bbox[0] + init_bbox[2] // 2, init_bbox[1] + init_bbox[3] // 2)
                # Получение текущего прямоугольника
                success, bbox = tracker.update(frame)
                if success:
                    # Получение центра текущего прямоугольника
                    current_center = (int(bbox[0] + bbox[2] // 2), int(bbox[1] + bbox[3] // 2))
                    # Нарисовать линию между центром инициализации и текущим центром
                    cv2.line(frame, init_center, current_center, (255, 0, 0), 2)  # Синий цвет линии
    
   # Отображение статуса сохранения снимков
    save_status = "save_on" if save_images else "save_off"
    text_color = (0, 255, 0) if save_images else (0, 0, 255)
    text_position = (frame_width - 150, frame_height - 20)
    cv2.putText(frame, save_status, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

    # Отображение кадра
    cv2.imshow("Frame", frame)

    # Обработка кадра только если сохранение снимков включено
    if trackers_initialized:
        # Отслеживание объектов и получение bbox
        bbox_data = []
        for i, tracker in enumerate(trackers):
            if tracker is not None:  # Проверяем, что трекер был инициализирован
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = map(int, bbox)
                    bbox_data.append((i+1, 1.0, x, y, w, h))  # Передаем ID, отклик всегда 1.0, и координаты bbox
            else:
                success, bbox = False, None

        # Отправка данных bbox по UDP
        frame_data = {'width': frame_width, 'height': frame_height}
        send_bbox_udp(udp_socket, bbox_data, udp_address, frame_data)
    else:
        bbox_data = []
        frame_data = {'width': frame_width, 'height': frame_height}
        send_bbox_udp(udp_socket, bbox_data, udp_address, frame_data)

    # Выход из цикла по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()