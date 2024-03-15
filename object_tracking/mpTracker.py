import cv2
import queue
import threading
import argparse

# Функция обработки кадра
def process_frame(frame, save_fragment=False):
    # Ваш код обработки кадра

    if save_fragment:
        # Определение параметров фрагмента
        x = 100
        y = 100
        width = 200
        height = 150
        fragment = frame[y:y+height, x:x+width]  # Выделение фрагмента изображения
        cv2.imwrite("fragment.jpg", fragment)  # Сохранение фрагмента в файл

# Функция для чтения команд из аргументов командной строки и установки параметра сохранения фрагментов
def command_reader(command_queue):
    while True:
        command = input("Введите команду (save_on/save_off): ")
        if command == "save_on":
            command_queue.put(True)
            print("Сохранение фрагментов включено")
        elif command == "save_off":
            command_queue.put(False)
            print("Сохранение фрагментов выключено")
        else:
            print("Неверная команда")

# Открытие видеопотока с камеры
cap = cv2.VideoCapture(1)

# Создание очереди для команд
command_queue = queue.Queue()

# Запуск потока для чтения команд
command_thread = threading.Thread(target=command_reader, args=(command_queue,))
command_thread.daemon = True
command_thread.start()

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description="Real-time camera processing with OpenCV")
parser.add_argument('--save', dest='save', action='store_true', help='Enable saving of image fragments')

args = parser.parse_args()

# Получение значения параметра сохранения фрагментов из аргументов командной строки
save_fragment = args.save

# Цикл обработки каждого кадра видеопотока
while True:
    ret, frame = cap.read()  # Получение следующего кадра

    if not ret:
        break  # Если чтение кадра не удалось, выход из цикла

    # Отображение кадра
    cv2.imshow("Frame", frame)

    # Получение параметра сохранения фрагментов из очереди
    try:
        save_fragment = command_queue.get_nowait()
    except queue.Empty:
        pass

    # Обработка кадра с возможностью сохранения фрагмента
    process_frame(frame, save_fragment=save_fragment)

    # Ожидание нажатия клавиши для выхода из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()