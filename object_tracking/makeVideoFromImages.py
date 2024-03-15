import cv2
import os

def images_to_video(image_folder, output_video_path, fps):
    # Получаем список всех изображений в папке
    images = [img for img in os.listdir(image_folder) if img.endswith(".bmp")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Создаем объект VideoWriter
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Проходим по каждому изображению и добавляем его в видео
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Освобождаем ресурсы
    cv2.destroyAllWindows()
    video.release()

# Укажите путь к папке с изображениями и путь для сохранения видео
image_folder = '/home/pavel/techmash/datasets/2024-03-10/test_1923/'
output_video_path = 'test_2110.mp4'

# Укажите желаемое количество кадров в секунду (fps)
fps = 25

# Собираем видео
images_to_video(image_folder, output_video_path, fps)
