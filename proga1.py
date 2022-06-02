from imutils import paths # нужна для работы с изображениями
import face_recognition # нужна для определения лиц
import pickle # сохранение/загрузка файлов без лишних преобразований
import cv2 # нужна для обработки изображений
import os # нужна для переключения между каталогами
 
# в директории Images хранятся папки со всеми изображениями
imagePaths = list(paths.list_images('C:\\Users\\shkos\\Desktop\\VS\\Images'))
knownEncodings = []
knownNames = []
# перебираем все папки с изображениями
for (i, imagePath) in enumerate(imagePaths):
    # извлекаем имя человека из названия папки
    name = imagePath.split(os.path.sep)[-2]
    # загружаем изображение и конвертируем его из BGR (OpenCV ordering)
    # в dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # используем библиотеку Face_recognition для обнаружения лиц
    boxes = face_recognition.face_locations(rgb,model='hog')
    # вычисляем эмбеддинги для каждого лица
    encodings = face_recognition.face_encodings(rgb, boxes)
    # цикл по encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
# сохраним эмбеддинги вместе с их именами в формате словаря
data = {"encodings": knownEncodings, "names": knownNames}
# для сохранения данных в файл используем метод pickle
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()