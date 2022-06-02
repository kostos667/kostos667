import face_recognition # нужна для определения лиц
import pickle # сохранение/загрузка файлов без лишних преобразований
import cv2 # нужна для обработки изображений
import os # нужна для переключения между каталогами
 
# находим путь к xml-файлу, содержащему каскадный файл haar
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# загружаем haarcascade в каскадный классификатор
faceCascade = cv2.CascadeClassifier(cascPathface)
# загружаем известные грани и вложения, сохраненные в последнем файле
data = pickle.loads(open('face_enc', "rb").read())
 
print("Streaming started")
video_capture = cv2.VideoCapture(0)
# цикл по кадрам из потока видеофайлов
while True:
    # захватываем кадр из потокового видеопотока
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
 
    # преобразуем входной кадр из BGR в RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # данные лица для лица во входных данных
    encodings = face_recognition.face_encodings(rgb)
    names = []
    for encoding in encodings:
       # Сравниваем encodings с encodings в данных data["encodings"]
       # Совпадения содержат массив с логическими значениями True для вложений, которым он точно соответствует
       # и False для остальных
        matches = face_recognition.compare_faces(data["encodings"],
         encoding)
        # имя - неизвестно, если кодировка не совпадает
        name = "Unknown"
        # проверка на совпадения
        if True in matches:
            #Находим позиции, в которых мы получаем True, и сохраняем их
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                # Проверяем имена по соответствующим индексам, которые мы сохранили в matchedIdxs
                name = data["names"][i]
                # увеличиваем count для имени, которое мы получили
                counts[name] = counts.get(name, 0) + 1
            # устанавливаем имя, которое имеет наибольший count
            name = max(counts, key=counts.get)
 
 
        # обновляем список имен
        names.append(name)
        # цикл по распознанным лицам
        for ((x, y, w, h), name) in zip(faces, names):
            # изменение масштаба координат лица
            # пишем предсказанное имя лица на изображении
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()