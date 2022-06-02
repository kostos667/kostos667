from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
currentname = "unknown"
#Определение граней из модели файла face_enc, созданной в обучающей программе
encodingsP = "face_enc"
# загружаем известные грани и эмбеддинги вместе с Haar в OpenCV
# каскад для распознавания лиц
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
# инициализируем видеопоток
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
# запускаем счетчик кадров в секунду
fps = FPS().start()
# цикл по кадрам из потока видеофайлов
while True:
# возьмем кадр из потокового видеопотока и изменим его размер
# до 500 пикселей (для ускорения обработки)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	# обнаружение лиц
	boxes = face_recognition.face_locations(frame)
	# вычисление эмбеддингов лиц для каждого ограничивающего прямоугольника лица
	encodings = face_recognition.face_encodings(frame, boxes)
	names = []
	for encoding in encodings:
		# попытка сопоставить каждое лицо на входном изображении с нашим известным
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown" #если лицо не распознано, то выведем: Неизвестное
		# проверка на совпадение
		if True in matches:
			# находим индексы всех совпадающих граней
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			# перебираем сопоставленные индексы
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
# определяем распознанное лицо с наибольшим числом
# количества голосов (примечание: в случае маловероятного равенства голосов
# будет выбрана первая запись в словаре)
			name = max(counts, key=counts.get)
			# если кто-то совпал - выводим имя
			if currentname != name:
				currentname = name
				print(currentname)
		names.append(name)
	# цикл по распознанным лицам
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# рисуем предсказанное имя лица на изображении
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)
	# выводим изображение на наш экран
	cv2.imshow("Facial Recognition is Running", frame)
	key = cv2.waitKey(1) & 0xFF
	# выход при нажатии клавиши "q"
	if key == ord("q"):
		break
	# обновляем счетчик кадров в секунду
	fps.update()
# останавливаем таймер и отображаем информацию о кадрах в секунду
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# очищаем камеру
cv2.destroyAllWindows()
vs.stop()