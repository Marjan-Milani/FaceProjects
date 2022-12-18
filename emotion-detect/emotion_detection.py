from fer import FER
import matplotlib.pyplot as plt
img = plt.imread("pic3.jpeg")
detector = FER(mtcnn=True)
print(detector.detect_emotions(img))
plt.imshow(img)
emotion, score = detector.top_emotion(img)
print(emotion, score)
