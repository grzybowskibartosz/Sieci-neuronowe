import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Załaduj model (jeśli tego jeszcze nie zrobiłeś)
model_path = "pieski.h5"  # Zmień na odpowiednią ścieżkę
loaded_model = tf.keras.models.load_model(model_path)


# Przetestuj model na pojedynczym obrazie (przykładowo)
# W tym przykładzie zakładam, że masz pojedynczy obraz testowy o nazwie test_image

test_image = r'D:\Pojemnik danych\Repositories\Pycharm\Sieci-neuronowe\images\Images\n02106662-German_shepherd\n02106662_662.jpg'
test_image= cv2.imread(test_image)
test_image = cv2.resize(test_image, (150, 150))  # Przygotuj obraz
test_image = np.expand_dims(test_image, axis=0)/255  # Dodaj wymiar wszechstronny (batch_size=1)

# Uzyskaj prognozę modelu na jednym obrazie
predictions = loaded_model.predict(test_image)

predicted_index = np.argmax(predictions)

# Wyświetl prognozy
print("Predictions:", predicted_index, predictions[0, predicted_index])
