import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Check for GPU
print("GPU", "available (YESS!!!!)" if tf.config.list_physical_devices("GPU") else "not available :(")
tf.config.list_physical_devices("GPU")


# Ścieżki do folderów z obrazami i adnotacjami
images_folder = r"D:\Pojemnik danych\Repositories\Pycharm\Sieci-neuronowe\images\Images"
annotations_folder = r"D:\Pojemnik danych\Repositories\Pycharm\Sieci-neuronowe\annotations"



# Funkcja do wczytywania adnotacji
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    class_name = root.find('object/name').text
    xmin = int(root.find('object/bndbox/xmin').text)
    ymin = int(root.find('object/bndbox/ymin').text)
    xmax = int(root.find('object/bndbox/xmax').text)
    ymax = int(root.find('object/bndbox/ymax').text)

    return filename, class_name, xmin, ymin, xmax, ymax


# Wczytaj obrazy
data = []

for breed_folder in os.listdir(images_folder):
    breed_path = os.path.join(images_folder, breed_folder)

    # Sprawdź, czy to jest folder (wykluczmy pliki)
    if os.path.isdir(breed_path):
        breed_code, _, breed_name = breed_folder.partition('-')

        for img_filename in os.listdir(breed_path):
            img_path = os.path.join(breed_path, img_filename)

            # Tutaj możesz dodać kod do wczytywania obrazu i przetwarzania go, jeśli to konieczne

            data.append((img_path, breed_code, breed_name))

# Przykład wydrukowania kilku informacji
for img_path, breed_code, breed_name in data[:5]:
    print(f"Image: {img_path}, Breed Code: {breed_code}, Breed Name: {breed_name}")

# Rozpakuj dane
images_paths, breed_codes, breed_names = zip(*data)

# Przygotuj generator treningowy
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    images_folder,
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse'
)

# Przygotuj generator walidacyjny
val_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_generator = val_datagen.flow_from_directory(
    images_folder,
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse'
)

# Wczytaj obrazy do listy
images = []
for img_path in images_paths:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    images.append(img)

    # Jeśli masz problem z zasobami, możesz spróbować zwolnić pamięć po każdym obrazie
    del img

# Przekształć listę obrazów na tablicę NumPy
images = np.stack(images)

# Zbuduj model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(120, activation='softmax'))

# Kompiluj model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trenuj model
model.fit(train_generator, epochs=30)