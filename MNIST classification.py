import numpy as np
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from PIL import Image
import os


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
img_train = train_images / 255
img_test = test_images / 255

lbl_train = to_categorical(train_labels, 10)
lbl_test = to_categorical(test_labels, 10)

num_samples_train, img_rows, img_cols = img_train.shape
img_train = img_train.reshape(num_samples_train, img_rows, img_cols, 1)

model = Sequential([
    Conv2D(8, 3, activation='relu', input_shape = (28, 28, 1)),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if not os.path.exists("digit_model.h5"):
    print("Training model...")
    model.fit(img_train, lbl_train, epochs=8, batch_size=32)
    model.save("digit_model.h5")

    num_samples_test, _, _ = img_test.shape
    img_test = img_test.reshape(num_samples_test, img_rows, img_cols, 1)

    evaluation = model.evaluate(img_test, lbl_test)
    print(evaluation)

    predictions = model.predict(img_test[:20,:])
    predictions_idx = predictions.argmax(1)
    print(predictions_idx)
    print(test_labels[:20])

    for i in range(5):
        plt.imshow(img_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {predictions_idx[i]} | Actual: {test_labels[i]}")
        plt.axis('off')
        plt.show()

else:
    print("Loading saved model...")
    model = keras.models.load_model("digit_model.h5")

    files = [f for f in os.listdir() if f.startswith("digit") and f.endswith(".png")]

    for file in files:
        img = Image.open(file).convert('L')
        img = img.resize((28,28))

        img = np.array(img)

        display_img = img / 255

        input_img = display_img.reshape(1, 28, 28, 1)

        prediction = model.predict(input_img)

        plt.imshow(display_img, cmap='gray')
        plt.title(f"Your digit is: {prediction.argmax()}")
        plt.axis('off')
        plt.show()

    



