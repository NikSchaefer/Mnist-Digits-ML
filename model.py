from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.core import Dropout
import tensorflow as tf
from keras.models import Sequential


SAVE_PATH = "save/model"

dataset = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = dataset.load_data()


train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(train_images, train_labels, epochs=20)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(f"\n loss: {test_loss}\n acc: {test_acc}")

model.save(SAVE_PATH)

"""
accuracy:
0.9850
"""
