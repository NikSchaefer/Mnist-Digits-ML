
# Mnist Handwritten digits Machine Learning

Machine Learning Model to predict handwritten digits based on the mnist dataset
of handwritten digits. Built with tensorflow and keras.

## Images into data

Data was taken from the tensorflow datasets on github so no image processing was
needed.

```py
dataset = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
```

## Model

The Model to classify vehicle images is using the Sequential model by Keras.
This model will execute the layers added to it in Sequential order to classify
the images.

```py
model = Sequential()
```

### Layers

The Layers of the model consisted of 7 layers. The first 2 layers use a Conv2d
layer to sharpen and highlight features of the images. The layer use 64 filters
with a kernel size of (3, 3). The Conv2D Layer is followed by a MaxPooling2D
Layer to bring the image back to its spatital dimensions.

The following layer is a flatten layer to prep the data for the dense layers.
The Final Dense layers dense down from 1024 to the 10 image classes(0-9 digit)
and classify the image. The final dense layer includes a softmax layer to bring
the data back to normalization.

```py
model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))
```

### Optimizer

This model uses an Adam optimizer from Keras

### Loss

This model uses categorical_crossentropy loss function to penalize the model

```py
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
```

## Accuracy

The Model was able to achieve a final test accuracy of 98.5% when evaluating the
test data.

```py
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(f"\n loss: {test_loss}\n acc: {test_acc}")
```

## Layout

the save folder contains the saved model that can be loaded.

```py
/save
model.py
```

## Data

Data is from the mnist dataset on [github](https://github.com/tensorflow/datasets)

## Installation

Install Python 3.8+

```bash
pip install tensorflow keras numpy
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
