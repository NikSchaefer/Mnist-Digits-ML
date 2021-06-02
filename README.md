# Mnist Handwritten digits Machine Learning

Machine Learning Model to predict handwritten digits based on the mnist dataset of handwritten digits. Built with tensorflow and keras.

## Layers

The sequential model consists of a Conv2D layer to highlight features, Flatten, and then the model Denses down into the 10 possibe digits.

## Accuracy

The Model Acheived 98.5% Accuracy when evaluated on the test set

## Layout

data is loaded from keras and split into train and test. Data is then placed between 0 and 1 and then reshaped to fit the model.

## Data

Data is from the mnist dataset on [github](https://github.com/tensorflow/datasets)

## Installation

Install Python 3.8+

```bash
pip install tensorflow keras numpy
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
