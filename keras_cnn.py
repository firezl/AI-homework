import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(
    (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))/255.0
x_test = x_test.reshape(
    (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))/255.0

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(
        5, 5), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=64, epochs=10)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)
