from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical


network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# verify the shapes of the training and test data sets
print('training_images.shape: {}\nlen(training_images): {}'.format(training_images.shape, len(training_images)))
print('test_images.shape: {}\nlen(test_images): {}'.format(test_images.shape, len(test_images)))

training_images_shape = training_images.shape
test_images_shape = test_images.shape

training_images = training_images.reshape(training_images_shape[0], training_images_shape[1] * training_images_shape[2])
training_images = training_images.astype('float32') / 255

test_images = test_images.reshape(test_images_shape[0], test_images_shape[1] * test_images_shape[2])
test_images = test_images.astype('float32') / 255

training_labels = to_categorical(training_labels)
test_labels = to_categorical(test_labels)

network.fit(training_images, training_labels, epochs=5, batch_size=128)

test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print('test_loss: {}\ntest_accuracy: {} %'.format(test_loss, 100*test_accuracy))
