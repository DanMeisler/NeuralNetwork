import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
TRAIN_PART = 80 / 100.0


def show_picture_from_pixels(pixels):
    plt.imshow(pixels.reshape(28, 28), cmap='Greys')
    plt.show()


def save_test_prediction(output_path, test_predictions):
    with open(output_path, "wb") as output_file:
        output_file.write("\n".join(map(str, test_predictions)))


def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)


def relu(x):
    return np.maximum(x, 0)


def relu_gradient(x):
    return np.maximum(np.sign(x), 0)


def sigmoid(x):
    try:
        res = 1 / (1 + np.exp(-x))
    except:
        res = 0.0
    return res


def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))


activation = namedtuple("activation", ["function", "function_gradient"])
activation.function = staticmethod(sigmoid)
activation.function_gradient = staticmethod(sigmoid_gradient)


class NeuralNetwork(object):
    def __init__(self, classes, training_and_validation_set, learning_rate=0.1,
                 hidden_layer_size=50, num_of_epochs=100, starting_batch_size=1):
        self.batch_size = starting_batch_size
        self.num_of_epochs = num_of_epochs
        self.learning_rate = learning_rate
        self.w_input = np.random.uniform(-0.2, 0.2, (hidden_layer_size, training_and_validation_set[0][0].shape[0]))
        self.b_input = np.random.uniform(-0.2, 0.2, (hidden_layer_size, 1))
        self.w_hidden = np.random.uniform(-0.2, 0.2, (len(classes), hidden_layer_size))
        self.b_hidden = np.random.uniform(-0.2, 0.2, (len(classes), 1))
        self.params = None

    def predict(self, inputs):
        return np.argmax(self.forward_propagation(inputs)["v2"])

    def back_propagation(self, caches):
        w_input_gradient = np.zeros(self.w_input.shape)
        b_input_gradient = np.zeros(self.b_input.shape)
        w_hidden_gradient = np.zeros(self.w_hidden.shape)
        b_hidden_gradient = np.zeros(self.b_hidden.shape)

        for cache in caches:
            loss_gradient = cache["v2"]
            loss_gradient[cache["y"]] -= 1
            w_input_gradient += np.dot(
                activation.function_gradient(cache["z1"]) * np.dot(self.w_hidden.T, loss_gradient),
                cache["v0"].T)
            b_input_gradient += activation.function_gradient(cache["z1"]) * np.dot(self.w_hidden.T, loss_gradient)
            w_hidden_gradient += loss_gradient * cache["v1"].T
            b_hidden_gradient += loss_gradient

        w_input_gradient /= len(caches)
        b_input_gradient /= len(caches)
        w_hidden_gradient /= len(caches)
        b_hidden_gradient /= len(caches)

        return w_input_gradient, b_input_gradient, w_hidden_gradient, b_hidden_gradient

    def forward_propagation(self, x):
        v0 = x
        z1 = np.dot(self.w_input, v0) + self.b_input
        v1 = activation.function(z1)
        z2 = np.dot(self.w_hidden, v1) + self.b_hidden
        v2 = softmax(z2)
        return {"v0": v0, "z1": z1, "v1": v1, "z2": z2, "v2": v2}

    def update_parameters(self, w_input_gradient, b_input_gradient,
                          w_hidden_gradient, b_hidden_gradient):
        self.w_input -= self.learning_rate * w_input_gradient
        self.b_input -= self.learning_rate * b_input_gradient
        self.w_hidden -= self.learning_rate * w_hidden_gradient
        self.b_hidden -= self.learning_rate * b_hidden_gradient

    def train(self):

        training_set = training_and_validation_set[:int(TRAIN_PART * len(training_and_validation_set))]
        validation_set = training_and_validation_set[int(TRAIN_PART * len(training_and_validation_set)):]

        for epoch_number in range(self.num_of_epochs):

            np.random.shuffle(training_set)

            self.batch_size += 1

            training_correct_times_count = 0
            caches = []
            for data in training_set:
                params = self.forward_propagation(data[0])
                params["y"] = data[1]
                caches.append(params)

                if np.argmax(params["v2"]) == params["y"]:
                    training_correct_times_count += 1

                if self.batch_size == len(caches):
                    self.update_parameters(*self.back_propagation(caches))
                    caches = []

            if len(caches) != 0:
                self.update_parameters(*self.back_propagation(caches))

            validation_correct_times_count = 0
            for data in validation_set:
                if self.predict(data[0]) == data[1]:
                    validation_correct_times_count += 1

            print "Epoch number %d, batch size = %d, training accuracy = %f, validation accuracy = %f" % \
                  (epoch_number,
                   self.batch_size,
                   float(training_correct_times_count) / len(training_set),
                   float(validation_correct_times_count) / len(validation_set))


if __name__ == "__main__":
    training_and_validation_set = zip(
        np.expand_dims(np.loadtxt("resources/train_x", dtype=np.uint8), axis=2) / 255.0,
        np.loadtxt("resources/train_y", dtype=np.uint8)
    )

    np.random.shuffle(training_and_validation_set)

    neural_network = NeuralNetwork(CLASSES, training_and_validation_set)

    neural_network.train()

    x_of_test_set = np.expand_dims(np.loadtxt("resources/test_x"), axis=2) / 255.0

    test_predictions = []
    for x in xrange(len(x_of_test_set)):
        prediction_index = neural_network.predict(x_of_test_set[x])
        test_predictions.append(prediction_index)

    save_test_prediction("test.pred", test_predictions)

    print "Done!"
