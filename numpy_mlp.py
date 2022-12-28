import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class MLP:

    def __init__(self, hidden_units, minibatch_size, regularization_rate, learning_rate):
        self.hidden_units = hidden_units
        self.minibatch_size = minibatch_size
        self.regularization_rate = regularization_rate
        self.learning_rate = learning_rate

    def relu_function(self, matrix_content, matrix_dim_x, matrix_dim_y):
        ret_vector = np.zeros((matrix_dim_x, matrix_dim_y))

        for i in range(matrix_dim_x):
            for j in range(matrix_dim_y):
                ret_vector[i, j] = max(0, matrix_content[i, j])

        return ret_vector

    def grad_relu(self, matrix_content, matrix_dim_x, matrix_dim_y):
        ret_vector = np.zeros((matrix_dim_x, matrix_dim_y))

        for i in range(matrix_dim_x):
            for j in range(matrix_dim_y):
                if matrix_content[i, j] > 0:
                    ret_vector[i, j] = 1
                else:
                    ret_vector[i, j] = 0

        return ret_vector

    def softmax_function(self, vector_content):
        return np.exp(vector_content - np.max(vector_content)) / np.sum(np.exp(vector_content - np.max(vector_content)), axis=0)

    def iterate_minibatches(self, inputs, targets, batch_size, shuffle=False):

        assert inputs.shape[0] == targets.shape[0]

        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)

        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            yield inputs[excerpt], targets[excerpt]

    def train(self, trainX, trainY, epochs):

        w1_mat = np.random.randn(self.hidden_units, 28*28) * \
            np.sqrt(2./(self.hidden_units+28*28))
        w2_mat = np.random.randn(10, self.hidden_units) * \
            np.sqrt(2./(10+self.hidden_units))
        b1_vec = np.zeros((self.hidden_units, 1))
        b2_vec = np.zeros((10, 1))

        trainX = np.reshape(trainX, (trainX.shape[0], 28*28))
        trainY = np.reshape(trainY, (trainY.shape[0], 1))

        for num_epochs in tqdm(range(epochs)):

            for batch in self.iterate_minibatches(trainX, trainY, self.minibatch_size, shuffle=True):
                x_batch, y_batch = batch
                x_batch = x_batch.T
                y_batch = y_batch.T

                z1 = np.dot(w1_mat, x_batch) + b1_vec
                a1 = self.relu_function(
                    z1, self.hidden_units, self.minibatch_size)

                z2 = np.dot(w2_mat, a1) + b2_vec
                a2_softmax = self.softmax_function(z2)

                gt_vector = np.zeros((10, self.minibatch_size))
                for example_num in range(self.minibatch_size):
                    gt_vector[y_batch[0, example_num], example_num] = 1

                d_w2_mat = self.regularization_rate*w2_mat
                d_w1_mat = self.regularization_rate*w1_mat

                delta_2 = np.array(a2_softmax - gt_vector)
                d_w2_mat = d_w2_mat + np.dot(delta_2, (np.matrix(a1)).T)
                d_b2_vec = np.sum(delta_2, axis=1, keepdims=True)

                delta_1 = np.array(np.multiply((np.dot(w2_mat.T, delta_2)), self.grad_relu(
                    z1, self.hidden_units, self.minibatch_size)))
                d_w1_mat = d_w1_mat + np.dot(delta_1, np.matrix(x_batch).T)
                d_b1_vec = np.sum(delta_1, axis=1, keepdims=True)

                d_w2_mat = d_w2_mat/self.minibatch_size
                d_w1_mat = d_w1_mat/self.minibatch_size
                d_b2_vec = d_b2_vec/self.minibatch_size
                d_b1_vec = d_b1_vec/self.minibatch_size

                w2_mat = w2_mat - self.learning_rate*d_w2_mat
                b2_vec = b2_vec - self.learning_rate*d_b2_vec

                w1_mat = w1_mat - self.learning_rate*d_w1_mat
                b1_vec = b1_vec - self.learning_rate*d_b1_vec

        self.w1_mat, self.b1_vec, self.w2_mat, self.b2_vec = w1_mat, b1_vec, w2_mat, b2_vec

    def test(self, testX):
        output_labels = np.zeros(testX.shape[0])

        num_examples = testX.shape[0]

        testX = np.reshape(testX, (num_examples, 28*28))
        testX = testX.T

        z1 = np.dot(self.w1_mat, testX) + self.b1_vec
        a1 = self.relu_function(z1, self.hidden_units, num_examples)

        z2 = np.dot(self.w2_mat, a1) + self.b2_vec
        a2_softmax = self.softmax_function(z2)

        for i in range(num_examples):
            pred_col = a2_softmax[:, [i]]
            output_labels[i] = np.argmax(pred_col)

        return output_labels


def main(units, lr, mnist_data):

    labels = np.array(list(map(lambda x: int(x), mnist_data['target'])))
    images = np.array(mnist_data['data'])/255.0

    x_train, x_test,  y_train, y_test = train_test_split(
        images, labels, test_size=0.2)

    epochs = 25
    num_hidden_units = units
    minibatch_size = 128
    regularization_rate = 0.1
    learning_rate = lr

    print(num_hidden_units, learning_rate)

    mlp = MLP(num_hidden_units, minibatch_size,
              regularization_rate, learning_rate)

    mlp.train(x_train, y_train, epochs)

    labels = mlp.test(x_test)
    print("Error number: %s" % str(np.sum(labels != y_test)))
    accuracy = np.mean((labels == y_test)) * 100.0
    print("Test accuracy: %lf%%" % accuracy)


if __name__ == "__main__":
    units = [500, 1000, 1500, 2000]
    lrs = [0.1, 0.01, 0.001, 0.0001]
    mnist_data = fetch_openml("mnist_784")
    for unit in units:
        for lr in lrs:
            print("units: %d, lr: %d" % (unit, lr))
            main(unit, lr, mnist_data)
