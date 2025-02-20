import numpy as np

from activation_functions import softmax


class VanillaRNN:
    def __init__(self, n_a, n_x, n_y):
        """
        Initializes the Vanilla RNN.

        Parameters
        ----------
        n_a: (int) - number of units in the hidden state
        n_x: (int) - number of units in the input data
        n_y: (int) - number of units in the output data
        """

        self.parameters = self.initialize_parameters(n_a, n_x, n_y)
        self.a0 = np.zeros((n_a, 1))

    def initialize_parameters(self, n_a, n_x, n_y):
        """
        Initializes the parameters for the RNN.

        Parameters
        ----------
        n_a: (int) - number of units in the hidden state
        n_x: (int) - number of units in the input data
        n_y: (int) - number of units in the output data

        Returns
        -------
        parameters: (dict) - initialized parameters
            "Wax": (ndarray (n_a, n_x)) - weight matrix multiplying the input
            "Waa": (ndarray (n_a, n_a)) - weight matrix multiplying the hidden state
            "Wya": (ndarray (n_y, n_a)) - weight matrix relating the hidden-state to the output
            "ba": (ndarray (n_a, 1)) - bias
            "by": (ndarray (n_y, 1)) - bias
        """

        Wax = np.random.randn(n_a, n_x) * 0.01
        Waa = np.random.randn(n_a, n_a) * 0.01
        Wya = np.random.randn(n_y, n_a) * 0.01
        ba = np.zeros((n_a, 1))
        by = np.zeros((n_y, 1))
        parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
        return parameters

    def update_parameters(self, parameters, gradients, learning_rate):
        """
        Updates the parameters using the gradients.

        Parameters
        ----------
        parameters: (dict) - the parameters
            Waa: (ndarray (n_a, n_a)) - weight matrix multiplying the hidden state at_prev
            Wax: (ndarray (n_a, n_x)) - weight matrix multiplying the input xt
            Wya: (ndarray (n_y, n_a)) - weight matrix relating the hidden-state to the output
            ba: (ndarray (n_a, 1)) - bias
            by: (ndarray (n_y, 1)) - bias
        gradients: (dict) - the gradients
            dWaa: (ndarray (n_a, n_a)) - gradient of the weight matrix multiplying the hidden state at_prev
            dWax: (ndarray (n_a, n_x)) - gradient of the weight matrix multiplying the input xt
            dWya: (ndarray (n_y, n_a)) - gradient of the weight matrix relating the hidden-state to the output
            dba: (ndarray (n_a, 1)) - gradient of the bias
            dby: (ndarray (n_y, 1)) - gradient of the bias
        learning_rate: (float) - the learning rate
        """

        parameters["Waa"] -= learning_rate * gradients["dWaa"]
        parameters["Wax"] -= learning_rate * gradients["dWax"]
        parameters["Wya"] -= learning_rate * gradients["dWya"]
        parameters["ba"] -= learning_rate * gradients["dba"]
        parameters["by"] -= learning_rate * gradients["dby"]

    def clip(self, gradients, max_value):
        """
        Clips the gradients to a maximum value.

        Parameters
        ----------
        gradients: (dict) - the gradients
            dWaa: (ndarray (n_a, n_a)) - gradient of the weight matrix multiplying the hidden state at_prev
            dWax: (ndarray (n_a, n_x)) - gradient of the weight matrix multiplying the input xt
            dWya: (ndarray (n_y, n_a)) - gradient of the weight matrix relating the hidden-state to the output
            dba: (ndarray (n_a, 1)) - gradient of the bias
            dby: (ndarray (n_y, 1)) - gradient of the bias
        max_value: (float) - the maximum value to clip the gradients

        Returns
        -------
        gradients: (dict) - the clipped gradients
        """

        dWaa, dWax, dWya = gradients["dWaa"], gradients["dWax"], gradients["dWya"]
        dba, dby = gradients["dba"], gradients["dby"]

        for gradient in [dWax, dWaa, dWya, dba, dby]:
            np.clip(gradient, -max_value, max_value, out=gradient)

        gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}
        return gradients

    def compute_loss(self, Y_hat, Y):
        """
        Computes the cross-entropy loss.

        Parameters
        ----------
        Y_hat: (ndarray (n_y, m, T_x)) - predictions for each timestep
        Y: (ndarray (n_y, m, T_x)) - true labels

        Returns
        -------
        loss: (float) - the cross-entropy loss
        """

        return -np.sum(Y * np.log(Y_hat))

    def cell_forward(self, xt, at_prev, parameters):
        """
        Implements a single forward step of the RNN-cell.

        Parameters
        ----------
        xt: (ndarray (n_x, m)) - input data at timestep "t"
        at_prev: (ndarray (n_a, m)) - hidden state at timestep "t-1"
        parameters:
            Waa: (ndarray (n_a, n_a)) - weight matrix multiplying the hidden state at_prev
            Wax: (ndarray (n_a, n_x)) - weight matrix multiplying the input xt
            Wya: (ndarray (n_y, n_a)) - weight matrix relating the hidden-state to the output
            ba: (ndarray (n_a, 1)) - bias
            by: (ndarray (n_y, 1)) - bias

        Returns
        -------
        at: (ndarray (n_a, m)) - hidden state at timestep "t"
        yt: (ndarray (n_y, m)) - prediction at timestep "t"
        cache: (tuple) - returning (at, at_prev, xt, zxt, y_hat_t, zyt) for the backpropagation
        """

        Wax, Waa, ba = parameters["Wax"], parameters["Waa"], parameters["ba"]
        Wyx, by = parameters["Wya"], parameters["by"]

        zxt = Waa @ at_prev + Wax @ xt + ba
        at = np.tanh(zxt)
        zyt = Wyx @ at + by
        y_hat_t = softmax(zyt)

        cache = (at, at_prev, xt, zxt, y_hat_t, zyt)
        return at, y_hat_t, cache

    def forward(self, X, a0, parameters):
        """
        Implements the forward propagation of the RNN.

        Parameters
        ----------
        X: (ndarray (n_x, m, T_x)) - input data for each timestep
        a0: (ndarray (n_a, m)) - initial hidden state
        parameters:
            Waa: (ndarray (n_a, n_a)) - weight matrix multiplying the hidden state at_prev
            Wax: (ndarray (n_a, n_x)) - weight matrix multiplying the input xt
            Wyx: (ndarray (n_y, n_a)) - weight matrix relating the hidden-state to the output
            ba: (ndarray (n_a, 1)) - bias
            by: (ndarray (n_y, 1)) - bias

        Returns
        -------
        a: (ndarray (n_a, m, T_x)) - hidden states for each timestep
        Y_hat: (ndarray (n_y, m, T_x)) - predictions for each timestep
        caches: (tuple) - returning (list of cache, x) for the backpropagation
        """

        caches = []

        n_x, m, T_x = X.shape
        n_y, n_a = parameters["Wya"].shape

        A = np.zeros((n_a, m, T_x))
        Y_hat = np.zeros((n_y, m, T_x))

        at_prev = a0
        for t in range(T_x):
            at_prev, y_hat_t, cache = self.cell_forward(X[:, :, t], at_prev, parameters)
            A[:, :, t] = at_prev
            Y_hat[:, :, t] = y_hat_t
            caches.append(cache)

        return A, Y_hat, caches

    def cell_backward(self, y, dat, cache, parameters):
        """
        Implements a single backward step of the RNN-cell.

        Parameters
        ----------
        y: (ndarray (n_y, m)) - true labels at timestep "t"
        dat: (ndarray (n_a, m)) - gradient of the hidden state at timestep "t"
        cache: (tuple) - (at, at_prev, xt, zxt, y_hat_t, zyt)
        parameters:
            Waa: (ndarray (n_a, n_a)) - weight matrix multiplying the hidden state at_prev
            Wax: (ndarray (n_a, n_x)) - weight matrix multiplying the input xt
            Wya: (ndarray (n_y, n_a)) - weight matrix relating the hidden-state to the output
            ba: (ndarray (n_a, 1)) - bias
            by: (ndarray (n_y, 1)) - bias

        Returns
        -------
        gradients: (dict) - the gradients
            dWaa: (ndarray (n_a, n_a)) - gradient of the weight matrix multiplying the hidden state at_prev
            dWax: (ndarray (n_a, n_x)) - gradient of the weight matrix multiplying the input xt
            dWya: (ndarray (n_y, n_a)) - gradient of the weight matrix relating the hidden-state to the output
            dba: (ndarray (n_a, 1)) - gradient of the bias
            dby: (ndarray (n_y, 1)) - gradient of the bias
            dat: (ndarray (n_a, m)) - gradient of the hidden state
        """

        at, at_prev, xt, zt, y_hat_t, zyt = cache
        dy = y_hat_t - y
        gradients = {
            "dWya": dy @ at.T,
            "dby": np.sum(dy, axis=1, keepdims=True),
        }
        dat = parameters["Wya"].T @ dy + dat
        dz = (1 - at ** 2) * dat
        gradients["dba"] = np.sum(dz, axis=1, keepdims=True)
        gradients["dWax"] = dz @ xt.T
        gradients["dWaa"] = dz @ at_prev.T
        gradients["dat"] = parameters["Waa"].T @ dz
        return gradients

    def backward(self, X, Y, parameters, caches):
        """
        Implements the backward propagation of the RNN.

        Parameters
        ----------
        X: (ndarray (n_x, m, T_x)) - input data
        Y: (ndarray (n_y, m, T_x)) - true labels
        parameters:
            Waa: (ndarray (n_a, n_a)) - weight matrix multiplying the hidden state at_prev
            Wax: (ndarray (n_a, n_x)) - weight matrix multiplying the input xt
            Wya: (ndarray (n_y, n_a)) - weight matrix relating the hidden-state to the output
            ba: (ndarray (n_a, 1)) - bias
            by: (ndarray (n_y, 1)) - bias
        caches: (list) - list of caches from rnn_forward

        Returns
        -------
        gradients: (dict) - the gradients
            dWaa: (ndarray (n_a, n_a)) - gradient of the weight matrix multiplying the hidden state at_prev
            dWax: (ndarray (n_a, n_x)) - gradient of the weight matrix multiplying the input xt
            dWya: (ndarray (n_y, n_a)) - gradient of the weight matrix relating the hidden-state to the output
            dba: (ndarray (n_a, 1)) - gradient of the bias
            dby: (ndarray (n_y, 1)) - gradient of the bias
            dat: (ndarray (n_a, m)) - gradient of the hidden state
        """

        n_x, m, T_x = X.shape
        a1, a0, x1, zx1, y_hat1, zy1 = caches[0]
        Waa, Wax, ba = parameters['Waa'], parameters['Wax'], parameters['ba']
        Wya, by = parameters['Wya'], parameters['by']

        gradients = {
            "dWaa": np.zeros_like(Waa), "dWax": np.zeros_like(Wax), "dba": np.zeros_like(ba),
            "dWya": np.zeros_like(Wya), "dby": np.zeros_like(by),
        }

        dat = np.zeros_like(a0)
        for t in reversed(range(T_x)):
            grads = self.cell_backward(Y[:, :, t], dat, caches[t], parameters)
            gradients["dWaa"] += grads["dWaa"]
            gradients["dWax"] += grads["dWax"]
            gradients["dWya"] += grads["dWya"]
            gradients["dba"] += grads["dba"]
            gradients["dby"] += grads["dby"]
            dat = grads["dat"]

        return gradients

    def optimize(self, X, Y, a_prev, parameters, learning_rate, clip_value):
        """
        Implements the forward and backward propagation of the RNN.

        Parameters
        ----------
        X: (ndarray (n_x, m, T_x)) - the input data
        Y: (ndarray (n_y, m, T_x)) - true labels
        a_prev: (ndarray (n_a, m)) - the initial hidden state
        parameters: (dict) - the initial parameters
            Waa: (ndarray (n_a, n_a)) - weight matrix multiplying the hidden state A_prev
            Wax: (ndarray (n_a, n_x)) - weight matrix multiplying the input Xt
            Wya: (ndarray (n_y, n_a)) - weight matrix relating the hidden-state to the output
            ba: (ndarray (n_a, 1)) - bias
            by: (ndarray (n_y, 1)) - bias
        learning_rate: (float) - the learning rate

        Returns
        -------
        a: (ndarray (n_a, m)) - the hidden state at the last timestep
        loss: (float) - the cross-entropy loss
        """

        A, Y_hat, caches = self.forward(X, a_prev, parameters)
        loss = self.compute_loss(Y_hat, Y)
        gradients = self.backward(X, Y, parameters, caches)
        gradients = self.clip(gradients, clip_value)
        self.update_parameters(parameters, gradients, learning_rate)

        at = A[:, :, -1]
        return at, loss

    def preprocess_inputs(self, inputs, char_to_idx):
        """
        Preprocess the input text data.

        Parameters
        ----------
        inputs: (str) - input text data
        char_to_idx: (dict) - dictionary mapping characters to indices

        Returns
        -------
        X: (ndarray (n_x, 1, T_x)) - input data for each time step
        Y: (ndarray (n_y, 1, T_x)) - true labels for each time step
        """

        n_x = n_y = len(char_to_idx)
        T_x = T_y = len(inputs) - 1
        X = np.zeros((n_x, 1, T_x))
        Y = np.zeros((n_y, 1, T_y))
        for t in range(T_x):
            X[char_to_idx[inputs[t]], 0, t] = 1
            Y[char_to_idx[inputs[t + 1]], 0, t] = 1
        return X, Y

    def train(self, inputs, char_to_idx, num_iterations=100, learning_rate=0.01, clip_value=5):
        """
        Train the RNN model on the given text.

        Parameters
        ----------
        inputs: (str) - input text data
        char_to_idx: (dict) - dictionary mapping characters to indices
        num_iterations: (int) - number of iterations for the optimization loop
        learning_rate: (float) - learning rate for the optimization algorithm
        clip_value: (float) - maximum value for the gradients

        Returns
        -------
        losses: (list) - cross-entropy loss at each iteration
        """

        losses = []
        a_prev = self.a0

        X, Y = self.preprocess_inputs(inputs, char_to_idx)

        for i in range(num_iterations):
            a_prev, loss = self.optimize(X, Y, a_prev, self.parameters, learning_rate, clip_value)
            losses.append(loss)
            print(f"Iteration {i}, Loss: {loss}")

        return losses

    def sample(self, start_char, char_to_idx, idx_to_char, num_chars=100):
        """
        Generate text using the RNN model.

        Parameters
        ----------
        start_char: (str) - starting character for the text generation
        char_to_idx: (dict) - dictionary mapping characters to indices
        idx_to_char: (dict) - dictionary mapping indices to characters
        num_chars: (int) - number of characters to generate

        Returns
        -------
        text: (str) - generated text
        """

        x = np.zeros((len(char_to_idx), 1))
        a_prev = np.zeros_like(self.a0)

        idx = char_to_idx[start_char]
        x[idx] = 1

        indices = [idx]

        while len(indices) < num_chars:
            a_prev, y_hat, cache = self.cell_forward(x, a_prev, self.parameters)
            idx = np.random.choice(range(len(char_to_idx)), p=y_hat.ravel())
            indices.append(idx)
            x = np.zeros_like(x)
            x[idx] = 1

        text = "".join([idx_to_char[idx] for idx in indices])
        return text


if __name__ == "__main__":
    with open("shakespeare.txt", "r") as file:
        text = file.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    rnn = VanillaRNN(64, vocab_size, vocab_size)
    losses = rnn.train(text, char_to_idx, num_iterations=100, learning_rate=0.01, clip_value=5)

    generated_text = rnn.sample("T", char_to_idx, idx_to_char, num_chars=100)
    print(generated_text)
