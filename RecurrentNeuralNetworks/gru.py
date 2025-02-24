import numpy as np

from activation_functions import sigmoid, softmax, tanh


class GRU:
    def __init__(self, n_a, n_x, n_y):
        """
        Initializes the LSTM network.

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
        Initializes the weights and biases of the GRU network.

        Parameters
        ----------
        n_a: (int) - number of units in the hidden state
        n_x: (int) - number of units in the input data
        n_y: (int) - number of units in the output data

        Returns
        -------
        parameters: (dict) - dictionary containing the weights and biases of the GRU network
            Wu: (ndarray (n_a, n_a + n_x)) - weights of the update gate
            bu: (ndarray (n_a, 1)) - biases of the update gate
            Wr: (ndarray (n_a, n_a + n_x)) - weights of the reset gate
            br: (ndarray (n_a, 1)) - biases of the reset gate
            Wc: (ndarray (n_a, n_a + n_x)) - weights of the candidate value
            bc: (ndarray (n_a, 1)) - biases of the candidate value
            Wy: (ndarray (n_y, n_a)) - weights of the output layer
            by: (ndarray (n_y, 1)) - biases of the output layer
        """

        Wu = np.random.randn(n_a, n_a + n_x) * 0.01
        Wr = np.random.randn(n_a, n_a + n_x) * 0.01
        Wc = np.random.randn(n_a, n_a + n_x) * 0.01
        Wy = np.random.randn(n_y, n_a) * 0.01
        bu = np.zeros((n_a, 1))
        br = np.zeros((n_a, 1))
        bc = np.zeros((n_a, 1))
        by = np.zeros((n_y, 1))
        parameters = {'Wu': Wu, 'Wr': Wr, 'Wc': Wc, 'Wy': Wy, 'bu': bu, 'br': br, 'bc': bc, 'by': by}
        return parameters

    def update_parameters(self, parameters, gradients, learning_rate):
        """
        Updates the weights and biases of the GRU network.

        Parameters
        ----------
        parameters: (dict) - dictionary containing the weights and biases of the GRU network
            Wu: (ndarray (n_a, n_a + n_x)) - weights of the update gate
            bu: (ndarray (n_a, 1)) - biases of the update gate
            Wr: (ndarray (n_a, n_a + n_x)) - weights of the reset gate
            br: (ndarray (n_a, 1)) - biases of the reset gate
            Wc: (ndarray (n_a, n_a + n_x)) - weights of the candidate value
            bc: (ndarray (n_a, 1)) - biases of the candidate value
            Wy: (ndarray (n_y, n_a)) - weights of the output layer
            by: (ndarray (n_y, 1)) - biases of the output layer
        gradients: (dict) - dictionary containing the gradients of the weights and biases of the GRU network
            dWu: (ndarray (n_a, n_a + n_x)) - gradients of the weights of the update gate
            dbu: (ndarray (n_a, 1)) - gradients of the biases of the update gate
            dWr: (ndarray (n_a, n_a + n_x)) - gradients of the weights of the reset gate
            dbr: (ndarray (n_a, 1)) - gradients of the biases of the reset gate
            dWc: (ndarray (n_a, n_a + n_x)) - gradients of the weights of the candidate value
            dbc: (ndarray (n_a, 1)) - gradients of the biases of the candidate value
            dWy: (ndarray (n_y, n_a)) - gradients of the weights of the output layer
            dby: (ndarray (n_y, 1)) - gradients of the biases of the output layer
        learning_rate: (float) - learning rate
        """

        parameters["Wu"] -= learning_rate * gradients["dWu"]
        parameters["bu"] -= learning_rate * gradients["dbu"]
        parameters["Wr"] -= learning_rate * gradients["dWr"]
        parameters["br"] -= learning_rate * gradients["dbr"]
        parameters["Wc"] -= learning_rate * gradients["dWc"]
        parameters["bc"] -= learning_rate * gradients["dbc"]
        parameters["Wy"] -= learning_rate * gradients["dWy"]
        parameters["by"] -= learning_rate * gradients["dby"]

    def clip(self, gradients, max_value):
        """
        Clips the gradients to prevent exploding gradients.

        Parameters
        ----------
        gradients: (dict) - dictionary containing the gradients of the weights and biases of the GRU network
            dWu: (ndarray (n_a, n_a + n_x)) - gradients of the weights of the update gate
            dbu: (ndarray (n_a, 1)) - gradients of the biases of the update gate
            dWr: (ndarray (n_a, n_a + n_x)) - gradients of the weights of the reset gate
            dbr: (ndarray (n_a, 1)) - gradients of the biases of the reset gate
            dWc: (ndarray (n_a, n_a + n_x)) - gradients of the weights of the candidate value
            dbc: (ndarray (n_a, 1)) - gradients of the biases of the candidate value
            dWy: (ndarray (n_y, n_a)) - gradients of the weights of the output layer
            dby: (ndarray (n_y, 1)) - gradients of the biases of the output layer
        max_value: (float) - maximum value to clip the gradients

        Returns
        -------
        gradients: (dict) - the clipped gradients
        """

        dWu, dbu = gradients["dWu"], gradients["dbu"]
        dWr, dbr = gradients["dWr"], gradients["dbr"]
        dWc, dbc = gradients["dWc"], gradients["dbc"]
        dWy, dby = gradients["dWy"], gradients["dby"]

        for gradient in [dWu, dbu, dWr, dbr, dWc, dbc, dWy, dby]:
            np.clip(gradient, -max_value, max_value, out=gradient)

        gradients = {"dWu": dWu, "dbu": dbu, "dWr": dWr, "dbr": dbr, "dWc": dWc, "dbc": dbc, "dWy": dWy, "dby": dby}
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
        Implements a single forward step for the GRU-cell.

        Parameters
        ----------
        xt: (ndarray (n_x, m)) - input data for the current timestep
        at_prev: (ndarray (n_a, m)) - hidden state from the previous timestep
        parameters: (dict) - dictionary containing the weights and biases of the GRU network
            Wu: (ndarray (n_a, n_a + n_x)) - weights of the update gate
            bu: (ndarray (n_a, 1)) - biases of the update gate
            Wr: (ndarray (n_a, n_a + n_x)) - weights of the reset gate
            br: (ndarray (n_a, 1)) - biases of the reset gate
            Wc: (ndarray (n_a, n_a + n_x)) - weights of the candidate value
            bc: (ndarray (n_a, 1)) - biases of the candidate value
            Wy: (ndarray (n_y, n_a)) - weights of the output layer
            by: (ndarray (n_y, 1)) - biases of the output layer

        Returns
        -------
        at: (ndarray (n_a, m)) - hidden state for the current timestep
        y_hat_t: (ndarray (n_y, m)) - prediction for the current timestep
        cache: (tuple) - values needed for the backward pass
        """

        Wu, bu = parameters["Wu"], parameters["bu"]  # update gate weights and biases
        Wr, br = parameters["Wr"], parameters["br"]  # reset gate weights and biases
        Wc, bc = parameters["Wc"], parameters["bc"]  # candidate value weights and biases
        Wy, by = parameters["Wy"], parameters["by"]  # prediction weights and biases

        concat = np.concatenate((at_prev, xt), axis=0)

        ut = sigmoid(Wu @ concat + bu)  # update gate
        rt = sigmoid(Wr @ concat + br)  # reset gate
        cct = tanh(Wc @ np.concatenate((rt * at_prev, xt), axis=0) + bc)  # candidate value
        at = ut * cct + (1 - ut) * at_prev  # hidden state

        zyt = Wy @ at + by
        y_hat_t = softmax(zyt)
        cache = (at, at_prev, ut, rt, cct, xt, y_hat_t, zyt)
        return at, y_hat_t, cache

    def forward(self, X, a0, parameters):
        """
        Implements the forward pass of the GRU network.

        Parameters
        ----------
        X: (ndarray (n_x, m, T_x)) - input data for each time step
        a0: (ndarray (n_a, m)) - initial hidden state
        parameters: (dict) - dictionary containing the weights and biases of the GRU network
            Wu: (ndarray (n_a, n_a + n_x)) - weights of the update gate
            bu: (ndarray (n_a, 1)) - biases of the update gate
            Wr: (ndarray (n_a, n_a + n_x)) - weights of the reset gate
            br: (ndarray (n_a, 1)) - biases of the reset gate
            Wc: (ndarray (n_a, n_a + n_x)) - weights of the candidate value
            bc: (ndarray (n_a, 1)) - biases of the candidate value
            Wy: (ndarray (n_y, n_a)) - weights of the output layer
            by: (ndarray (n_y, 1)) - biases of the output layer

        Returns
        -------
        A: (ndarray (n_a, m, T_x)) - hidden states for each timestep
        Y_hat: (ndarray (n_y, m, T_x)) - predictions for each timestep
        caches: (list) - values needed for the backward pass
        """

        caches = []

        Wy = parameters["Wy"]
        x_x, m, T_x = X.shape
        n_y, n_a = Wy.shape

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
        Implements a single backward step for the GRU-cell.

        Parameters
        ----------
        y: (ndarray (n_y, m)) - true labels for the current timestep
        dat: (ndarray (n_a, m)) - gradient of the hidden state for the current timestep
        cache: (tuple) - values needed for the backward pass
        parameters: (dict) - dictionary containing the weights and biases of the GRU network
            Wu: (ndarray (n_a, n_a + n_x)) - weights of the update gate
            bu: (ndarray (n_a, 1)) - biases of the update gate
            Wr: (ndarray (n_a, n_a + n_x)) - weights of the reset gate
            br: (ndarray (n_a, 1)) - biases of the reset gate
            Wc: (ndarray (n_a, n_a + n_x)) - weights of the candidate value
            bc: (ndarray (n_a, 1)) - biases of the candidate value
            Wy: (ndarray (n_y, n_a)) - weights of the output layer
            by: (ndarray (n_y, 1)) - biases of the output layer

        Returns
        -------
        gradients: (dict) - dictionary containing the gradients of the weights and biases of the GRU network
            dWu: (ndarray (n_a, n_a + n_x)) - gradients of the weights of the update gate
            dbu: (ndarray (n_a, 1)) - gradients of the biases of the update gate
            dWr: (ndarray (n_a, n_a + n_x)) - gradients of the weights of the reset gate
            dbr: (ndarray (n_a, 1)) - gradients of the biases of the reset gate
            dWc: (ndarray (n_a, n_a + n_x)) - gradients of the weights of the candidate value
            dbc: (ndarray (n_a, 1)) - gradients of the biases of the candidate value
            dWy: (ndarray (n_y, n_a)) - gradients of the weights of the output layer
            dby: (ndarray (n_y, 1)) - gradients of the biases of the output
        """

        at, at_prev, ut, rt, cct, xt, y_hat_t, zyt = cache
        n_a, m = at.shape

        dzy = y_hat_t - y
        dWy = dzy @ at.T
        dby = np.sum(dzy, axis=1, keepdims=True)

        dat = parameters["Wy"].T @ dzy + dat

        dcct = dat * ut * (1 - cct ** 2)  # dn_t
        dut = dat * (cct - at_prev) * ut * (1 - ut)
        dat_prev = dat * (1 - ut)

        dcct_ra_x = parameters["Wc"].T @ dcct
        dcct_r_at_prev = dcct_ra_x[:n_a, :]
        dcct_xt = dcct_ra_x[n_a:, :]
        drt = (dcct_r_at_prev * at_prev) * rt * (1 - rt)

        concat = np.concatenate((at_prev, xt), axis=0)

        dWc = dcct @ np.concatenate((rt * at_prev, xt), axis=0).T
        dbc = np.sum(dcct, axis=1, keepdims=True)
        dWr = drt @ concat.T
        dbr = np.sum(drt, axis=1, keepdims=True)
        dWu = dut @ concat.T
        dbu = np.sum(dut, axis=1, keepdims=True)

        dat_prev = (
            dat_prev + dcct_r_at_prev * rt + parameters["Wr"][:, :n_a].T @ drt + parameters["Wu"][:, :n_a].T @ dut
        )
        dxt = (
            dcct_xt + parameters["Wr"][:, n_a:].T @ drt + parameters["Wu"][:, n_a:].T @ dut
        )

        gradients = {
            "dWu": dWu, "dbu": dbu, "dWr": dWr, "dbr": dbr, "dWc": dWc, "dbc": dbc, "dWy": dWy, "dby": dby,
            "dat_prev": dat_prev, "dxt": dxt
        }
        return gradients

    def backward(self, X, Y, parameters, caches):
        """
        Implements the backward pass of the GRU network.

        Parameters
        ----------
        X: (ndarray (n_x, m, T_x)) - input data for each time step
        Y: (ndarray (n_y, m, T_x)) - true labels for each time step
        parameters: (dict) - dictionary containing the weights and biases of the GRU network
            Wu: (ndarray (n_a, n_a + n_x)) - weights of the update gate
            bu: (ndarray (n_a, 1)) - biases of the update gate
            Wr: (ndarray (n_a, n_a + n_x)) - weights of the reset gate
            br: (ndarray (n_a, 1)) - biases of the reset gate
            Wc: (ndarray (n_a, n_a + n_x)) - weights of the candidate value
            bc: (ndarray (n_a, 1)) - biases of the candidate value
            Wy: (ndarray (n_y, n_a)) - weights of the output layer
            by: (ndarray (n_y, 1)) - biases of the output layer
        caches: (list) - values needed for the backward pass

        Returns
        -------
        gradients: (dict) - dictionary containing the gradients of the weights and biases of the GRU network
            dWu: (ndarray (n_a, n_a + n_x)) - gradients of the weights of the update gate
            dbu: (ndarray (n_a, 1)) - gradients of the biases of the update gate
            dWr: (ndarray (n_a, n_a + n_x)) - gradients of the weights of the reset gate
            dbr: (ndarray (n_a, 1)) - gradients of the biases of the reset gate
            dWc: (ndarray (n_a, n_a + n_x)) - gradients of the weights of the candidate value
            dbc: (ndarray (n_a, 1)) - gradients of the biases of the candidate value
            dWy: (ndarray (n_y, n_a)) - gradients of the weights of the output layer
            dby: (ndarray (n_y, 1)) - gradients of the biases of the output layer
        """

        n_x, m, T_x = X.shape
        a1, a0, u0, r1, cc1, x1, y_hat_1, zyt1 = caches[0]
        Wu, Wr, Wc, Wy = parameters["Wu"], parameters["Wr"], parameters["Wc"], parameters["Wy"]
        bu, br, bc, by = parameters["bu"], parameters["br"], parameters["bc"], parameters["by"]

        gradients = {
            "dWu": np.zeros_like(Wu), "dbu": np.zeros_like(bu), "dWr": np.zeros_like(Wr), "dbr": np.zeros_like(br),
            "dWc": np.zeros_like(Wc), "dbc": np.zeros_like(bc), "dWy": np.zeros_like(Wy), "dby": np.zeros_like(by),
        }

        dat = np.zeros_like(a0)
        for t in reversed(range(T_x)):
            grads = self.cell_backward(Y[:, :, t], dat, caches[t], parameters)
            gradients["dWu"] += grads["dWu"]
            gradients["dbu"] += grads["dbu"]
            gradients["dWr"] += grads["dWr"]
            gradients["dbr"] += grads["dbr"]
            gradients["dWc"] += grads["dWc"]
            gradients["dbc"] += grads["dbc"]
            gradients["dWy"] += grads["dWy"]
            gradients["dby"] += grads["dby"]
            dat = grads["dat_prev"]

        return gradients

    def optimize(self, X, Y, a_prev, parameters, learning_rate, clip_value):
        """
        Implements the forward and backward pass of the GRU network.

        Parameters
        ----------
        X: (ndarray (n_x, m, T_x)) - input data for each time step
        Y: (ndarray (n_y, m, T_x)) - true labels for each time step
        a_prev: (ndarray (n_a, m)) - initial hidden state
        parameters: (dict) - dictionary containing the weights and biases of the GRU network
            Wu: (ndarray (n_a, n_a + n_x)) - weights of the update gate
            bu: (ndarray (n_a, 1)) - biases of the update gate
            Wr: (ndarray (n_a, n_a + n_x)) - weights of the reset gate
            br: (ndarray (n_a, 1)) - biases of the reset gate
            Wc: (ndarray (n_a, n_a + n_x)) - weights of the candidate value
            bc: (ndarray (n_a, 1)) - biases of the candidate value
            Wy: (ndarray (n_y, n_a)) - weights of the output layer
            by: (ndarray (n_y, 1)) - biases of the output layer
        learning_rate: (float) - learning rate
        clip_value: (float) - maximum value to clip the gradients

        Returns
        -------
        at: (ndarray (n_a, m)) hidden state for the last time step
        loss: (float) - the cross-entropy
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
        Train the GRU model on the given text.

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
        Generate text using the GRU model.

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
        c_prev = np.zeros_like(self.a0)

        idx = char_to_idx[start_char]
        x[idx] = 1

        indices = [idx]

        while len(indices) < num_chars:
            a_prev, c_prev, y_hat, cache = self.cell_forward(x, a_prev, c_prev, self.parameters)
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

    gru = GRU(64, vocab_size, vocab_size)
    losses = gru.train(text, char_to_idx, num_iterations=100, learning_rate=0.01, clip_value=5)

    generated_text = gru.sample("T", char_to_idx, idx_to_char, num_chars=100)
    print(generated_text)
