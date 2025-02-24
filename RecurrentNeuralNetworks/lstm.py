import numpy as np

from activation_functions import sigmoid, softmax, tanh


class LSTM:
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
        Initializes parameters for the LSTM network.

        Parameters
        ----------
        n_a: (int) - number of units in the hidden state
        n_x: (int) - number of units in the input data
        n_y: (int) - number of units in the output data

        Returns
        -------
        parameters: (dict) - dictionary containing the following parameters:
            Wf: (ndarray (n_a, n_a + n_x)) - weight matrix of the forget gate
            bf: (ndarray (n_a, 1)) - bias of the forget gate
            Wi: (ndarray (n_a, n_a + n_x)) - weight matrix of the input gate
            bi: (ndarray (n_a, 1)) - bias of the input gate
            Wc: (ndarray (n_a, n_a + n_x)) - weight matrix of the candidate value
            bc: (ndarray (n_a, 1)) - bias of the candidate value
            Wo: (ndarray (n_a, n_a + n_x)) - weight matrix of the output gate
            bo: (ndarray (n_a, 1)) - bias of the output gate
            Wy: (ndarray (n_y, n_a)) - weight matrix relating the hidden-state to the output
            by: (ndarray (n_y, 1)) - bias relating the hidden-state to the
        """

        Wf = np.random.randn(n_a, n_a + n_x) * 0.01
        Wi = np.random.randn(n_a, n_a + n_x) * 0.01
        Wc = np.random.randn(n_a, n_a + n_x) * 0.01
        Wo = np.random.randn(n_a, n_a + n_x) * 0.01
        Wy = np.random.randn(n_y, n_a) * 0.01
        bf = np.zeros((n_a, 1))
        bi = np.zeros((n_a, 1))
        bc = np.zeros((n_a, 1))
        bo = np.zeros((n_a, 1))
        by = np.zeros((n_y, 1))
        parameters = {
            "Wf": Wf, "Wi": Wi, "Wc": Wc, "Wo": Wo, "Wy": Wy, "bf": bf, "bi": bi, "bc": bc, "bo": bo, "by": by
        }
        return parameters

    def update_parameters(self, parameters, gradients, learning_rate):
        """
        Updates the parameters of the LSTM network using gradient descent.

        Parameters
        ----------
        parameters: (dict) - dictionary containing the following parameters:
            Wf: (ndarray (n_a, n_a + n_x)) - weight matrix of the forget gate
            bf: (ndarray (n_a, 1)) - bias of the forget gate
            Wi: (ndarray (n_a, n_a + n_x)) - weight matrix of the input gate
            bi: (ndarray (n_a, 1)) - bias of the input gate
            Wc: (ndarray (n_a, n_a + n_x)) - weight matrix of the candidate value
            bc: (ndarray (n_a, 1)) - bias of the candidate value
            Wo: (ndarray (n_a, n_a + n_x)) - weight matrix of the output gate
            bo: (ndarray (n_a, 1)) - bias of the output gate
            Wy: (ndarray (n_y, n_a)) - weight matrix relating the hidden-state to the output
            by: (ndarray (n_y, 1)) - bias relating the hidden-state to the output
        gradients: (dict) - dictionary containing the following gradients:
            dWf: (ndarray (n_a, n_a + n_x)) - gradient of the forget gate weight
            dbf: (ndarray (n_a, 1)) - gradient of the forget gate bias
            dWi: (ndarray (n_a, n_a + n_x)) - gradient of the input gate weight
            dbi: (ndarray (n_a, 1)) - gradient of the input gate bias
            dWc: (ndarray (n_a, n_a + n_x)) - gradient of the candidate value weight
            dbc: (ndarray (n_a, 1)) - gradient of the candidate value bias
            dWo: (ndarray (n_a, n_a + n_x)) - gradient of the output gate weight
            dbo: (ndarray (n_a, 1)) - gradient of the output gate bias
            dWy: (ndarray (n_y, n_a)) - gradient of the prediction weight
            dby: (ndarray (n_y, 1)) - gradient of the prediction bias
        learning_rate: (float) - learning rate for the optimization algorithm
        """

        parameters["Wf"] -= learning_rate * gradients["dWf"]
        parameters["bf"] -= learning_rate * gradients["dbf"]
        parameters["Wi"] -= learning_rate * gradients["dWi"]
        parameters["bi"] -= learning_rate * gradients["dbi"]
        parameters["Wc"] -= learning_rate * gradients["dWc"]
        parameters["bc"] -= learning_rate * gradients["dbc"]
        parameters["Wo"] -= learning_rate * gradients["dWo"]
        parameters["bo"] -= learning_rate * gradients["dbo"]
        parameters["Wy"] -= learning_rate * gradients["dWy"]
        parameters["by"] -= learning_rate * gradients["dby"]

    def clip(self, gradients, max_value):
        """
        Clips the gradients to prevent exploding gradients.

        Parameters
        ----------
        gradients: (dict) - dictionary containing the following gradients:
            dWf: (ndarray (n_a, n_a + n_x)) - gradient of the forget gate weight
            dbf: (ndarray (n_a, 1)) - gradient of the forget gate bias
            dWi: (ndarray (n_a, n_a + n_x)) - gradient of the input gate weight
            dbi: (ndarray (n_a, 1)) - gradient of the input gate bias
            dWc: (ndarray (n_a, n_a + n_x)) - gradient of the candidate value weight
            dbc: (ndarray (n_a, 1)) - gradient of the candidate value bias
            dWo: (ndarray (n_a, n_a + n_x)) - gradient of the output gate weight
            dbo: (ndarray (n_a, 1)) - gradient of the output gate bias
            dWy: (ndarray (n_y, n_a)) - gradient of the prediction weight
            dby: (ndarray (n_y, 1)) - gradient of the prediction bias
        max_value: (float) - maximum value for the gradients

        Returns
        -------
        gradients: (dict) - the clipped gradients
        """

        dWf, dbf = gradients["dWf"], gradients["dbf"]
        dWi, dbi = gradients["dWi"], gradients["dbi"]
        dWc, dbc = gradients["dWc"], gradients["dbc"]
        dWo, dbo = gradients["dWo"], gradients["dbo"]
        dWy, dby = gradients["dWy"], gradients["dby"]

        for gradient in [dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby]:
            np.clip(gradient, -max_value, max_value, out=gradient)

        gradients = {
            "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi, "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo,
            "dWy": dWy, "dby": dby
        }
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

    def cell_forward(self, xt, at_prev, ct_prev, parameters):
        """
        Implement a single forward step of the LSTM-cell.

        Parameters
        ----------
        xt: (ndarray (n_x, m)) - input data at time step "t"
        at_prev: (ndarray (n_a, m)) - hidden state at time step "t-1"
        ct_prev: (ndarray (n_a, m)) - memory state at time step "t-1"
        parameters: (dict) - dictionary containing:
            Wf: (ndarray (n_a, n_a + n_x)) - weight matrix of the forget gate
            bf: (ndarray (n_a, 1)) - bias of the forget gate
            Wi: (ndarray (n_a, n_a + n_x)) - weight matrix of the input gate
            bi: (ndarray (n_a, 1)) - bias of the input gate
            Wc: (ndarray (n_a, n_a + n_x)) - weight matrix of the candidate value
            bc: (ndarray (n_a, 1)) - bias of the candidate value
            Wo: (ndarray (n_a, n_a + n_x)) - weight matrix of the output gate
            bo: (ndarray (n_a, 1)) - bias of the output gate
            Wy: (ndarray (n_y, n_a)) - weight matrix relating the hidden-state to the output
            by: (ndarray (n_y, 1)) - bias relating the hidden-state to the output

        Returns
        -------
        at: (ndarray (n_a, m)) - hidden state at time step "t"
        ct: (ndarray (n_a, m)) - memory state at time step "t"
        y_hat_t: (ndarray (n_y, m)) - prediction at time step "t"
        cache: (tuple) - values needed for the backward pass, contains (at, ct, at_prev, ct_prev, ft, it, cct, ot, xt, y_hat_t, zyt)
        """

        Wf, bf = parameters["Wf"], parameters["bf"]  # forget gate weight and biases
        Wi, bi = parameters["Wi"], parameters["bi"]  # input gate weight and biases
        Wc, bc = parameters["Wc"], parameters["bc"]  # candidate value weight and biases
        Wo, bo = parameters["Wo"], parameters["bo"]  # output gate weight and biases
        Wy, by = parameters["Wy"], parameters["by"]  # prediction weight and biases

        concat = np.concatenate((at_prev, xt), axis=0)

        ft = sigmoid(Wf @ concat + bf)  # forget gate
        it = sigmoid(Wi @ concat + bi)  # update gate
        cct = tanh(Wc @ concat + bc)  # candidate value
        ct = ft * ct_prev + it * cct  # memory state
        ot = sigmoid(Wo @ concat + bo)  # output gate
        at = ot * tanh(ct)  # hidden state

        zyt = Wy @ at + by
        y_hat_t = softmax(zyt)
        cache = (at, ct, at_prev, ct_prev, ft, it, cct, ot, xt, y_hat_t, zyt)
        return at, ct, y_hat_t, cache

    def forward(self, X, a0, parameters):
        """
        Implement the forward pass of the LSTM network.

        Parameters
        ----------
        X: (ndarray (n_x, m, T_x)) input data for each time step
        a0: (ndarray (n_a, m)) initial hidden state
        parameters: (dict) dictionary containing:
            Wf: (ndarray (n_a, n_a + n_x)) weight matrix of the forget gate
            bf: (ndarray (n_a, 1)) bias of the forget gate
            Wi: (ndarray (n_a, n_a + n_x)) weight matrix of the update gate
            bi: (ndarray (n_a, 1)) bias of the update gate
            Wc: (ndarray (n_a, n_a + n_x)) weight matrix of the candidate value
            bc: (ndarray (n_a, 1)) bias of the candidate value
            Wo: (ndarray (n_a, n_a + n_x)) weight matrix of the output gate
            bo: (ndarray (n_a, 1)) bias of the output gate
            Wy: (ndarray (n_y, n_a)) weight matrix relating the hidden-state to the output
            by: (ndarray (n_y, 1)) bias relating the hidden-state to the output

        Returns
        -------
        A: (ndarray (n_a, m, T_x)) - hidden states for each time step
        C: (ndarray (n_a, m, T_x)) - memory states for each time step
        Y_hat: (ndarray (n_y, m, T_x)) - predictions for each time step
        caches: (list) - values needed for the backward pass
        """

        caches = []

        Wy = parameters["Wy"]
        x_x, m, T_x = X.shape
        n_y, n_a = Wy.shape

        A = np.zeros((n_a, m, T_x))
        C = np.zeros((n_a, m, T_x))
        Y_hat = np.zeros((n_y, m, T_x))

        at_prev = a0
        ct_prev = np.zeros((n_a, m))

        for t in range(T_x):
            at_prev, ct_prev, y_hat_t, cache = self.cell_forward(X[:, :, t], at_prev, ct_prev, parameters)
            A[:, :, t] = at_prev
            C[:, :, t] = ct_prev
            Y_hat[:, :, t] = y_hat_t
            caches.append(cache)

        return A, C, Y_hat, caches

    def cell_backward(self, y, dat, dct, cache, parameters):
        """
        Implement the backward pass for the LSTM-cell.

        Parameters
        ----------
        y: (ndarray (n_y, m)) - true labels for time step "t"
        dat: (ndarray (n_a, m)) - hidden state gradient for time step "t"
        dct: (ndarray (n_a, m)) - memory state gradient for time step "t"
        cache: (tuple) - values from the forward pass at time step "t"
        parameters: (dict) dictionary containing:
            Wf: (ndarray (n_a, n_a + n_x)) - weight matrix of the forget gate
            bf: (ndarray (n_a, 1)) - bias of the forget gate
            Wi: (ndarray (n_a, n_a + n_x)) - weight matrix of the update gate
            bi: (ndarray (n_a, 1)) - bias of the update gate
            Wc: (ndarray (n_a, n_a + n_x)) - weight matrix of the candidate value
            bc: (ndarray (n_a, 1)) - bias of the candidate value
            Wo: (ndarray (n_a, n_a + n_x)) - weight matrix of the output gate
            bo: (ndarray (n_a, 1)) - bias of the output gate
            Wy: (ndarray (n_y, n_a)) - weight matrix relating the hidden-state to the output
            by: (ndarray (n_y, 1)) - bias relating the hidden-state to the output

        Returns
        -------
        gradients: (dict) - dictionary containing the following gradients:
            dWf: (ndarray (n_a, n_a + n_x)) gradient of the forget gate weight
            dbf: (ndarray (n_a, 1)) gradient of the forget gate bias
            dWi: (ndarray (n_a, n_a + n_x)) gradient of the update gate weight
            dbi: (ndarray (n_a, 1)) gradient of the update gate bias
            dWc: (ndarray (n_a, n_a + n_x)) gradient of the candidate value weight
            dbc: (ndarray (n_a, 1)) gradient of the candidate value bias
            dWo: (ndarray (n_a, n_a + n_x)) gradient of the output gate weight
            dbo: (ndarray (n_a, 1)) gradient of the output gate bias
            dWy: (ndarray (n_y, n_a)) gradient of the prediction weight
            dby: (ndarray (n_y, 1)) gradient of the prediction bias
            dat_prev: (ndarray (n_a, m)) gradient of the hidden state for time step "t-1"
            dct_prev: (ndarray (n_a, m)) gradient of the memory state for time step "t-1"
            dxt: (ndarray (n_x, m)) gradient of the input data for time step "t"
        """

        at, ct, at_prev, ct_prev, ft, it, cct, ot, xt, y_hat_t, zyt = cache
        n_a, m = at.shape

        dzy = y_hat_t - y
        dWy = dzy @ at.T
        dby = np.sum(dzy, axis=1, keepdims=True)

        dat = parameters["Wy"].T @ dzy + dat

        dot = dat * tanh(ct) * ot * (1 - ot)
        dcct = (dct + dat * ot * (1 - tanh(ct) ** 2)) * it * (1 - cct ** 2)
        dit = (dct + dat * ot * (1 - tanh(ct) ** 2)) * cct * it * (1 - it)
        dft = (dct + dat * ot * (1 - tanh(ct) ** 2)) * ct_prev * ft * (1 - ft)

        concat = np.concatenate((at_prev, xt), axis=0)

        dWo = dot @ concat.T
        dbo = np.sum(dot, axis=1, keepdims=True)
        dWc = dcct @ concat.T
        dbc = np.sum(dcct, axis=1, keepdims=True)
        dWi = dit @ concat.T
        dbi = np.sum(dit, axis=1, keepdims=True)
        dWf = dft @ concat.T
        dbf = np.sum(dft, axis=1, keepdims=True)

        dat_prev = (
            parameters["Wo"][:, :n_a].T @ dot + parameters["Wc"][:, :n_a].T @ dcct +
            parameters["Wi"][:, :n_a].T @ dit + parameters["Wf"][:, :n_a].T @ dft
        )
        dct_prev = dct * ft + ot * (1 - tanh(ct) ** 2) * ft * dat
        dxt = (
            parameters["Wo"][:, n_a:].T @ dot + parameters["Wc"][:, n_a:].T @ dcct +
            parameters["Wi"][:, n_a:].T @ dit + parameters["Wf"][:, n_a:].T @ dft
        )

        gradients = {
            "dWo": dWo, "dbo": dbo, "dWc": dWc, "dbc": dbc, "dWi": dWi, "dbi": dbi, "dWf": dWf, "dbf": dbf,
            "dWy": dWy, "dby": dby, "dct_prev": dct_prev, "dat_prev": dat_prev, "dxt": dxt,
        }
        return gradients

    def backward(self, X, Y, parameters, caches):
        """
        Implement the backward pass for the LSTM network.

        Parameters
        ----------
        X: (ndarray (n_x, m, T_x)) input data for each time step
        Y: (ndarray (n_y, m, T_x)) true labels for each time step
        parameters: (dict) dictionary containing:
            Wf: (ndarray (n_a, n_a + n_x)) weight matrix of the forget gate
            bf: (ndarray (n_a, 1)) bias of the forget gate
            Wi: (ndarray (n_a, n_a + n_x)) weight matrix of the update gate
            bi: (ndarray (n_a, 1)) bias of the update gate
            Wc: (ndarray (n_a, n_a + n_x)) weight matrix of the candidate value
            bc: (ndarray (n_a, 1)) bias of the candidate value
            Wo: (ndarray (n_a, n_a + n_x)) weight matrix of the output gate
            bo: (ndarray (n_a, 1)) bias of the output gate
            Wy: (ndarray (n_y, n_a)) weight matrix relating the hidden-state to the output
            by: (ndarray (n_y, 1)) bias relating the hidden-state to the output
        caches: (list) values needed for the backward pass

        Returns
        -------
        gradients: (dict) - dictionary containing the following gradients:
            dWf: (ndarray (n_a, n_a + n_x)) gradient of the forget gate weight
            dbf: (ndarray (n_a, 1)) gradient of the forget gate bias
            dWi: (ndarray (n_a, n_a + n_x)) gradient of the update gate weight
            dbi: (ndarray (n_a, 1)) gradient of the update gate bias
            dWc: (ndarray (n_a, n_a + n_x)) gradient of the candidate value weight
            dbc: (ndarray (n_a, 1)) gradient of the candidate value bias
            dWo: (ndarray (n_a, n_a + n_x)) gradient of the output gate weight
            dbo: (ndarray (n_a, 1)) gradient of the output gate bias
            dWy: (ndarray (n_y, n_a)) gradient of the prediction weight
            dby: (ndarray (n_y, 1)) gradient of the prediction bias
        """

        n_x, m, T_x = X.shape
        a1, c1, a0, c0, f1, i1, cc1, o1, x1, y_hat_1, zy1 = caches[0]
        Wf, Wi, Wc, Wo, Wy = parameters["Wf"], parameters["Wi"], parameters["Wc"], parameters["Wo"], parameters["Wy"]
        bf, bi, bc, bo, by = parameters["bf"], parameters["bi"], parameters["bc"], parameters["bo"], parameters["by"]

        gradients = {
            "dWf": np.zeros_like(Wf), "dWi": np.zeros_like(Wi), "dWc": np.zeros_like(Wc), "dWo": np.zeros_like(Wo),
            "dbf": np.zeros_like(bf), "dbi": np.zeros_like(bi), "dbc": np.zeros_like(bc), "dbo": np.zeros_like(bo),
            "dWy": np.zeros_like(Wy), "dby": np.zeros_like(by),
        }

        dat = np.zeros_like(a0)
        dct = np.zeros_like(c0)
        for t in reversed(range(T_x)):
            grads = self.cell_backward(Y[:, :, t], dat, dct, caches[t], parameters)
            gradients["dWf"] += grads["dWf"]
            gradients["dWi"] += grads["dWi"]
            gradients["dWc"] += grads["dWc"]
            gradients["dWo"] += grads["dWo"]
            gradients["dbf"] += grads["dbf"]
            gradients["dbi"] += grads["dbi"]
            gradients["dbc"] += grads["dbc"]
            gradients["dbo"] += grads["dbo"]
            gradients["dWy"] += grads["dWy"]
            gradients["dby"] += grads["dby"]
            dat = grads["dat_prev"]
            dct = grads["dct_prev"]

        return gradients

    def optimize(self, X, Y, a_prev, parameters, learning_rate, clip_value):
        """
        Implements the forward and backward propagation of the LSTM.

        Parameters
        ----------
        X: (ndarray (n_x, m, T_x)) - input data for each time step
        Y: (ndarray (n_y, m, T_x)) - true labels for each time step
        a_prev: (ndarray (n_a, m)) - initial hidden state
        parameters: (dict) - dictionary containing:
            Wf: (ndarray (n_a, n_a + n_x)) - weight matrix of the forget gate
            bf: (ndarray (n_a, 1)) - bias of the forget gate
            Wi: (ndarray (n_a, n_a + n_x)) - weight matrix of the update gate
            bi: (ndarray (n_a, 1)) - bias of the update gate
            Wc: (ndarray (n_a, n_a + n_x)) - weight matrix of the candidate value
            bc: (ndarray (n_a, 1)) - bias of the candidate value
            Wo: (ndarray (n_a, n_a + n_x)) - weight matrix of the output gate
            bo: (ndarray (n_a, 1)) - bias of the output gate
            Wy: (ndarray (n_y, n_a)) - weight matrix relating the hidden-state to the output
            by: (ndarray (n_y, 1)) - bias relating the hidden-state to the output
        learning_rate: (float) - learning rate for the optimization algorithm
        clip_value: (float) - maximum value for the gradients

        Returns
        -------
        at: (ndarray (n_a, m)) hidden state for the last time step
        loss: (float) - the cross-entropy
        """

        A, C, Y_hat, caches = self.forward(X, a_prev, parameters)
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
        Train the LSTM model on the given text.

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
        Generate text using the LSTM model.

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

    lstm = LSTM(64, vocab_size, vocab_size)
    losses = lstm.train(text, char_to_idx, num_iterations=100, learning_rate=0.01, clip_value=5)

    generated_text = lstm.sample("T", char_to_idx, idx_to_char, num_chars=100)
    print(generated_text)
