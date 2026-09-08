import numpy as np

#Helper
def ReLU(x: np.ndarray)->np.ndarray:
    return np.maximum(0, x)

def ReLU_derivative(x: int)->bool:
    return x>0
def softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=0, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=0, keepdims=True)
def one_hot_encode(y:np.ndarray)->np.ndarray:
    y_encode = np.zeros([y.size, y.max() + 1])
    y_encode[np.arange(y.size),y] = 1
    y_encode = y_encode.T
    return y_encode

def get_pred (output_layer: np.ndarray):
    return np.argmax(output_layer, axis=0)
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

class NeuralNetwork:
    def __init__(self, layer_dims, activations=None, seed=42):
        """
        layer_dims : list, e.g. [784, 128, 64, 10]
                     input size, hidden sizes..., output size.
        activations: list of length len(layer_dims)-1, one name per
                     weighted layer (e.g. ["relu", "relu", "softmax"]).
                     Defaults to relu for hidden layers, softmax for output.
        """
        # store layer_dims, activations, L = len(layer_dims) - 1
        # call self._initialize_param() to set self.W, self.b

        self.layer_dims = layer_dims
        self.n_layer = len(layer_dims) - 1 

        if activations is not None:
            self.activations = activations
        else:
            self.activations = ["none"] + ["relu"] * (self.n_layer - 1) + ["softmax"]
            # {1: "None", 2: "relu", 3: "softmax"}, Layer đầu tiên không cần hàm active

        self.seed = seed

        self._initialize_param()

    # ---- setup ----
    def _initialize_param(self):
        """Build self.W (dict, layer -> weight matrix) and self.b likewise."""
        rng = np.random.default_rng(self.seed)

        self.W = {}
        self.b = {}
        
        for i in range(1,self.n_layer+1):
            He_initialization = np.sqrt(2 / self.layer_dims[i-1])
            self.W[i] = rng.standard_normal((self.layer_dims[i], self.layer_dims[i-1])) *He_initialization 
            self.b[i] = np.zeros([self.layer_dims[i],1]) 

    # ---- activation dispatch ----
    def _activate(self, Z, name):
        """Apply the named activation function to Z."""
        activate_func = {
            "relu": ReLU,
            "softmax": softmax,
            "none": lambda x: x,
        }
        return activate_func[name](Z)


    def _activate_derivative(self, Z, name):
        """Derivative of the named activation, evaluated at Z."""
        activate_derivative = {
            "relu": ReLU_derivative,
        }
        return activate_derivative[name](Z)

    # ---- core training steps ----
    def _forward(self, X):
        """
        Run X through all L layers.
        Returns (A_L, cache) where cache holds every Z[l] and A[l]
        needed by _backward.
        """
        Z = {}
        A = {}
        A[0] = X

        for i in range(1, self.n_layer + 1):
            Z[i] = self.W[i] @ A[i-1] + self.b[i]
            A[i] = self._activate(Z[i], self.activations[i])

        cache = {"Z": Z, "A": A}
        return A[self.n_layer], cache


    def _backward(self, y, cache):
        y_ohe = one_hot_encode(y)
        n_samples = y.shape[0]
        dZ = {}
        dW = {}
        db = {}

        dZ[self.n_layer] = cache["A"][self.n_layer] - y_ohe
        for i in range(self.n_layer, 0, -1):
            dW[i] = (1/n_samples) * dZ[i] @ cache["A"][i-1].T
            db[i] = (1/n_samples) * np.sum(dZ[i], axis=1, keepdims=True)

            if i > 1:
                grad = self._activate_derivative(cache["Z"][i-1], self.activations[i-1])
                dZ[i-1] = (self.W[i].T @ dZ[i]) * grad

        grads = {"dW": dW, "db": db}
        return grads
        

    def _update(self, grads, learning_rate):
        """Apply gradient descent to self.W / self.b in place."""
        for i in range(1, self.n_layer+1):
            self.W[i] -= learning_rate * grads["dW"][i]
            self.b[i] -= learning_rate * grads["db"][i]
    

    # ---- public interface ----
    def fit(self, X, y, X_val=None, y_val=None, learning_rate=0.1,
            epochs=1000, print_every=10):
        """Training loop: repeatedly call _forward, _backward, _update."""
        for i in range(epochs):
            A_L, cache = self._forward(X)
            grads = self._backward(y, cache)
            self._update(grads, learning_rate)

            if i % print_every == 0:
                print("Iteration: ", i)
                predictions = get_pred(A_L)
                print(get_accuracy(predictions, y))

                if X_val is not None and y_val is not None:
                    val_acc = self.accuracy(X_val, y_val)
                    print(f"  val_acc: {val_acc:.4f}")

    def predict(self, X):
        """Run _forward, return argmax over the output layer."""
        A_L, _ = self._forward(X)
        return get_pred(A_L)

    def accuracy(self, X, y):
        """predict(X) compared against true labels y."""
        predictions = self.predict(X)
        return get_accuracy(predictions, y)
        
