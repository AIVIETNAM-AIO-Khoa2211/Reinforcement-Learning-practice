# Generalizing your Neural Network to N Layers, M Neurons

Your current code is hardcoded for exactly 2 layers (`W1, b1, W2, b2`) and
passes those parameters around as separate function arguments. To generalize
it to N layers, two things need to change together:

1. **Store parameters indexed by layer** instead of named by number
   (`W1`, `W2`, ...) — so the code doesn't change shape when you add a layer.
2. **Wrap it in a class** so that architecture (`layer_dims`, `activations`)
   and parameters (`W`, `b`) live as state on `self`, instead of being
   threaded through every function call by hand.

This doc covers both together: the class skeleton, and what each method is
responsible for. No implementations — you fill those in.

---

## 1. Why a class (briefly)

With standalone functions, every call ends up looking like:

```
W, b = update_param(W, b, dW, db, learning_rate)
Z_cache, A_cache = forward_prop(X, W, b, activations)
```

`W`, `b`, `layer_dims`, `activations` all have to be passed in and reassigned
every time, across every function. As a class:

- `layer_dims` and `activations` get set once in `__init__` and never passed
  around again.
- `W`/`b` become `self.W`/`self.b`, updated in place — no more reassigning
  tuples every training iteration.
- The public surface shrinks to `fit`, `predict`, `accuracy` — the
  forward/backward/update logic becomes internal helper methods. This is
  close to the shape of an sklearn-style estimator, which fits the pattern
  you're already using in your other pipeline work.

---

## 2. Class skeleton

```python
class NeuralNetwork:
    def __init__(self, layer_dims, activations=None, seed=None):
        """
        layer_dims : list, e.g. [784, 128, 64, 10]
                     input size, hidden sizes..., output size.
        activations: list of length len(layer_dims)-1, one name per
                     weighted layer (e.g. ["relu", "relu", "softmax"]).
                     Defaults to relu for hidden layers, softmax for output.
        """
        # store layer_dims, activations, L = len(layer_dims) - 1
        # call self._initialize_param() to set self.W, self.b
        pass

    # ---- setup ----
    def _initialize_param(self):
        """Build self.W (dict, layer -> weight matrix) and self.b likewise."""
        pass

    # ---- activation dispatch ----
    def _activate(self, Z, name):
        """Apply the named activation function to Z."""
        pass

    def _activate_derivative(self, Z, name):
        """Derivative of the named activation, evaluated at Z."""
        pass

    # ---- core training steps ----
    def _forward(self, X):
        """
        Run X through all L layers.
        Returns (A_L, cache) where cache holds every Z[l] and A[l]
        needed by _backward.
        """
        pass

    def _backward(self, y, cache):
        """
        Compute dW[l], db[l] for every layer, working backwards from
        the output layer to layer 1.
        """
        pass

    def _update(self, grads, learning_rate):
        """Apply gradient descent to self.W / self.b in place."""
        pass

    # ---- public interface ----
    def fit(self, X, y, X_val=None, y_val=None, learning_rate=0.1,
            epochs=1000, print_every=10):
        """Training loop: repeatedly call _forward, _backward, _update."""
        pass

    def predict(self, X):
        """Run _forward, return argmax over the output layer."""
        pass

    def accuracy(self, X, y):
        """predict(X) compared against true labels y."""
        pass
```

Everything below explains what goes inside each method and why — same
content as before, just mapped onto `self`.

---

## 3. What each method is responsible for

### `__init__` / `_initialize_param`
**Purpose:** build the initial weight/bias structure for an arbitrary number
of layers, and store the architecture on `self`.

**What to think about:**
- `self.L = len(layer_dims) - 1` — number of weighted layers.
- Loop `for l in range(1, self.L + 1)`, creating `self.W[l]` of shape
  `(layer_dims[l], layer_dims[l-1])` and `self.b[l]` of shape `(layer_dims[l], 1)`.
  A dict keyed by layer index reads closest to the math notation (`W^{[l]}`),
  which makes it easier to cross-check against backprop derivations.
- **Initialization scale matters more as depth grows.** Your current
  `np.random.rand(...) - 0.5` (uniform around 0) works okay for 1 hidden
  layer but tends to cause vanishing/exploding activations with more layers.
  Look into **He initialization** (`* sqrt(2 / n_in)`) for ReLU layers — the
  standard default. Worth understanding *why* it works (keeping activation
  variance roughly constant layer to layer) rather than just copying the
  formula.
- Also store `self.activations` here — a list like `["relu", "relu", "softmax"]`
  of length `L`, so each layer's activation is looked up independently
  instead of ReLU being hardcoded into the forward pass.

---

### `_activate` / `_activate_derivative`
**Purpose:** centralize the activation lookup instead of hardcoding
`ReLU(...)` inline.

**What to think about:**
- A small dict-based dispatch (`{"relu": ..., "sigmoid": ..., "tanh": ...}`)
  keeps `_forward`/`_backward` free of `if/elif` chains and makes it trivial
  to try sigmoid/tanh hidden layers later.
- `softmax` only needs the forward direction — you never call
  `_activate_derivative` on the output layer, because of the shortcut below.

---

### `_forward`
**Purpose:** run input through all `L` layers instead of just 2, caching
everything backprop will need.

**What to think about:**
- Loop `l = 1..L`: `Z[l] = W[l] @ A[l-1] + b[l]`, then `A[l] = self._activate(Z[l], activations[l-1])`.
- Treat `A[0]` as `X` — this makes the loop uniform, no special-casing the
  input layer.
- Cache **every** `Z[l]` and `A[l]`, not just the final output — backprop
  needs all of them. A dict like `{"Z1": ..., "A1": ..., "Z2": ...}` works
  fine, or two separate dicts.

---

### `_backward`
**Purpose:** the trickiest generalization — propagate error backward through
an arbitrary number of layers.

**What to think about:**
- Start the same way you do now: for softmax output + cross-entropy loss,
  `dZ[L] = A[L] - Y_onehot`. This shortcut still holds regardless of depth —
  it's a property of that loss/activation pairing, not something tied to
  having exactly 2 layers.
- Loop **backwards**, `l = L, L-1, ..., 1`:
  - `dW[l] = (1/m) * dZ[l] @ A[l-1].T`
  - `db[l] = (1/m) * sum(dZ[l], axis=1, keepdims=True)`
  - If `l > 1`: `dZ[l-1] = (W[l].T @ dZ[l]) * self._activate_derivative(Z[l-1], activations[l-2])`
- This is why activations need to be tracked per layer explicitly — backprop
  needs to know which derivative to apply at each step.

---

### `_update`
**Purpose:** apply gradient descent across all layers, in place on `self.W`/`self.b`.

**What to think about:**
- Simplest method — loop `l = 1..L`:
  `self.W[l] -= lr * grads[f"dW{l}"]`, same for `b`.

---

### `fit` / `predict` / `accuracy`
**Purpose:** the public interface — mostly unchanged conceptually from your
current `main_loop`/`predict`, just calling the internal methods above.

**What to think about:**
- `fit` is your training loop: call `_forward`, `_backward`, `_update` each
  iteration, optionally printing train/val accuracy every `print_every` steps.
- `predict` calls `_forward` and takes `argmax` over the final layer's output.
- `accuracy` compares `predict(X)` against true labels.

---

## 4. Suggested order to implement/test in

1. `__init__` / `_initialize_param` — instantiate with a 3–4 layer
   `layer_dims` list, print every `self.W[l].shape`/`self.b[l].shape`,
   sanity-check dimensions line up (`W[l].shape == (layer_dims[l], layer_dims[l-1])`).
2. `_forward` — run one forward pass on a small batch, check
   `A[L].shape == (num_classes, m)` and that each column of `A[L]` sums to ~1
   (softmax).
3. `_backward` — the easiest way to catch bugs here is **gradient checking**:
   numerically perturb one weight, compare the finite-difference estimate of
   the gradient to what `_backward` computes. Worth doing once even if you
   skip it later — this is where dimension-mismatch bugs hide.
4. `_update` + `fit` — train a 3-layer network first (`[784, 128, 10]`)
   before trying deeper ones, so you're only debugging the generalization,
   not also debugging a fragile deep architecture.

---

## 5. Things that get harder as you add depth (worth knowing, not urgent)

- **Vanishing/exploding gradients** get worse with more layers — this is why
  the initialization scheme matters more than it did at 2 layers.
- **Training gets slower to converge** — you may need to tune `learning_rate`
  down, or eventually look at Adam instead of plain SGD.
- **Overfitting** shows up faster with more capacity — worth watching once
  you're past 2–3 hidden layers on MNIST.

None of these require touching the method signatures above — just parameters
to tune once the class is working.