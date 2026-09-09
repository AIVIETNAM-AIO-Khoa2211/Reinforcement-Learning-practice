import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from src import NeuralNetwork

# 1. Load MNIST
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# 2. Chuẩn hóa pixel về [0, 1]
X = X / 255.0

# 3. Chia train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# 4. QUAN TRỌNG: transpose để đúng quy ước (n_features, n_samples)
X_train = X_train.T   # (784, m_train)
X_val   = X_val.T     # (784, m_val)
# y_train, y_val giữ nguyên shape (m,) — không transpose

# 5. Khởi tạo model
model = NeuralNetwork(layer_dims=[784, 128, 64, 10], seed=42)

# 6. Train
model.fit(X_train, y_train, X_val=X_val, y_val=y_val,
          learning_rate=0.1, epochs=500, print_every=20)

# 7. Đánh giá
print("Final train accuracy:", model.accuracy(X_train, y_train))
print("Final val accuracy:", model.accuracy(X_val, y_val))