# ------------------------------
# 自定义softmax函数
# ------------------------------

def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition  # 这里应用了广播机制

try:
    softmax(np.array([[50]]))
except Exception as e:
    print(e)