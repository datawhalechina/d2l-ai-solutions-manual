# ------------------------------
# 尝试调整超参数，例如批量大小、迭代周期数和学习率，并查看结果。
# ------------------------------
import torch
from torch import nn
from d2l import torch as d2l
import time

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

# 设置损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 设置优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# %%time
start_time = time.time()
num_epochs = 10
batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
net.apply(init_weights);
num_epochs = 100
batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
net.apply(init_weights);
num_epochs = 10
batch_size = 16

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
net.apply(init_weights);
num_epochs = 10
batch_size = 64

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
net.apply(init_weights);
num_epochs = 10
batch_size = 128

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
net.apply(init_weights);
num_epochs = 20
batch_size = 1024

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
net.apply(init_weights);
num_epochs = 20
batch_size = 128

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
# 设置损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# 设置优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

net.apply(init_weights);
num_epochs = 10
batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
try:
    # 设置损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 设置优化器
    trainer = torch.optim.SGD(net.parameters(), lr=1)

    net.apply(init_weights);
    num_epochs = 10
    batch_size = 256

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
except Exception as e:
    print(e)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
try:
    # 设置损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 设置优化器
    trainer = torch.optim.SGD(net.parameters(), lr=1)

    net.apply(init_weights);
    num_epochs = 100
    batch_size = 256

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
except Exception as e:
    print(e)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
try:
    # 设置损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 设置优化器
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)

    net.apply(init_weights);
    num_epochs = 10
    batch_size = 256

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
except Exception as e:
    print(e)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
try:
    # 设置损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 设置优化器
    trainer = torch.optim.SGD(net.parameters(), lr=0.001)

    net.apply(init_weights);
    num_epochs = 10
    batch_size = 256

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
except Exception as e:
    print(e)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
try:
    # 设置损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 设置优化器
    trainer = torch.optim.SGD(net.parameters(), lr=0.001)

    net.apply(init_weights);
    num_epochs = 100
    batch_size = 256

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
except Exception as e:
    print(e)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")

# %%time
start_time = time.time()
try:
    # 设置损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 设置优化器
    trainer = torch.optim.SGD(net.parameters(), lr=0.0001)

    net.apply(init_weights);
    num_epochs = 10
    batch_size = 256

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
except Exception as e:
    print(e)
# 记录结束时间
end_time = time.time()

# 计算并打印执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time} 秒")