import os
import sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from model import load_mnist
from model import MultiLayerNet
from model import SGD


(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True)

network = MultiLayerNet(input_size=784, hidden_size_list=[64, 64],
                        output_size=10, l2_lambda=0.05)
optimizer = SGD(lr=0.05, exponetial_decay=True)
max_epochs = 21
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
test_loss_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    grads = network.gradient(x_batch, y_batch)
    optimizer.update(network.params, grads, epoch_cnt)

    if i % iter_per_epoch == 0:
        train_loss = network.loss(x_train, y_train)
        test_loss = network.loss(x_test, y_test)
        test_acc = network.accuracy(x_test, y_test)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print("epoch" + str(epoch_cnt) + ", train loss:" + str(train_loss) + ",test loss:" + str(test_loss))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

network.save_params('params.pkl')
print('模型参数已保存！')

def loss_curve():
    plt.figure(figsize=(20, 10), dpi=70)  # 设置图像大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplot(1, 2, 1)
    x = np.arange(max_epochs)
    plt.plot(x, train_loss_list, color="lightcoral", linewidth=5.0, linestyle="-", label="train loss")
    plt.plot(x, test_loss_list, color="mediumpurple", linewidth=5.0, linestyle="--", label="test loss")
    plt.legend(["train loss", "test loss"], ncol=2)
    plt.xlabel("epoch", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.show()

def acc_curve():
    plt.figure(figsize=(20, 10), dpi=70)  # 设置图像大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplot(1,2,2)
    x = np.arange(max_epochs)
    plt.plot(x, test_acc_list, color="mediumpurple", linewidth=5.0, linestyle="--", label="test acc")
    plt.legend(["test acc"], ncol=2)
    plt.xlabel("epoch", fontsize=20)
    plt.ylabel("ACC", fontsize=20)
    plt.show()

loss_curve()
acc_curve()

network = MultiLayerNet(input_size=784, hidden_size_list=[64, 64],
                        output_size=10, l2_lambda=0.05)
network.load_params('params.pkl')
W1,W2=network.params['W1'],network.params['W2']

plt.imshow(W1, cmap='plasma',interpolation='nearest')
plt.ylabel("Input Layer")
plt.xlabel("Hidden Layer")
plt.colorbar(label='Parameter Value')
plt.savefig('W1.png', dpi=128)
plt.savefig('W1.svg')
plt.show()

plt.imshow(W2, cmap='plasma',interpolation='nearest')
plt.xlabel("Output Layer")
plt.ylabel("Hidden Layer")
plt.colorbar(label='Parameter Value')
plt.savefig('W2.png', dpi=128)
plt.savefig('W2.svg')
plt.show()
