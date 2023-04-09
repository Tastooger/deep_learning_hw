# deep_learning_hw
#### 作业要求
用Python搭建2层神经网络实现mnist数据集的数字分类。  
至少包含以下三个代码文件/部分  
    1. 训练：激活函数，反向传播，loss以及梯度的计算，学习率下降策略，L2正则化，优化器SGD，保存模型。  
    2. 参数查找：学习率，隐藏层大小，正则化强度。  
    3. 测试：导入模型，用经过参数查找后的模型进行测试，输出分类精度。  
不可使用pytorch，tensorflow等python package，可以使用numpy。  

#### 实现方式
model.py：神经网络模型  
parameter.py：参数查找  
solution.py：模型训练、测试与可视化  
    ①	首先运行parameter.py 进行参数查找  
    ②	将找到的最优参数用于神经网络的训练与测试，运行solution.py，同时会可视化训练和测试的loss曲线，测试的accuracy曲线，以及可视化每层的网络参数。  
    ③	训练得到的模型会保存在params.pkl  
注：数据集文件mnist.rar需先解压
