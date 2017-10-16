# TensorFlow的一些学习示例代码
linearregression.py 是一个线性回归例子

mnist_dnn.py 是设计的一个含5层隐藏层的手写数字识别神经网络，使用到了一些降低过拟合等的方法，在验证集、测试集上的准确度在0.984-0.988之间，比CNN的准确度低0.005-0.007左右。训练数据需要从TensorFlow网站上下载，放到与mnist_dnn.py文件平行的data目录下。

