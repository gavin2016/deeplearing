# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 神经网络参数
BATCH_SIZE = 32
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_EPOCHS = 50
KEEP_PROB = 0.4
# 滑动平均
MOVING_AVERAGE_DECAY = 0.99

# 输入层特征数
INPUT_SIZE = 784
# 0-9 共10个类别
OUTPUT_SIZE = 10
# 各隐藏层神经元数目
LAYER1_SIZE = 512
LAYER2_SIZE = 256
LAYER3_SIZE = 128
LAYER4_SIZE = 500


def getWeightVariable(shape, unitsSize, layerNo):
    weights = tf.Variable(tf.random_normal(shape, stddev=0.1) * tf.sqrt(2.0 / unitsSize ** (layerNo - 1)),
                          name='weight', dtype=tf.float32)
    return weights


EPSILON = 0.001


# 批归一化
def batchNormalization(Wx_plus_b, out_size, ema, norm=True, trainning=True):
    if norm:
        fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0], )
        scale = tf.Variable(tf.ones([out_size]))
        shift = tf.Variable(tf.zeros([out_size]))
        if trainning:
            def mean_var_with_update():
                ema_apply = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)

        Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, EPSILON)
    return Wx_plus_b


# 每层数据都做批归一化
def createDNN(input_tensor, mvDecay=0.99, keep_prob=None, trainning=True):
    fc_mean, fc_var = tf.nn.moments(
        input_tensor,
        axes=[0],
    )
    scale = tf.Variable(tf.ones([1]))
    shift = tf.Variable(tf.zeros([1]))
    ema = tf.train.ExponentialMovingAverage(decay=mvDecay)

    def mean_var_with_update():
        ema_apply = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply]):
            return tf.identity(fc_mean), tf.identity(fc_var)

    mean, var = mean_var_with_update()
    input_tensor = tf.nn.batch_normalization(input_tensor, mean, var, shift, scale, EPSILON)

    # 声明各隐藏层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):
        weights = getWeightVariable([INPUT_SIZE, LAYER1_SIZE], INPUT_SIZE, 1)
        biases = tf.Variable(tf.zeros([LAYER1_SIZE]), name='biases', dtype=tf.float32)
        Wx_plus_b = tf.matmul(input_tensor, weights) + biases
        Wx_plus_b = batchNormalization(Wx_plus_b, LAYER1_SIZE, ema, True, trainning)
        layer1 = tf.nn.relu(Wx_plus_b)

    with tf.variable_scope('layer2'):
        weights = getWeightVariable([LAYER1_SIZE, LAYER2_SIZE], LAYER1_SIZE, 2)
        biases = tf.Variable(tf.zeros([LAYER2_SIZE]), name='biases', dtype=tf.float32)
        Wx_plus_b = tf.matmul(layer1, weights) + biases
        Wx_plus_b = batchNormalization(Wx_plus_b, LAYER2_SIZE, ema, True, trainning)
        layer2 = tf.nn.relu(Wx_plus_b)

    with tf.variable_scope('layer3'):
        weights = getWeightVariable([LAYER2_SIZE, LAYER3_SIZE], LAYER2_SIZE, 3)
        biases = tf.Variable(tf.zeros([LAYER3_SIZE]), name='biases', dtype=tf.float32)
        Wx_plus_b = tf.matmul(layer2, weights) + biases
        Wx_plus_b = batchNormalization(Wx_plus_b, LAYER3_SIZE, ema, True, trainning)
        layer3 = tf.nn.relu(Wx_plus_b)

    with tf.variable_scope('layer4'):
        weights = getWeightVariable([LAYER3_SIZE, LAYER4_SIZE], LAYER3_SIZE, 4)
        biases = tf.Variable(tf.zeros([LAYER4_SIZE]), name='biases', dtype=tf.float32)
        Wx_plus_b = tf.matmul(layer3, weights) + biases
        Wx_plus_b = batchNormalization(Wx_plus_b, LAYER4_SIZE, ema, True, trainning)
        layer4 = tf.nn.relu(Wx_plus_b)
        if keep_prob != None:
            # dropout处理
            fc_drop = tf.nn.dropout(layer4, keep_prob)
        else:
            fc_drop = layer4
    with tf.variable_scope('layer5'):
        weights = getWeightVariable([LAYER4_SIZE, OUTPUT_SIZE], LAYER4_SIZE, 5)
        biases = tf.Variable(tf.zeros([OUTPUT_SIZE]), name='biases', dtype=tf.float32)
        Wx_plus_b = tf.matmul(fc_drop, weights) + biases
        layer5 = batchNormalization(Wx_plus_b, OUTPUT_SIZE, ema, True, trainning)

    # 返回最后前向传播的结果
    return layer5


def train(mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(
        tf.float32, [None, INPUT_SIZE], name='x-input')
    y_ = tf.placeholder(
        tf.float32, [None, OUTPUT_SIZE], name='y-input')

    global_step = tf.Variable(0, trainable=False)

    keep_prob = tf.placeholder(tf.float32)
    #
    hypothesis = createDNN(x, MOVING_AVERAGE_DECAY, keep_prob, True)

    # 损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=hypothesis, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(cross_entropy)

    # 学习率衰减
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    # 梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_apply = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([optimizer, ema_apply]):
        optimizer_op = tf.no_op(name='train')

    # 准确度
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_batch = int(mnist.train.num_examples / BATCH_SIZE)
        begin = time.time()
        for epoch in range(TRAINING_EPOCHS):
            avg_cost = 0.0
            for i in range(total_batch):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                feed_dict = {x: xs, y_: ys, keep_prob: KEEP_PROB}
                _, loss_value, step = sess.run([optimizer_op, loss, global_step], feed_dict=feed_dict)
                avg_cost += loss_value

            avg_cost = avg_cost * 1.0 / total_batch
            print('Epoch(s):', '%03d' % (epoch + 1), 'cost =', '{:.6f}'.format(avg_cost))

        end = time.time()
        print('训练耗时:' + str(1000 * (end - begin)) + " ms")

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1}
        accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
        print('validation accuracy ={:.6f}'.format(accuracy_score))

        test_feed = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}
        accuracy_score = sess.run(accuracy, feed_dict=test_feed)
        print('test accuracy ={:.6f}'.format(accuracy_score))


def main(argv=None):
    mnist = input_data.read_data_sets("../data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
