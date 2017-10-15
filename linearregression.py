# -*- coding: utf-8 -*-
'''
简单线性回归的例子
'''
import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(argv=None):
  W = tf.Variable(tf.random_normal([1]), name='weight')
  b = tf.Variable(tf.random_normal([1]), name='bias')
  X = tf.placeholder(tf.float32, shape=[None])
  Y = tf.placeholder(tf.float32, shape=[None])

  # hypothesis函数XW+b
  hypothesis = X * W + b
  # 代价/损失 函数
  cost = tf.reduce_mean(tf.square(hypothesis - Y))
  # 最小化
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train = optimizer.minimize(cost)

  with tf.Session() as sess:
      # 打开一个会话Session
      # 初始化变量
      sess.run(tf.global_variables_initializer())

      TRAIN_STEPS = 3600
      DISPLAY_STEPS = 200

      # 迭代训练
      for step in range(TRAIN_STEPS):
         cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
         if step % DISPLAY_STEPS == 0:
             print('step(s):', '%04d' % (step + 1), 'cost =', '{:.6f}'.format(cost_val), 'W:', W_val,'b:', b_val)
      print('cost =', '{:.6f}'.format(cost_val), 'W:', W_val, 'b:', b_val)

if __name__ == "__main__":
  main()
