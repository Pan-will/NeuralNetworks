from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 一、设置输入和输出节点的个数,配置神经网络的参数。
# MNIST数据集相关参数
INPUT_NODE = 784  # 输入层的节点数，等于图片的像素。28*28=784
OUTPUT_NODE = 10  # 输出层的节点数，等于类别的数目。要识别的是10个（或者10类）手写数字0~9
LAYER1_NODE = 500  # 隐藏层数，只设置一个隐藏层，该隐藏层设置500个节点。

BATCH_SIZE = 100  # 每次batch打包的样本个数，越小训练过程越接近随机梯度下降；越大训练过程越接近梯度下降。

# 模型相关的参数
LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARAZTION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数。
TRAINING_STEPS = 4000  # 训练轮数/次数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 二、定义辅助函数来计算前向传播结果，使用ReLU做为激活函数。
"""
inference函数是辅助函数，给定神经网络的输入和所有参数。计算神经网络的前向传播结果。
在这里定义的是一个使用ReLU激活函数的三层全连接神经网络。
通过加入隐藏层实现了多层神经网络结构；通过ReLU激活函数实现了去线性化。
在这个函数中也支持传入用于计算参数平均值的类，这样方便在测试时使用滑动平均模型。
"""
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 不使用滑动平均类：直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果，这里使用ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        """
        计算前向传播结果。因为在计算损失函数时会一并计算softmax函数，所以这里不用加激活函数。
        而且不加入softmax不会影响预测结果，因为预测时使用的是不同类别对应节点输出值的相对大小，有没有softmax层对最后的分类结果的计算没有影响。
        因此在计算整个神经网络前向传播结果时可以不加入最后的softmax层。
        """
        return tf.matmul(layer1, weights2) + biases2
    # 使用滑动平均类
    else:
        # 首先使用avg_class.average函数来计算得出变量的滑动平均值
        # 然后在计算相应神经网络的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 三、定义训练神经网络模型的过程。
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    # 计算在当前参数下神经网络前向传播的结果。这里给出的用于计算滑动平均的类为None，所以函数不会使用参数的滑动平均值。
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数及相关的滑动平均类
    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以指定为不可训练的变量。
    # 使用TensorFlow训练神经网络时，一般都会将表示训练轮数的变量指定为不可训练的变量。
    global_step = tf.Variable(0, trainable=False)
    # 给定训练轮数的变量可以加速早期变量的更新速度。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion

    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("%d 轮训练后, 模型准确率在验证集上达到： %g " % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("%d 轮训练后, 模型准确率在测试集上达到： %g" % (TRAINING_STEPS, test_acc)))


# 主程序入口，设定训练次数为5000次
def main(argv=None):
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
