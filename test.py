from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

import numpy as np

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))

import random
import numpy as np
import matplotlib.pyplot as plt

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128

#####################################################################################################

from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean = mu, stddev = sigma))
    b1 = tf.Variable(tf.zeros([6]))
    conv1 = tf.nn.conv2d(x, w1, strides = [1, 1, 1, 1], padding = 'VALID')
    conv1 = tf.nn.bias_add(conv1, b1)

    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob = prob)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    w2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean = mu, stddev = sigma))
    b2 = tf.Variable(tf.zeros([16]))
    conv2 = tf.nn.conv2d(pool1, w2, strides = [1, 1, 1, 1], padding = 'VALID')
    conv2 = tf.nn.bias_add(conv2, b2)

    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob = prob)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flat = flatten(pool2)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    w3 = tf.Variable(tf.truncated_normal([400, 120], mean = mu, stddev = sigma))
    b3 = tf.Variable(tf.zeros([120]))
    full1 = tf.matmul(flat, w3)
    full1 = tf.nn.bias_add(full1, b3)

    # TODO: Activation.
    full1 = tf.nn.relu(full1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    w4 = tf.Variable(tf.truncated_normal([120, 84], mean = mu, stddev = sigma))
    b4 = tf.Variable(tf.zeros([84]))
    full2 = tf.matmul(full1, w4)
    full2 = tf.nn.bias_add(full2, b4)

    # TODO: Activation.
    full2 = tf.nn.relu(full2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    w5 = tf.Variable(tf.truncated_normal([84, 10], mean = mu, stddev = sigma))
    b5 = tf.Variable(tf.zeros([10]))
    logits = tf.matmul(full2, w5)
    logits = tf.nn.bias_add(logits, b5)

    return logits

#####################################################################################################

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
prob = tf.placeholder(tf.float32, (None))
one_hot_y = tf.one_hot(y, 10)

rate = 0.0001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, prob: 1.})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

from tqdm import tqdm
import sys

pbar = tqdm(total = EPOCHS * (len(X_train) + 1)//BATCH_SIZE)

with tf.Session() as sess:

    choice = input('\n\nDo you want to: \n\n1. Start training afresh \n\
2. Continue from the last saved training \nEnter your response here: ')

    if choice == '1':

        sess.run(tf.global_variables_initializer())

    elif choice == '2':
        saver.restore(sess, './lenet')

    else:
        sys.exit()

    num_examples = len(X_train)

    tqdm.write("Training...")
    tqdm.write('\n')
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)

        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, prob: 0.5})

            pbar.update(1)

        validation_accuracy = evaluate(X_validation, y_validation)
        tqdm.write("EPOCH {} ...".format(i+1))
        tqdm.write("Validation Accuracy = {:.3f}".format(validation_accuracy))
        tqdm.write('\n')

    saver.save(sess, './lenet')
    tqdm.write("Model saved")

# Test accuracy
''' with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    tqdm.write("Test Accuracy = {:.3f}".format(test_accuracy))  '''
