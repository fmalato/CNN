import random

import numpy as np
import cv2
import tensorflow as tf
import os
import convolutionalNN as cnn
from PIL import Image
import cnn
import dataset
from matplotlib import pyplot as plt

#### IMAGE READING FROM FOLDER #####


##### IMAGE CROPPING #####

"""img = cv2.imread("dataSet/images/1.jpg")
print(img.shape)
imageArray = np.zeros(240*320)
for i in range(240, 480):
    for j in range(160, 480):
        imageArray[i+j] = sum(img[i][j])/3
print imageArray
imageArray.resize((76800, 1, 1))
print imageArray.shape"""


##### CNN TEST #####

classes = ['forward', 'left', 'right', 'stop']
num_classes = len(classes)

train_path = 'dataSet/images/'

# validation split
validation_size = 0.3

# batch size
batch_size = 16

data = dataset.read_train_sets(train_path, 128, 96, classes, validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

session = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 96, 128, 3], name='x')

layer_conv1 = cnn.create_convolutional_layer(input=x,
                                         num_input_channels=3,
                                         conv_filter_size=3,
                                         num_filters=128)

layer_batchNorm1 = cnn.create_batch_normalization(layer_conv1)

layer_conv2 = cnn.create_convolutional_layer(input=layer_batchNorm1,
                                         num_input_channels=128,
                                         conv_filter_size=3,
                                         num_filters=64)

layer_batchNorm2 = cnn.create_batch_normalization(layer_conv2)

layer_conv3 = cnn.create_convolutional_layer(input=layer_batchNorm2,
                                         num_input_channels=64,
                                         conv_filter_size=3,
                                         num_filters=32)

layer_batchNorm3 = cnn.create_batch_normalization(layer_conv3)

layer_flat = cnn.create_flatten_layer(layer_batchNorm3)

layer_fc1 = cnn.create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=64,
                            use_relu=True)

layer_fc2 = cnn.create_fc_layer(input=layer_fc1,
                            num_inputs=64,
                            num_outputs=num_classes,
                            use_relu=True)

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost) ## Original: 1e-4
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=1e-6).minimize(cost) ## Original: 1e-2
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def show_progress(epoch, feed_dict_train, feed_dict_validate, valLosses, trainLearn, validLearn):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    val_loss = session.run(cost, feed_dict=feed_dict_validate)
    valLosses.append(val_loss)
    trainLearn.append(acc)
    validLearn.append(val_acc)

    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

saver = tf.train.Saver()

def train(num_iteration):
    global total_iterations
    initOp = tf.global_variables_initializer()
    session.run(initOp)
    valLosses = []
    trainLearn = []
    validLearn = []

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        epoch = i

        show_progress(epoch, feed_dict_tr, feed_dict_val, valLosses, trainLearn, validLearn)
        saver.save(session, 'results/steering_model')

    total_iterations += num_iteration
    return valLosses, trainLearn, validLearn

session.run(tf.global_variables_initializer())
valLosses, trainLearn, validLearn = train(30) # 250 IS ENOUGH

plt.figure(1)
plt.plot(range(0, total_iterations), valLosses)
plt.xlabel("Iteration")
plt.ylabel("Validation loss")
plt.show()

plt.figure(2)
plt.plot(range(0, total_iterations), trainLearn, validLearn)
plt.xlabel("Iteration")
plt.show()
