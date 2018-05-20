import tensorflow as tf

def cnn_model_fn(features, labels, mode):

    """Model function for CNN."""

    # Input Layer
    #input_layer = tf.reshape(tf.cast(features["x"], tf.float32), [-1, 320, 240, 1])
    x = tf.placeholder(tf.float32, shape=[16, 480, 640, 3], name='x')

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[1, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=[1, 2])

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=8,
        kernel_size=[1, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 5], strides=[1, 5])

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=2,
        kernel_size=[154, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #3
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1, 2], strides=[1, 2])

    # Dense Layer
    pool3_flat = tf.reshape(pool3, [16, 154 * 5 * 2])

    dense = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=4)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    multiclass_labels = tf.reshape(tf.cast(labels, tf.int32), [-1, 154])
    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=multiclass_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)