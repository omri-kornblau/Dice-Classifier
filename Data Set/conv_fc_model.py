import tensorflow as tf
import numpy as np
import os

# Disable some tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.InteractiveSession()

# Data set params
num_samples = 200
num_train = 1000
num_test = 200
num_acc_axis = 3
num_classes = 3

# Make fake cube accelerations data
label_data = np.random.randint(0, num_classes-1, num_train)

X_train = np.random.randn(num_train, num_samples, num_acc_axis)
y_train = np.zeros([num_train, num_classes])
y_train[range(num_train), label_data] = 1

# Training params
learning_rate = 6e-4
display_step = 50
batch_size = 500
maxiter = 500

# Net params
n_conv_layer = 6
stride = 2
n_hidden_1 = 200
n_hidden_2 = 300
n_conv_out = int(num_samples/stride)

# Graph input
X = tf.placeholder("float", [None, num_samples, num_acc_axis])
Y = tf.placeholder("float", [None, num_classes])

# Store weights and biases
conv_layer = tf.Variable(tf.random_normal([n_conv_layer, num_acc_axis, 1]), name="conv_filter")

weights = {
    'w1': tf.Variable(tf.random_normal([n_conv_out, n_hidden_1]), name="W1"),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="W2"),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, num_classes]), name="W3")
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1]), name="b1"),
    'b2': tf.Variable(tf.zeros([n_hidden_2]), name="b2"),
    'b3': tf.Variable(tf.zeros([num_classes]), name="b3")
}

# Create model computations
def neural_net(x):
    # Convolve with trainable filter to turn the data to a 2d matrix
    conv = tf.nn.conv1d(x, conv_layer, stride, "SAME")
    conv = tf.reshape(conv, [batch_size, n_conv_out])

    # Hidden fully connected layer with relu
    layer_1 = tf.add(tf.matmul(conv, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden fully connected layer with relu
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output fully connected layer with a neuron for each class
    scores = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])

    return scores

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Merge all the summerise and write them out to C:/train
train_writer = tf.summary.FileWriter("C:" + '/train', sess.graph)

init_op = tf.global_variables_initializer()

final_scores = 0

# Training
with tf.Session() as sess:
    # Run initializer
    sess.run(init_op)

    for step in range(1, maxiter+1):
        # Create random choice 'batch_size' size batch
        mask_idx = np.random.choice(range(num_train), batch_size)
        X_batch, y_batch = X_train[mask_idx], y_train[mask_idx]

        # Run optimization op
        sess.run(train_op, feed_dict={X: X_batch, Y: y_batch})

        if ((step % display_step == 0) or (step == 1)):
            # Calculate batch loss and accuracy
            loss, acc = sess.run(
                [loss_op, accuracy],
                feed_dict = {X: X_batch, Y: y_batch}
            )

            print("Step " + str(step) \
                + ", Minibatch Loss= " + "{:.4f}".format(loss) \
                + ", Minibatch accuracy= " + "{:.3f}".format(acc))

    print("Optimaztion Finished!")
