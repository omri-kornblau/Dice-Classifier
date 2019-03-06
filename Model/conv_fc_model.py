import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os

sys.path.insert(0, '../Data Set')
import preprocess as prep

# Disable some tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.InteractiveSession()

# Data set params
num_samples = 10
num_train = 130
num_test = 60
num_acc_axis = 4
num_classes = 2

root_dir = "..\\Data Set\\Throws\\"
cd = lambda x: root_dir + x

# Low_Elastic_Throw
# Low_Hard_Throw
# High_Hard_Throw
# High_Elastic_Throw
# Roll

# Log files
log_paths = [cd("Throws_#1.txt")]

# Throw times files
times_paths = [cd("Throws_Times_#1.txt")]

X, y = prep.load_data_from_files(log_paths, times_paths, num_samples, graph=0, to_polar=True)

print(X.shape)
print(np.delete(X, 0, axis=2).shape)

# input("press any key to continue...")
print("Preprocess Finished!")

# Seperate into training and test data
X_train = X[:num_train]
temp_y_train = y[:num_train]

X_test = X[num_train:(num_train+num_test)]
temp_y_test = y[num_train:(num_train+num_test)]

# Format the label data as one hot
y_train = np.zeros((num_train, num_classes))
y_test = np.zeros((num_test, num_classes))

y_train[range(num_train), temp_y_train] = 1
y_test[range(num_test), temp_y_test] = 1

# Training params
learning_rate = 1e-2
display_step = 400
batch_size = num_test
maxiter = 10000

# Net params
stride = 1
n_conv_layer = 10
n_conv_out = int(num_samples/stride)
n_hidden_1 = 20
n_hidden_2 = 10

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
    scores = tf.nn.relu(layer_2)

    # Output fully connected layer with a neuron for each class
    scores = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])

    return scores

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits,
    labels=Y))

optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Merge all the summerise and write them out to C:/train
train_writer = tf.summary.FileWriter("C:" + '/train', sess.graph)

init_op = tf.global_variables_initializer()

final_scores = 0
losses = []

# Optimization
with tf.Session() as sess:
    # Run initializer
    sess.run(init_op)

    for step in range(1, maxiter+1):
        # Create random choice 'batch_size' size batch
        mask_idx = np.random.choice(range(num_train), batch_size)
        X_batch, y_batch = X_train[mask_idx], y_train[mask_idx]

        # Run optimization op
        sess.run(train_op, feed_dict={X: X_batch, Y: y_batch})
        losses.append(sess.run([loss_op], feed_dict = {X: X_batch, Y: y_batch}))

        if ((step % display_step == 0) or (step == 1)):
            # Calculate batch loss and accuracy
            loss, acc = sess.run(
                [loss_op, accuracy],
                feed_dict = {X: X_batch, Y: y_batch}
            )

            print("Step " + str(step) \
                + ", Minibatch Loss= " + "{:.4f}".format(loss) \
                + ", Minibatch accuracy= " + "{:.3f}".format(acc))

    final_scores = sess.run(accuracy, feed_dict={X: X_test,
                                      Y: y_test})

    print(final_scores)
    plt.plot(losses)
    print("Optimaztion Finished!")

print("\n Final test score %f" %final_scores)

plt.grid()
plt.show()