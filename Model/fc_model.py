import tensorflow as tf
import numpy as np
import os

# Disable some tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

sess = tf.InteractiveSession()

np.random.seed(1)
train_data = np.random.rand(10000, 2)
train_labels = np.random.randint(0, 2, 10000)

X_train = train_data.reshape(train_data.shape[0], -1)
y_train = np.reshape(np.prod(X_train, axis=1), (X_train.shape[0],1))

X_test = np.random.randn(5, 2)
y_test = np.reshape(np.prod(X_test, axis=1), (X_test.shape[0],1))

# y_train = np.zeros((200, 2))
# y_train[range(200), train_labels] = 1


# Training params
num_train = y_train.shape[0]
learning_rate = 6e-4
display_step = 500
batch_size = 1000
maxiter = 5000

# Net params
n_hidden_1 = 500
n_hidden_2 = 200
num_input = X_train.shape[1]
num_classes = 1

# tf Graph input
X = tf.placeholder("float", [None, num_input], name="input")
Y = tf.placeholder("float", [None, num_classes], name="answers")

# Store weights and biases
weights = {
    'w1': tf.Variable(tf.random_normal([num_input, n_hidden_1]), name="W1"),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="W2"),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, num_classes]), name="W3")
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1]), name="b1"),
    'b2': tf.Variable(tf.zeros([n_hidden_2]), name="b2"),
    'b3': tf.Variable(tf.zeros([num_classes]), name="b3")
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with relu
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.maximum(layer_1, 0)

    # Hidden fully connected layer with relu
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.maximum(layer_2, 0)

    # Output fully connected layer with a neuron for each class
    scores = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])

    return scores

# Construct model
prediction = neural_net(X)
tf.summary.scalar('prediction', prediction)
# prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.losses.mean_squared_error(labels=Y, predictions=prediction)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = prediction

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("C:" + '/train',
                                      sess.graph)

# Initialize variables
init_op = tf.global_variables_initializer()

final_scores = 0

# Training
with tf.Session() as sess:
    # Run initializer
    sess.run(init_op)

    for step in range(1, maxiter+1):
        mask_idx = np.random.choice(range(num_train), batch_size)
        batch_x, batch_y = X_train[mask_idx], y_train[mask_idx]
        # Run optimization op
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if (step % display_step == 0) or (step == 1):
            # Calculate batch loss and accuracy
            loss, acc = sess.run(
                [loss_op, accuracy],
                feed_dict={X: batch_x, Y: batch_y})

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss))

    print("Optimization Finished!")

    final_scores = sess.run(accuracy, feed_dict={X: X_test,
                                      Y: y_test})

print("\nI think that: ")
for idx, pair in enumerate(X_test):
    print (pair[0], " X ",pair[1], " = ", final_scores[idx][0], "<-", y_test[idx][0])
