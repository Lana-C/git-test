
import os
import os.path
import sys
import numpy as np
import tensorflow as tf
import keras

LOGDIR = sys.argv[1]
learning_rate = float(sys.argv[2])
n_epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])

# Load the data   (default load directory:  ~/.keras/datasets)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Data transformation
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
        
# One hot encoding for labels
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Function for creating mini-batches
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


# Define a simple convolutional layer
def conv_layer(input, channels_in, channels_out, name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([5, 5, channels_in, channels_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# Define fully connected layer
def fc_layer(input, channels_in, channels_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="B")
    act = tf.matmul(input, w) + b
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act

# Define hyperparamater string
def hparam_string(learning_rate, n_epochs, batch_size):
    return "lr_%s,e_%s,mb_%s" % (learning_rate, n_epochs, batch_size)


# Build model 
def mnist_model(learning_rate, n_epochs, batch_size):
    tf.reset_default_graph()
    sess = tf.Session()
    
    # Create hyperparameter string
    hparam = hparam_string(learning_rate, n_epochs, batch_size)
    
    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    
    # Create the network
    conv1 = conv_layer(x_image, 1, 32, "conv1")
    conv2 = conv_layer(conv1, 32, 64, "conv2")
    flattened = tf.reshape(conv2, [-1, 7*7*64])
    
    fc1 = fc_layer(flattened, 7*7*64, 1024, "fc1")
    relu = tf.nn.relu(fc1)
    logits = fc_layer(relu, 1024, 10, "fc2")
    
    # Compute cross entropy as loss function
    with tf.name_scope("xent"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", cross_entropy)
    
    # Use an AdamOptimizer to train the network
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
    # Compute the accuracy
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
        tf.summary.scalar("accuracy", accuracy)
    
    # Merge all summaries for tensorboard
    merged_summary = tf.summary.merge_all()
    
    # Initiate saver to save the model 
    saver = tf.train.Saver()
    
    # Initialize all variables 
    sess.run(tf.global_variables_initializer())
    
    # Initiate writer for tensorboard and add current graph
    writer = tf.summary.FileWriter(LOGDIR + "tf_log/" + hparam)
    writer.add_graph(sess.graph)
    
    # Train the model and save results
    for epoch in range(n_epochs):
        print ("Starting epoch %s" %epoch)
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(train_step, feed_dict={x: X_batch, y: y_batch})

        s = sess.run(merged_summary, feed_dict={x: X_batch, y: y_batch})
        writer.add_summary(s, epoch)
        
        acc_batch = accuracy.eval(feed_dict={x: X_batch, y: y_batch}, session=sess)
        acc_test = accuracy.eval(feed_dict={x: X_test, y: y_test}, session=sess)
        print("Epoch:", epoch, "Last batch accuracy:", acc_batch, " Test accuracy:", acc_test)

        saver.save(sess, os.path.join(LOGDIR, "ckpt/"+ hparam + "/model.ckpt"), epoch)


def main():
    
    # Run the model 
    mnist_model(learning_rate, n_epochs, batch_size)

    print('Done training!')
    print('Run `tensorboard --logdir=%stf_log` to see the results.' % LOGDIR)
    print('Checkpoints are located in %sckpt' % LOGDIR)


if __name__ == '__main__':
    main()

