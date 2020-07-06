import numpy as np
from input_data import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
def cross_entropy(label, prob):
    nums =label.shape[0]
    imask = tf.equal(prob, 0.0)
    prob = tf.where(imask, 1e-10, prob)
    loss = -tf.math.reduce_sum(label * tf.math.log(prob))/nums
    return loss
def test_accuracy(net1,net2, images, labels):
    nums = int(labels.shape[0])
    fully_input =net1(images)
    fully_output =net2(fully_input)
    cost = cross_entropy(labels, fully_output)
    cost=cost.numpy()
    lab = tf.argmax(labels, axis=1)
    pred = tf.argmax(fully_output, axis=1)
    acc =(tf.reduce_sum(tf.cast(tf.equal(pred, lab), tf.float32))) / nums
    acc =acc.numpy()
    return (acc, cost)
mnist=mnist()
x_train = mnist.train_down_sample.xs.astype(np.float32)
y_train = mnist.train_down_sample.ys.astype(np.float32)
x_train = tf.convert_to_tensor(x_train,dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train,dtype=tf.float32)
x_test = mnist.test_down_sample.xs.astype(np.float32)
y_test = mnist.test_down_sample.ys.astype(np.float32)
x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test,dtype=tf.float32)
import time
import tensorflow as tf
num_of_neurons = 512
ITER =100
linear_r=np.zeros(ITER)
train_acc=np.zeros(ITER)
train_cost=np.zeros(ITER)
test_acc=np.zeros(ITER)
test_cost=np.zeros(ITER)
rho =10
tau =10000
initializer=RandomNormal(mean=0.0, stddev=0.1, seed=0)
net1 = Sequential([
Dense(num_of_neurons, activation='relu',kernel_initializer=initializer,input_shape=(196,)),
Dense(num_of_neurons, activation='relu',kernel_initializer=initializer),
Dense(num_of_neurons, activation='relu',kernel_initializer=initializer),
Dense(num_of_neurons, activation='relu',kernel_initializer=initializer),
Dense(num_of_neurons, activation='relu',kernel_initializer=initializer)
])
net1.compile(Adam(lr=rho/tau), loss='mse', metrics=['mse'])
input =net1.input
output =net1.layers[4].output
fun =K.function(input,output)
q1=tf.convert_to_tensor(fun(x_train),dtype=tf.float32)
p2=tf.convert_to_tensor(q1,dtype=tf.float32)
u1=tf.zeros_like(p2)
net2 = Sequential([
Dense(num_of_neurons, activation='relu',kernel_initializer=initializer,input_shape=(num_of_neurons,)),
Dense(num_of_neurons, activation='relu',kernel_initializer=initializer),
Dense(num_of_neurons, activation='relu',kernel_initializer=initializer),
Dense(num_of_neurons, activation='relu',kernel_initializer=initializer),
Dense(10, activation='softmax',kernel_initializer=initializer)])
net2.compile(Adam(lr=rho/tau), loss='categorical_crossentropy', metrics=['accuracy'])
net1_last = tf.keras.models.clone_model(net1)
net2_last = tf.keras.models.clone_model(net2)
for i in range(ITER):
    print("i=", i)
    net1_last.set_weights(net1.get_weights())
    net1.fit(
        x_train,
        q1,
        steps_per_epoch=500,
        epochs=1,
        verbose=0)
    net2_last.set_weights(net2.get_weights())
    net2.fit(
        p2,
        y_train,
        steps_per_epoch=500,
        epochs=1,
        verbose=0)
    train_acc[i], train_cost[i] = test_accuracy(net1, net2, x_train, y_train)
    if i > 0 and train_cost[i] - train_cost[i - 1] > 0.1:
        net1.set_weights(net1_last.get_weights())
        net2.set_weights(net2_last.get_weights())
        train_acc[i], train_cost[i] = test_accuracy(net1, net2, x_train, y_train)
    with tf.GradientTape() as tape:
        tape.watch(p2)
        outputs = net2(p2)
        obj = cross_entropy(y_train, outputs) + rho / 2 * tf.reduce_sum((p2 - q1) * (p2 - q1)) + tf.linalg.trace(
            tf.matmul(tf.transpose(u1), p2 - q1))
    grads = tape.gradient(obj, p2)
    p2 = p2 - grads / tau
    q1 = (rho * fun(x_train) + rho * p2 + u1) / (2 * rho)
    u1 = u1 + rho * (p2 - q1)
    print("training accuracy:", train_acc[i])
    print("training cost:", train_cost[i])
    test_acc[i], test_cost[i] = test_accuracy(net1, net2, x_test, y_test)
    print("test accuracy:", test_acc[i])
    print("test cost:", test_cost[i])

