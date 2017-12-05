import numpy as np
import tensorflow as tf

number_attri = 8

file = open("data.txt", "r")
training_data = []
X_list = []
Y_list = []

#read lines from file
for line in file:

	ele_array  = line.split()
	X_list.append([int(ele) for ele in ele_array[:number_attri]])
	Y_list.append([int(ele_array[-1])])

X = np.array(X_list).astype(np.float32)
Y = np.array(Y_list).astype(np.float32)

m = X.shape[0]
j = 0
b = 0
learning_rate_bgd = 0.000007825
dw = 0
db = 0
w = np.zeros((number_attri, 1))

print("\nBatch Gradient Descent\n")

for iter in range(10000):
	#forward path
	z = X.dot(w) + b
	a = z
	j = (1/m)*0.5*np.sum((a-Y)*(a-Y))#1

	#backward path
	dz = a-Y
	dw = (1/m)*(X.T.dot(dz))
	db = (1/m)*np.sum(dz)
	w -= learning_rate_bgd*dw
	b -= learning_rate_bgd*db
	if iter%2000==0:
		print(iter, j)

print("w of BGD:", w)

print("\nStochastic Gradient Descent\n")

w = np.zeros((number_attri, 1))
num_steps = 5001
batch_size = 20
learning_rate_sgd = 0.00000099

for step in range(num_steps):

	offset = (step * batch_size) % (m - batch_size)
	X_batch = X[offset:offset + batch_size,:]
	Y_batch = Y[offset:offset + batch_size,:]

	Z = X_batch.dot(w) + b
	a = Z
	j = (1/m)*0.5*np.sum((a-Y_batch)*(a-Y_batch))#1

	#backward path
	dz = a-Y_batch
	dw = (1/m)*(X_batch.T.dot(dz))
	db = (1/m)*np.sum(dz)
	w -= learning_rate_sgd*dw
	b -= learning_rate_sgd*db
	if step%1000==0:
		print(step, j)
test_Z = X.dot(w) + b
print("\nw of SGD:", w)

def accuracy(A, Y):
    num_agreements =0
    for i in range(Y.shape[0]):
    	if (abs(A[i]-Y[i])/Y[i]) < 0.1:
    		num_agreements+=1
    return num_agreements / Y.shape[0] * 100

#use tensorflow with 1 hidden flow
num_hidden_nodes = 10
tf_X = tf.constant(X)
tf_Y = tf.constant(Y)
tf_w1 = tf.Variable(tf.truncated_normal((number_attri, num_hidden_nodes)))
tf_w2 = tf.Variable(tf.truncated_normal((num_hidden_nodes, 1)))
tf_b1 = tf.Variable(tf.zeros((1, num_hidden_nodes)))
tf_b2 = tf.Variable(tf.zeros((1, 1)))

tf_Z1 = tf.matmul(tf_X, tf_w1) + tf_b1
tf_Z2 = tf.matmul(tf_Z1, tf_w2) + tf_b2

tf_J = tf.reduce_mean(tf.losses.absolute_difference(labels=tf_Y, predictions = tf_Z2))

optimizer = tf.train.GradientDescentOptimizer(0.00000775).minimize(tf_J)

tf_Z1_test = tf.matmul(tf_X, tf_w1) + tf_b1
tf_Z2_test = tf.matmul(tf_Z1_test, tf_w2) + tf_b2

session = tf.InteractiveSession()

tf.global_variables_initializer().run()
print("Initialized")

for iter in range(3000):
	_, J, A = session.run([optimizer, tf_J, tf_Z2])
	if (iter%500 == 0): print(iter, J)
tf_Z2 = tf_Z2.eval()

for num in range(len(tf_Z2)):
	print(num+1, tf_Z2[num])
print("\naccuracy of bgd", accuracy(z, Y), "%")
print("accuracy of sgd", accuracy(test_Z, Y), "%")
print("accuracy of sgd-tensorflow with one hidden layer", accuracy(tf_Z2, Y), "%")