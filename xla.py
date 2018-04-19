import tensorflow as tf

with tf.Session() as sess:
	x = tf.placeholder(tf.float32, [4])
	with tf.device("device:XLA_GPU:0"):
		y = x * x
	
	result = sess.run(y, {x: [1.5, 0.5, -0.5, -1.5]})
