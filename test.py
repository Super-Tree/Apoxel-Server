import tensorflow as tf

indices = tf.constant([[4,1], [1,5], [1,1], [3,2]])
updates = tf.constant([[4,1], [4,3], [3,1], [4,5]])
shape = tf.constant([6,6,2])

scatter = tf.scatter_nd(indices, updates, shape)
with tf.Session() as sess:
    print(sess.run(scatter))