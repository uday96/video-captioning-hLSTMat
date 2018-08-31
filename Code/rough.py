import tensorflow as tf
import numpy as np
import utils

vocab_size = 10
embedding_size = 7
batch_size = 2
num_steps = 5  #auto-set?


X = tf.placeholder(tf.int64, shape=(batch_size,num_steps))
Y = tf.placeholder(tf.int64, shape=(num_steps,batch_size))

# word embeddings matrix and lookup
with tf.device("/cpu:0"):
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
    # input_.input_data shape = batch_size*num_steps
    # inputs shape = batch_size*num_steps*embedding_size
    inputsx = tf.nn.embedding_lookup(embedding, X)
    inputsy = tf.nn.embedding_lookup(embedding, Y)

x = np.array([[0,2,3,1,4],[3,1,0,0,2]])
y = np.array([[0,3],[2,1],[3,0],[1,0],[4,2]])
wemb = utils.norm_weight(vocab_size, embedding_size)

sess = tf.InteractiveSession()
# ansx = inputsx.eval(feed_dict={X:x,embedding:wemb})
# ansy = inputsy.eval(feed_dict={Y:y,embedding:wemb})

# ip = tf.placeholder(tf.float32, shape=(64,None,2048), name='ctx')
# ipmask = tf.placeholder(tf.float32, shape=(64,None), name='ctx_mask')
# counts = tf.expand_dims(tf.reduce_sum(ipmask, 1), 1)
# ctx_mean = tf.reduce_sum(ip, 1) / counts
# ans = ctx_mean.eval(feed_dict={ipmask:ctx_mask,ip:ctx})

vec = tf.placeholder(tf.float32, shape=(2,3))
vectanh = eval('tf.tanh')(vec)
ans = vectanh.eval(feed_dict={vec:})
sess.close()