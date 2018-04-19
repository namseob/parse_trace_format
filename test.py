'''
import tensorflow as tf

x = tf.constant([[[1., 1., 1., 1., 1., 1., 1.], [2., 2., 2., 2., 2., 2., 2.], [3.,3.,3.,3.,3.,3.,3.]], [[3., 3., 3., 3., 3., 3., 3.], [4., 4., 4., 4., 4., 4., 4.], [5.,5.,5.,5.,5.,5.,5.]]])
print(x.shape)
print(tf.Session().run(x))

y = tf.reduce_mean(x, [1,1,1], name='pool5', keep_dims=True)
print(y.shape)
print(tf.Session().run(y))
'''




'''
import time
import datetime
from time import localtime, strftime

#print(strftime("%y/%m/%d %H:%M:%S", localtime()))

timestamp = 1463460958000
datetimeobj = datetime.datetime.fromtimestamp(timestamp/1000)
print(datetimeobj)

timestamp = time.mktime(datetimeobj.timetuple())
#print(timestamp)
'''




'''
import re

ret = re.sub("edge_\d*_", '', "edge_1_vgg_19")

print(ret)
'''
