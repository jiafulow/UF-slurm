import tensorflow as tf
print(tf.__version__)

# __________________________________________________________
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# __________________________________________________________
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
