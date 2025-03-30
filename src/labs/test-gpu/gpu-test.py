import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

if tf.config.list_physical_devices("GPU"):
    print("GPU is available")
    # Try a simple GPU operation
    with tf.device("/GPU:0"):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print(c)
else:
    print("GPU is NOT available")
