import tensorflow as tf


def inference(size=None, training=True, weigh_decay=0.0):
    if size is None:
        size = [256, 256, 100]
    reg = tf.keras.regularizers.l2(weigh_decay)
    x = tf.keras.layers.Input(size, name='Entrada') 
    net = tf.keras.layers.Conv2D(64, 3, strides=2, kernel_regularizer=reg,padding='SAME', activation='relu', name='dis_1')(x)
    net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(net)
    net = tf.keras.layers.Conv2D(128, 3, strides=2, kernel_regularizer=reg, padding='SAME', activation='relu', name='dis_2')(net)
    net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(net)
    
    # net = attention(net, 128, scope='att1')
    
    net = tf.keras.layers.Conv2D(256, 3, strides=2, kernel_regularizer=reg, padding='SAME', activation='relu', name='dis_3')(net)
    net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(net)
    net = tf.keras.layers.Conv2D(512, 3, strides=2, kernel_regularizer=reg, padding='SAME', activation='relu', name='dis_4')(net)
    net = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.995, epsilon=0.001, scale=True, trainable=training)(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid')(net)
    
    return tf.keras.Model(x, net, name='discriminator_lambda')