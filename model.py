import tensorflow as tf


# spectrogram based Unet architecture

def conv_bat_relu(X,filters):
    out=tf.layers.conv2d(inputs=X, filters=filters, kernel_size=5, strides=2)
    out = tf.layers.batch_normalization(out)
    out=tf.nn.leaky_relu(out)
    return out

def deconv_bat_relu(X,filters):
    out=tf.layers.conv2d_transpose(X,filters=filters,kernel_size=5,strides=2)
    out=tf.layers.batch_normalization(out)
    out=tf.nn.relu(out)
    return out

def spec_unet(X,Y,keep_prob):
    conv1=conv_bat_relu(X,16)
    conv2=conv_bat_relu(conv1,32)
    conv3 =conv_bat_relu(conv2,64)
    conv4 = conv_bat_relu(conv3,128)
    conv5 = conv_bat_relu(conv4,256)
    conv6 = conv_bat_relu(conv5,512)

    deconv1=deconv_bat_relu(conv6,256)
    deconv1=tf.nn.dropout(deconv1,keep_prob)
    deconv2 = deconv_bat_relu(tf.concat([deconv1,conv6],1), 128)
    deconv2 = tf.nn.dropout(deconv2, keep_prob)
    deconv3 = deconv_bat_relu(tf.concat([deconv2,conv5],1), 64)
    deconv3 = tf.nn.dropout(deconv3, keep_prob)
    deconv4 = deconv_bat_relu(tf.concat([deconv3,conv4],1), 32)
    deconv5 = deconv_bat_relu(tf.concat([deconv4,conv3],1), 16)
    logits=deconv5*X
    
    return logits
# implementation of W-shaped architecture

def spec_wnet(X,Y):
    pass
def unet(X,Y):
    pass
def wnet(X,Y):
    pass