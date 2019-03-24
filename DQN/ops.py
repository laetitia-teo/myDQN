import tensorflow as tf

def conv2d(X, size, num_f, name):
    """
    Conv2D layer
    
    Args :
        - X_in (Tensor): input;
        - size (Integer) : size (height and width) of the filters;
        - num_f (Integer): number of filters;
    Returns :
        - X_out (Tensor): output;
    """
    with tf.variable_scope(name):
        init = tf.truncated_normal_initializer(stddev=0.02)
        W = tf.get_variable('W', [size, size, X.shape[-1], num_f], initializer=init)
        b = tf.get_variable('b', [num_f], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(X, W, strides=[1, 2, 2, 1], padding="SAME") + b
    return conv

def deconv2d(X, size, output_shape, name):
    """
    Transpose of Conv2D layer, improperly called deconvolution layer
    
    Args :
        - X (Tensor): input;
        - size (Integer) : size (height and width) of the filters;
        - output_shape (list of Integers): shape of the output Tensor;
        - name (str): name of the operation;
    Returns :
        - X_out (Tensor): output;
    """
    with tf.variable_scope(name):
        init = tf.truncated_normal_initializer(stddev=0.02)
        W = tf.get_variable('W', 
                            [size, size, output_shape[-1], X.shape[-1]],
                            initializer=init)
                            
        b = tf.get_variable('b',
                            [output_shape[-1]],
                            initializer=tf.constant_initializer(0.0))
        # check this !
        deconv = tf.nn.conv2d_transpose(X, W, output_shape, [1, 2, 2, 1]) + b
        
    return deconv

def dense(X, units, name):
    """
    Fully-connected layer
    
    Args :
        - X (Tensor): Input tensor;
        - units (Integer) : number of hidden units;
        - name (str) : name of the operation;
    """
    with tf.variable_scope(name):
        init = tf.random_normal_initializer(stddev=0.2)
        W = tf.get_variable('W', [X.shape[-1], units], initializer=init)
        b = tf.get_variable('b', [units], initializer=tf.constant_initializer(0.0))
        dense = tf.matmul(X, W) + b
    return dense




























