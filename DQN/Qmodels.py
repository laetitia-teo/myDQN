import tensorflow as tf
from ops import dense

def one_hidden_mlp(inpt, units, outsize):
    """
    Fully connected model, with one hidden layer of length units and relu
    nonlinearity.
    
    Args:
        - inpt (Tensor) : input of the model;
        - units (int) : numberr of hidden units.
    
    Returns:
        - output (Tensor) : output of the model.
    """
    with tf.variable_scope("one_hidden"):
        hidden = tf.nn.relu(dense(inpt, units=units, name="dense1"))
        output = dense(hidden, units=outsize, name="dense2")
    return output


