
from AttributionMethod import AttributionMethod
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import keras.backend as K
import numpy as np
from scipy.misc import imresize, imread, imsave
import theano
import theano.tensor as T

class AumannShapley(AttributionMethod):

    def __init__(self, resolution=50):
        """
        resolution: The number of middle steps between baseline and target. 
        For example: resolution of 1 means only sample at the instance,
        for simple Taylor use a resolution of 2 (origin and point)
        for Integrated gradients/Influence directed explanations use 50   
        """
        AttributionMethod.__init__(self)
        self.res = resolution

    def compile(self, Q, inpt, has_training_mode=False):
        grad = theano.grad(Q.sum(), wrt=inpt)
        # print(grad.eval())

        if has_training_mode:
            # If the model has a special training mode, we need to pass the
            # learning phase as an additonal parameter, which always takes
            # value 0 (test mode).
            grad_f = theano.function([inpt, K.learning_phase()], grad, allow_input_downcast=True, on_unused_input='ignore')
            self.dF = lambda(inp): grad_f(inp, 0)
        else:
            self.dF = theano.function([inpt], grad)
        return self

    def get_attributions(self, instance, ignore_difference_term, baseline=None):
        """
        -Inputs:
        instance: the instance to get attributions from. 
            (If we are visualizing/doing Integrated Gradients, the instance is an image; if we are calculating the influence of 
            hidden layers, the input is a neuron or a slice of a layer)
            ignore_difference_term: Whether we mutiply the gradients by activations. (We need that term for IG for completeness                axiom, not necessarily for influence-directed explanations)
        baseline: baseline for doing integration of incremental gradients. If not provided, set with zero. 
        -Output:
        Attribution values
        """
        baseline = np.zeros(instance.shape)
        scaled_inps = [baseline + (float(i)/self.res)*(instance-baseline) for i in range(1, self.res+1)]
        grads = self.dF(scaled_inps)
        # print(grads.shape)
        # print(grads[:, 441, 12, 16])
        if ignore_difference_term:
            integrated_gradients = np.average(grads, axis=0)
        else:
            integrated_gradients = (instance - baseline) * np.average(grads, axis=0)
        return integrated_gradients
