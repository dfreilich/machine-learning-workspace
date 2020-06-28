import numpy as np
import theano
import theano.tensor as T

import keras.backend as K

from keras.layers import *

from AumannShapley import AumannShapley

class Explanation:
    '''
    Top level explanation object. As explanations are parameterized to include a
    layer and a quantity of interest, we provide the ability to partially 
    compute an explanation given the available parameters using chaining.

    Typical usage for explaining quantity of interest Q from layer i would be

        expl = Explanation(model).forLayer(i).for_quantity(Q)

    The constructor for an Explanation takes the following:

        model (keras.models.Model): The model we are explaining.
        attribution_method (AttributionMethod): (optional) The way in which we
            calculate the attributions. We default to AumannShapley as this is
            what we describe in the paper.
    '''

    def __init__(self, model, attribution_method=None):
        self.model = model

        if attribution_method is None:
            self.attribution_method = AumannShapley()
        else:
            self.attribution_method = attribution_method

    def for_layer(self, layer):
        '''
        Returns an ExplanationForLayer object.
        '''
        return ExplanationForLayer(self.model, layer, self.attribution_method)

    def for_input(self):
        '''
        Same as for_layer(0).
        '''
        return ExplanationForLayer(self.model, 0, self.attribution_method)


class ExplanationForLayer:
    '''
    An explanation object where the layer has been fixed.
    '''

    def __init__(self, model, layer, attribution_method=AumannShapley()):
        # Note that in keras, the model's output is a tensor representing the
        # entire model up to that point.
        self.model = model
        self.model_layer_up = model.layers[-1].output
        self.layer_activations = model.layers[layer].input
        self.layer = layer
        self.attribution_method = attribution_method

        # Some models have a separate mode for training and testing, e.g., when
        # the model contains a Dropout layer. These models need an additional
        # input to specify the mode, which causes problems when compiling the
        # model to a function.
        self.has_training_mode = True
    
    
    def for_quantity(self, Q):
        '''
        Q is the quantity of interest, expressed symbolically.
        Returns an ExplanationForQuantity object.
        '''
        return ExplanationForQuantity(
            self.model,
            self.layer, 
            self.attribution_method.compile(
                Q, 
                self.layer_activations, 
                has_training_mode=self.has_training_mode))

    '''
    In absolute and comparative, the only thing you need to do compared to the for_quantity() function is to compute Q from class ID. 
    Returns an ExplanationForQuantity object.
    '''
    def absolute(self, class1):
        # We have to figure out how to transform class1 into a theano variable, that represents the class score
        # print(self.model.layers[36])
        # print(self.model_layer_up)
        # output = self.model.layers[36].output
        Q = T.take(self.model_layer_up, class1, axis=1)
        return self.for_quantity(Q)
    

    '''
    Returns an ExplanationForQuantity object.
    '''
    def comparative(self, class1, class2):
        Q1 = T.take(self.model_layer_up, class1, axis=1)
        Q2 = T.take(self.model_layer_up, class2, axis=1)
        return self.for_quantity(Q1-Q2)



class ExplanationForQuantity:
    '''
    This is a fully-compiled explanation function that maps instances from the 
    layer corresponding to this explanation to their respective attribution, 
    counterfactual, and interpretation.

    As all the attributes of this class are compiled, this class should not be
    instantiated directly.
    '''

    def __init__(self, model, layer, attribution_method, has_training_mode=False):
        self.attribution_method = attribution_method
        if layer > 0:
            inpt = model.layers[0].input
            outpt = model.layers[layer - 1].output
            if has_training_mode:
                features = theano.function([inpt, K.learning_phase()], outpt,
                                           allow_input_downcast=True,
                                           on_unused_input='ignore')
                self.features = lambda(inp): features(inp, 0)
            else:
                self.features = theano.function([inpt], outpt)
        else:
            self.features = lambda(x): x

    
    def explain(self, instance, ignore_difference_term=True):
        '''
        Returns an ExplanationData object.
        '''
        instance = np.expand_dims(instance, axis=0)
        instance = self.features(instance)[0]
        attributions = self.attribution_method.get_attributions(instance, ignore_difference_term)

        return ExplanationData(attributions, instance)


class ExplanationData:
    '''
    This class simply holds the data associated with an explanation; it is the
    terminal object in the chained explanation object flow.

    An explanation contains the following numpy arrays:

    - attribution
    - instance
    '''

    def __init__(self, attribution, instance):
        self.attribution = attribution
        self.instance = instance

