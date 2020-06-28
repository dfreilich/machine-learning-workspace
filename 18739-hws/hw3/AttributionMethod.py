'''
    Directly imported from Klas
'''

class AttributionMethod:
    '''
    Interface for an attribution method class.

    An attribution method takes a symbolic representation of the explanation layer
    and the network, F+, that is, the network from the explanation layer up, and
    returns a tensor in the same dimension as the explanation layer that represents
    the influence of 
    '''
    
    def __init__(self, **kwargs):
        pass

    
    def compile(self, Q, inpt, has_training_mode=False):
        '''
        This should be implemented by subclasses of this class.

        - Q is the quantity of interest function. This is a tensor
        - inpt is the input variable to the quantity of interest.

        Some models have separate behavior for training, which causes problems when
        copiling the Q function. Use has_training_mode in this case so the
        AttributionMethod can handle this case.
        '''
        raise NotImplementedError

    '''

    Returns 
    '''
    def get_attributions(self, instance, ignore_difference_term, **kwargs):
        raise NotImplementedError
