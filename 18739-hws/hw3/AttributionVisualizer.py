'''
    Directly imported from Klas
'''


import numpy as np

from scipy.misc import imresize, imread, imsave
from matplotlib import pyplot as plt

class AttributionMask():
    '''
    Visualize attributions on top of an image. When the image is shown the
    average attribution across the channels is used as the alpha value
    '''

    def visualize(self, explanation,  image_bound , plot = True, outfile = None):
        '''
        Input:
            explanation: ExplanationData Object
            image_bound: the image scale. [0 255] for images in imagenet. 
        '''
        attributions = explanation.attribution

        # Take the mean across the channels.
        attributions = np.mean(np.abs(attributions), axis=0)
        attributions = np.clip(attributions / np.percentile(attributions, 99), 0,1)
        instance = (explanation.instance -image_bound[0])/float(image_bound[1]-image_bound[0]) 

        vis = instance * [attributions, attributions, attributions]
        if outfile!=None:
             imsave(out_file, vis.transpose(1,2,0))
        if plot:
            plt.imshow(np.array(vis.transpose(1,2,0)))
        return vis.transpose(1,2,0)

class VisualizerTiler:
    '''
    Visualize multiple images side by side
    '''
    def __init__(self, attribution_visualizer):
        self.attribution_visualizer = attribution_visualizer

    def visualize(self, explanations, image_bound,shape=None):
        if shape is None:
            shape = (1, len(explanations))

        visualizations = []
        for explanation in explanations:
            visualizations.append(
                self.attribution_visualizer.visualize(explanation, image_bound,plot=False))
        n, w, w, c = np.array(visualizations).shape

        out = np.zeros((w * shape[0], w * shape[1], c))
        for i in range(shape[0]):
            for j in range(shape[1]):
                out[w*i:w*(i+1), w*j:w*(j+1), :] = (
                    visualizations[i*shape[1] + j])
        plt.imshow(out)
        return out, visualizations
