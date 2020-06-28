# import cStringIO
# import urllib2
# import json

import numpy as np
from scipy.misc import imresize, imread


def image(file):
    im = imresize(imread(file), (224, 224)).astype(np.float32)
    im = im.transpose((2,0,1))

    return im

def create_file(file, urls):
    instances = []
    for url in urls:
        try:
            instances.append(
                image(cStringIO.StringIO(urllib2.urlopen(url).read())))
        except:
            print('failed url: %s' % url)
    instances = np.array(instances)
    np.save(file, instances)

    return instances

def load_cached(file,urls=None):
    try:
        # Load file.
        return np.load(file)
    except:
        # Read urls and save file.
        print('Creating file.')
        return create_file(file, urls)


def get_starfish():
    return load_cached('starfish_instances.npy')

def get_chainsaws():
    return load_cached('chainsaws.npy')

# 1: 452
def get_bonnets():
    return load_cached('bonnets.npy')

# 2: 703
def get_park_benches():
    return load_cached('park_benches.npy')

# 3: 297
def get_sloth_bears():
    return load_cached('sloth_bears.npy')

# 4: 144
def get_pelicans():
    return load_cached('get_pelicans.npy')

def get_sports_cars():
    return load_cached('get_sports_cars.npy')

def more_sports_cars():
    return load_cached('more_sports_cars.npy')

def get_convertibles():
    return load_cached('get_convertibles.npy')

def get_cats():
    cat_links_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02121620'
    data = urllib2.urlopen(cat_links_url).read()
    urls = data.split()
    return load_cached('cats.npy', urls)

def get_dogs():
    #Gets Terriers
    dog_links_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02095314'
    data = urllib2.urlopen(dog_links_url).read()
    urls = data.split()
    return load_cached('dogs.npy', urls)

def get_more_dogs():
    poodle_links_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02113335'
    data = urllib2.urlopen(poodle_links_url).read()
    urls = data.split()
    return load_cached('more_dogs.npy', urls)

_imnet_random_classes = [
    get_chainsaws(),
    get_bonnets(),
    get_park_benches(),
    get_sloth_bears(),
    get_pelicans()
]

_imnet_random_class_labels = [
    491,
    452,
    703,
    297,
    144
]