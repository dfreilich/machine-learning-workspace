import numpy as np
import cnn_lenet
import pickle
import copy
import random
import matplotlib as mp
import matplotlib.pyplot as plt
import math

def get_lenet():
  """Define LeNet

  Explanation of parameters:
  type: layer type, supports convolution, pooling, relu
  channel: input channel
  num: output channel
  k: convolution kernel width (== height)
  group: split input channel into several groups, not used in this assignment
  """

  layers = {}
  layers[1] = {}
  layers[1]['type'] = 'DATA'
  layers[1]['height'] = 28
  layers[1]['width'] = 28
  layers[1]['channel'] = 1
  layers[1]['batch_size'] = 1

  layers[2] = {}
  layers[2]['type'] = 'CONV'
  layers[2]['num'] = 20
  layers[2]['k'] = 5
  layers[2]['stride'] = 1
  layers[2]['pad'] = 0
  layers[2]['group'] = 1

  layers[3] = {}
  layers[3]['type'] = 'POOLING'
  layers[3]['k'] = 2
  layers[3]['stride'] = 2
  layers[3]['pad'] = 0

  layers[4] = {}
  layers[4]['type'] = 'CONV'
  layers[4]['num'] = 50
  layers[4]['k'] = 5
  layers[4]['stride'] = 1
  layers[4]['pad'] = 0
  layers[4]['group'] = 1

  layers[5] = {}
  layers[5]['type'] = 'POOLING'
  layers[5]['k'] = 2
  layers[5]['stride'] = 2
  layers[5]['pad'] = 0

  layers[6] = {}
  layers[6]['type'] = 'IP'
  layers[6]['num'] = 500
  layers[6]['init_type'] = 'uniform'

  layers[7] = {}
  layers[7]['type'] = 'RELU'

  layers[8] = {}
  layers[8]['type'] = 'LOSS'
  layers[8]['num'] = 10
  return layers


def trainNet():
  # define lenet
  layers = get_lenet()

  # load data
  # change the following value to true to load the entire dataset
  fullset = True
  print("Loading MNIST Dataset...")
  xtrain, ytrain, xval, yval, xtest, ytest = cnn_lenet.load_mnist(fullset)
  print("MNIST Dataset Loading Complete!\n")

  xtrain = np.hstack([xtrain, xval])
  ytrain = np.hstack([ytrain, yval])
  m_train = xtrain.shape[1]

  # cnn parameters
  batch_size = 64
  mu = 0.9
  epsilon = 0.01
  gamma = 0.0001
  power = 0.75
  weight_decay = 0.0005
  w_lr = 1
  b_lr = 2

  test_interval = 100
  display_interval = 100
  snapshot = 5000
  max_iter = 10000
  # Lets it run the entire way

  # initialize parameters
  print("Initializing Parameters...")
  # You can make the params your params, and not the initialized ones, in order to visualize the results
  params = cnn_lenet.init_convnet(layers)
  param_winc = copy.deepcopy(params)
  print("Initilization Complete!\n")

  for l_idx in range(1, len(layers)):
    param_winc[l_idx]['w'] = np.zeros(param_winc[l_idx]['w'].shape)
    param_winc[l_idx]['b'] = np.zeros(param_winc[l_idx]['b'].shape)

  # learning iterations
  random.seed(100000)
  indices = range(m_train)
  random.shuffle(indices)

  train_cost = np.array([])
  train_accuracy = np.array([])
  test_cost = np.array([])
  test_accuracy = np.array([])

  print("Training Started. Printing report on training data every " + str(display_interval) + " steps.")
  print("Printing report on test data every " + str(test_interval) + " steps.\n")
  for step in range(max_iter):
    # get mini-batch and setup the cnn with the mini-batch
    start_idx = step * batch_size % m_train
    end_idx = (step+1) * batch_size % m_train
    if start_idx > end_idx:
      random.shuffle(indices)
      continue
    idx = indices[start_idx: end_idx]

    [cp, param_grad] = cnn_lenet.conv_net(params,
                                          layers,
                                          xtrain[:, idx],
                                          ytrain[idx], True)
    # True there is to get backtracking, but you can just use it for forward, to visualize
    # You have to make the function return output for you, so that you can reshape it into an image matrix, to show the image

    # we have different epsilons for w and b
    w_rate = cnn_lenet.get_lr(step, epsilon*w_lr, gamma, power)
    b_rate = cnn_lenet.get_lr(step, epsilon*b_lr, gamma, power)
    params, param_winc = cnn_lenet.sgd_momentum(w_rate,
                           b_rate,
                           mu,
                           weight_decay,
                           params,
                           param_winc,
                           param_grad)

    # display training loss
    if (step+1) % display_interval == 0:
      print 'training_cost = %f training_accuracy = %f' % (cp['cost'], cp['percent']) + ' current_step = ' + str(step + 1)
      train_cost = np.append(train_cost, cp['cost'])
      train_accuracy = np.append(train_accuracy, cp['percent'])

    # display test accuracy
    if (step+1) % test_interval == 0:
      layers[1]['batch_size'] = xtest.shape[1]
      cptest, _ = cnn_lenet.conv_net(params, layers, xtest, ytest, False)
      layers[1]['batch_size'] = 64
      print 'test_cost = %f test_accuracy = %f' % (cptest['cost'], cptest['percent']) + ' current_step = ' + str(step + 1) + '\n'
      test_cost = np.append(test_cost, cptest['cost'])
      test_accuracy = np.append(test_accuracy, cptest['percent'])

    # save params peridocally to recover from any crashes
    if (step+1) % snapshot == 0:
      pickle_path = 'lenet.mat'
      pickle_file = open(pickle_path, 'wb')
      pickle.dump(params, pickle_file)
      pickle_file.close()

    # Saves params at 30 for Question 4
    if (step+1) == 30:
      pickle_path = 'lenetAt30Iterations.mat'
      pickle_file = open(pickle_path, 'wb')
      pickle.dump(params, pickle_file)
      pickle_file.close()

    if (step+1) == max_iter:
      np.savetxt('trainCost.txt', train_cost)
      np.savetxt('trainAccuracy.txt', train_accuracy)
      np.savetxt('testCost.txt', test_cost)
      np.savetxt('testAccuracy.txt', test_accuracy)
      # np.savetxt('costsStacked.txt', np.column_stack(train_cost, test_cost))
      # np.savetxt('accuracyStacked.txt', np.column_stack(train_accuracy, test_accuracy))
      pickle_path = 'lenetAt10000Iterations.mat'
      pickle_file = open(pickle_path, 'wb')
      pickle.dump(params, pickle_file)
      pickle_file.close()

    if (step) == max_iter:
      np.savetxt('trainCost1.txt', train_cost)
      np.savetxt('trainAccuracy1.txt', train_accuracy)
      np.savetxt('testCost1.txt', test_cost)
      np.savetxt('testAccuracy1.txt', test_accuracy)
      # np.savetxt('costsStacked1.txt', np.column_stack(train_cost, test_cost))
      # np.savetxt('accuracyStacked1.txt', np.column_stack(train_accuracy, test_accuracy))
      pickle_path = 'lenetAtMAXPLUSONEIterations.mat'
      pickle_file = open(pickle_path, 'wb')
      pickle.dump(params, pickle_file)
      pickle_file.close()

def visualizeOutputOfSecondLayer(givenParams):
  # define lenet
  layers = get_lenet()

  # load data
  # change the following value to true to load the entire dataset
  fullset = True
  print("Loading MNIST Dataset...")
  xtrain, ytrain, xval, yval, xtest, ytest = cnn_lenet.load_mnist(fullset)
  print("MNIST Dataset Loading Complete!\n")

  xtrain = np.hstack([xtrain, xval])
  ytrain = np.hstack([ytrain, yval])
  m_train = xtrain.shape[1]

  # cnn parameters
  batch_size = 1

  # initialize parameters
  print("Initializing Parameters from given params")
  # You can make the params your params, and not the initialized ones, in order to visualize the results
  params = givenParams
  param_winc = copy.deepcopy(params)
  print("Initilization Complete!\n")

  for l_idx in range(1, len(layers)):
    param_winc[l_idx]['w'] = np.zeros(param_winc[l_idx]['w'].shape)
    param_winc[l_idx]['b'] = np.zeros(param_winc[l_idx]['b'].shape)

  # learning iterations
  random.seed(100000)
  indices = range(m_train)
  random.shuffle(indices)
  max_iter = 1

  # get mini-batch and setup the cnn with the mini-batch
  for step in range(max_iter):
    # get mini-batch and setup the cnn with the mini-batch
    start_idx = step * batch_size % m_train
    end_idx = (step + 1) * batch_size % m_train
    if start_idx > end_idx:
      random.shuffle(indices)
      continue
    idx = indices[start_idx: end_idx]

    [cp, param_grad, output] = cnn_lenet.conv_net(params,
                                          layers,
                                          xtrain[:, 0:1],
                                          ytrain[0:1], False)

    # conv_out = output[2]['data'].reshape(24,24,20)
    # plotNNFilter(conv_out)
    conv_out = output[1]['data'].reshape(28,28,1)
    plotNNFilter(conv_out)

    # for j in range(20):
    #   plt.figure()
    #   print j
    #   plt.imshow(conv_out[:,:,j], cmap="gray")
    #   plt.show()


    # plotNNFilter(additionalReturn['data'].reshape(24,24,20))
    # plotNNFilter(additionalReturn['data'].reshape())

    # You have to make the function return output for you, so that you can reshape it into an image matrix, to show the image

def plotNNFilter(units):
    filters = 1
    plt.figure(1, figsize=(24,24))
    n_columns = 4
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i+1))
        plt.imshow(units[:,:,i], interpolation="nearest", cmap="gray")
        plt.pause(100)

def visualizeCost():
  train_Cost = np.genfromtxt('trainCost.txt')
  test_Cost = np.genfromtxt('testCost.txt')

  plt.gca().set_color_cycle(['red', 'green'])
  plt.plot(train_Cost, train_Cost)
  plt.plot(train_Cost, test_Cost)
  # plt.axis([0,1,0,10000])

  plt.legend(['Train Cost', 'Test Cost'], loc='upper left')

  plt.show()

def visualizeAccuracy():
  train_Cost = np.genfromtxt('trainAccuracy.txt')
  test_Cost = np.genfromtxt('testAccuracy.txt')

  plt.gca().set_color_cycle(['red', 'green'])
  plt.plot(train_Cost, train_Cost)
  plt.plot(train_Cost, test_Cost)
  # plt.axis([0,1,0,10000])

  plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')

  plt.show()

if __name__ == '__main__':
  # params = pickle.load(open("lenetAt30Iterations.mat", "rb"))
  # params2 = pickle.load(open("lenetAt10000Iterations.mat", "rb"))
  # visualizeOutputOfSecondLayer(params2)
  visualizeAccuracy()

  # print(params2)
  # visualizeOutputOfSecondLayer()
  # main()

