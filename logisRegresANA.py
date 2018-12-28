import math
#import numeric
import random
#import autograd.numpy as np
#from autograd import grad
import numpy as np
import numpy.ma as ma   #maskedarrays

def sigmoid(inX):
    #return 0.5*(np.tanh(x) + 1)
    return 1.0/(1+np.exp(-inX))
def sigmoid_prime(x):
    '''derivative of sigmoid/logistic function'''
    return sigmoid(x) * (1 - sigmoid(x))
def predict(weights, inputs):
    return sigmoid(np.dot(inputs, weights))
def logistic_log_likelihoodi(xi, yi, weights):

    prediction = predict(weights, xi)
    if (prediction == 'nan'): print('predict(weights, xi) ', predict(weights, xi))
    if yi == 1:
       
        if (predict(weights, xi)==0):
            return 0.  
        else: return np.log(predict(weights, xi))
    #Yi == 0:
    elif yi == 0:
        if (predict(weights, xi)== 1.):
            return 0
        else: return (1-yi)*np.log((1 - predict(weights, xi)))
    else: 
        print("ALGO ESTRANHO!!")
        return 0

def logistic_log_likelihood(weights, X_train, Y_train):
    label_probabilities=np.array([logistic_log_likelihoodi(xi, yi, weights) for (xi, yi) in zip(X_train,Y_train)])
    print('- sum', -np.sum(label_probabilities))
    return -np.sum(label_probabilities)
def loss(weights, X_train,Y_train):  #algum erro aqui!!
    #preds = predict(weights, X_train)
    label_probabilities = 0
    
    for i, yi in enumerate(Y_train):   
        label_probabilities=label_probabilities+yi*np.dot(X_train[i], weights) -np.log(1+ np.exp(np.dot(X_train[i],weights)))

    return -(label_probabilities)

def grad_loss(weights, X_train, Y_train):
    #x, y = real(z), imag(z)
    return grad(loss, 0)(weights, X_train, Y_train) #derivative wrt. weights

def steepest_descent_auto(X_train, Y_train, alpha =0.001):
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    intercept = np.ones((X.shape[0], 1))
    X_train   = np.concatenate((intercept, X_train), axis=1) 
    weights = np.ones(X_train.shape[1])
  
    print('initial weights ', weights)
    loss_values = []

    for i in range(500):

        deriv = sigmoid_prime(np.dot(X_train, weights))
        #print('deriv ', deriv)
        error  = Y_train- predict( weights, X_train)
        l1 = error * deriv
        alpha = 4/(1.0+i)+0.0001
        nextweights = weights + np.dot(X_train.T,l1)* alpha

        weights = nextweights      
    return weights#, loss_values[-1]

################################
def logistic_reg(X_train, y, epochs, lr):   #ruim...
    xb = np.c_[np.ones((X_train.shape[0],1)),X_train]
   
    weights = np.ones(xb.shape[1]) #number of columns of xb
    for j in range(1,epochs):
        residual = sigmoid(xb @ weights) - y
        delta = (xb.T @ residual )* (1/xb.shape[0])
        weights = weights - lr* delta
    print(log_likelihood(xb, y, weights))
    return weights
def log_likelihood(xb, y, weights):
    scores = xb @ weights
    ll = (y * scores) - np.log(1+ np.exp(scores))
    return sum(ll)
##########################Neural network:
def neuralnetwork(sizes,X_train,Y_train,validation_x,validation_y,verbose=False,
                  epochs=30,mini_batch_size=10,lr=0.05,C='ce'):
    num_layers= len(sizes)
    listw = sizes[:-1] #not including the index -1 (last)
    listb = sizes[1:] # skips first element 

    biases = [np.random.randn(max(sizes)) for _ in listb]#[np.random.randn(max(listb)) for _ in listb]
    biases = np.stack(biases, axis=0)
    biases = ma.array(biases, mask=[[0]*listbi+[1]*(max(sizes)-listbi) for listbi in listb])#ma.array(biases, mask=[[0]*listbi+[1]*(max(listb)-listbi) for listbi in listb])
    weights = [np.random.randn(max(sizes),max(sizes)) for _,_ in zip(listw,listb)]#[np.random.randn(max(listb),max(listw)) for _,_ in zip(listw,listb)]
    weights = np.stack(weights, axis=0)
    #mask for weights:
    marraysw=[np.pad(np.zeros((listbi,listwi)),((0,max(sizes)-listbi),(0,max(sizes)-listwi)),'constant',constant_values=1) for listwi,listbi in zip(listw,listb)]
#[np.pad(np.zeros((listbi,listwi)), ((0, max(listb)-listbi),(0,max(listw)-listwi)), 'constant', constant_values=1) for listwi,listbi in zip(listw,listb)]
    marraysw = np.stack(marraysw, axis=0)
    weights = ma.array(weights, mask=marraysw)
    #print('biases initial ', biases)
    #print('initial weights ', weights)
    biases, weights = SGD(X_train, Y_train, epochs, mini_batch_size, lr, C, sizes, num_layers, 
        biases, weights, verbose, validation_x, validation_y)
    return biases, weights
def SGD(X_train, Y_train, epochs, mini_batch_size, lr, C, sizes, num_layers,
       biases, weights, verbose, validation_x, validation_y):
    #every epoch
    training_data =np.concatenate((X_train,np.array([Y_train]).T),axis=1) # zip(X_train,Y_train)
    for j in range(epochs):
        np.random.shuffle(training_data) #stochastic mini_batch (shuffle data)
        
        #Partition set into mini_batches
        mini_batches =  np.split(training_data, math.ceil(Y_train.shape[0]/mini_batch_size))#
                
        #feed-forward (and back) all mini_batches
        for i, minib in enumerate(mini_batches):
            lr = learning_schedule(j*math.ceil(Y_train.shape[0]/mini_batch_size) + i)
            biases, weights = update_mini_batch(minib, lr, C, sizes, num_layers,biases,weights)
            #print(i, 'biases ', biases)
            #print(i, 'weights ', weights)
        if(verbose):
            if(j % 1 ==0): 
                print('Epoch ', j, ' finished' )
                confusion = evaluate(X_train, Y_train, biases, weights)
    if (verbose): print('Training ended.')
    return biases, weights
def update_mini_batch(minibatch,lr, C,sizes,num_layers,biases,weights):
        #print('type(minibatch)', type(minibatch), minibatch.shape)
        #print('mini_batch ', minibatch)
        nmb = (minibatch).shape[0]   
        listb = sizes[1:]
        listw = sizes[:-1]
        #initialise updates with zero arrays
        nabla_b = np.dot(0.0 , (biases))   #numpy.array([[0] for _ in listb])
        nabla_w = np.dot(0.0, (weights)) #numpy.array([[0 for _ in listw] for _ in listb])
        #print('weights ', weights)
        #print('nabla_w from weights -shape ', nabla_w.shape)
        #print('nabla_b from weights -shape ', nabla_b.shape)
        for i, minib in enumerate(minibatch):
            #print('i: ',i, 'minib ', minib)
            #print('type of minib ', type(minib))
            #print('shape of minib ', minib.shape)
            #print('x = minib[:-1]', minib[:-1])
            #print(' y = minib[-1]',  minib[-1])
            x = minib[:-1]    
            y = minib[-1]    
            #backpropagation for each observation in mini_batch
            delta_nabla_b, delta_nabla_w = backprop(x, y, C,sizes,num_layers,biases,weights)
            #print(delta_nabla_b.shape, 'delta_nabla_b ', delta_nabla_b)
            #print(delta_nabla_w.shape, 'delta_nabla_w ', delta_nabla_w)
            #print('shapes dos2', delta_nabla_w[0].shape, delta_nabla_w[1].shape)
            #print(delta_nabla_b.shape, 'delta_nabla_b ', delta_nabla_b)
            #print('shapes dos2', delta_nabla_b[0].shape, delta_nabla_b[1].shape)
            #Add on deltas to nabla
            nabla_b = nabla_b + delta_nabla_b
            nabla_w = nabla_w + delta_nabla_w
        #print(type(weights[0]), 'weights[0] shape', weights[0].shape)
        #print(type(nabla_w[0]), 'nabla_w[0] shape', nabla_w[0].shape)
        #print(type(weights), 'weights ', weights)
        #print(type(nabla_w), 'weights ', weights)
        weights = weights - (lr/nmb) * nabla_w
        biases = biases - (lr/nmb) * nabla_b
        return biases, weights
def backprop(x, y, C, sizes, num_layers, biases, weights):
    listw = sizes[:-1]
    listb = sizes[1:]
    nabla_b_backprop = np.dot(0., biases)     #numpy.array([[0] for _ in listb])
    nabla_w_backprop = np.dot(0., weights)

    activation = np.array(x)           #first activation is input vector x
    activations = ma.array(np.zeros((len(sizes),max(sizes))), mask = [np.pad(np.zeros(sizesi), (0,max(sizes)-sizesi), 'constant', constant_values=1) for sizesi in sizes])
    
    zs = []                               #to store computation in each neuron
    i= 0
    activations[i][:len(activation)]  = activation
    
    for b, w in zip(biases,weights):
  
        z = np.ma.dot(w, activations[i],strict=True) +b
   
        zs.append(z)
        activation = sigmoid(z)    #activation function
        #activations.extend([activation])
        i = i +1
        activations[i][:len(activation)] = activation
    #Backwards (update gradients using errors)
    #last layer
    #print('activations: ', activations)
    delta = cost_delta(method= C, z = zs[-1],  a=activations[-1], y = y)  
    #print('delta :', delta)
    nabla_b_backprop[-1][~nabla_b_backprop[-1].mask] = delta

    if type(delta)==np.float64:
        nablaw = np.dot(delta, activations[-2])
    else:
        nablaw = np.ma.dot(delta[~delta.mask].reshape((len(delta[~delta.mask]),1)) , activations[-2].reshape((1,len(activations[-2]))), strict=True) 
    nabla_w_backprop[-1][~nabla_w_backprop[-1].mask] = nablaw[~nablaw.mask]

    if (num_layers>2):
        for k in range(2,(num_layers)):
   
            sp = sigmoid_prime(zs[-k])
 
            delta = np.ma.dot(weights[-k+1].transpose(), delta,strict=True) * sp
      
            nabla_b_backprop[- k][~nabla_b_backprop[-k].mask] = delta[~delta.mask]  
    
            nabla_w = np.ma.dot(delta[~delta.mask].reshape((len(delta[~delta.mask]),1)), activations[-k-1].reshape((1,len(activations[-k-1]))),strict=True)
            
            nabla_w_backprop[-k][~nabla_w_backprop[-k].mask] =nabla_w[~nabla_w.mask]  
        
    return nabla_b_backprop, nabla_w_backprop
def feedforward(a, biases, weights):
    for b,w in zip(biases,weights):
        a = sigmoid(np.ma.dot(w, a,strict=True) +b)    
    return a

def cost_delta(method, z, a , y):
    if(method=='ce'):      

        return np.ma.dot((a - y), sigmoid_prime(z),strict=True) #'ce' for cross-entropy loss function
    if(method=='re'):
        return 0.5*(((a-y))**2).sum()
    
def evaluater(X_test, Y_test, biases, weights):  
    """ for regression  """
    fitted_test = np.array([feedforward(x,biases,weights).compressed() for x in X_test]).flatten()
   
    R2_test = 1 - ( (fitted_test - Y_test)**2 ).sum()/(((Y_test - np.mean(Y_test)))**2).sum()
    MSE_test = np.sum((fitted_test - Y_test)**2)/len(Y_test)
    bias_test = np.sum((Y_test - np.mean(fitted_test))**2)/len(Y_test)
    var_test = np.sum((fitted_test - np.mean(fitted_test))**2)/len(Y_test)
    return R2_test, MSE_test, bias_test, var_test
        
def evaluate(X_test, Y_test, biases, weights):
    """ for classification"""
   
    test_result = [(classifyf(feedforward(x,biases,weights).compressed()), y) for (x,y) in zip(X_test,Y_test)]#
    return (sum(int(x==y) for (x,y) in test_result))/len(test_result)
def classifyf(prob):
    if prob > 0.5: return 1
    else: return 0

def simptest(weights, X_test, Y_test):
    intercept = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((intercept, X_test), axis=1) 
    errorCount = 0; numTestVec = 0.0
    
    for lineX, lineY in zip(X_test, Y_test):
        numTestVec += 1.0
        
        if int(classifyVector(lineX, weights))!= int(lineY):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print( "the error rate of this test is: %f" % errorRate)
    print("accuracy is %f" %  (1-errorRate))
    return errorRate
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


########################################end: Neural network
def in_random_order(data):
    """generator that returns the elements of data in random order"""
    indexes = [i for i, _ in enumerate(data)] # create a list of indexes
  
    random.shuffle(indexes)                   # shuffle them
    return indexes
def gradDscent(dataMat, classLabels, alpha = 0.001):

    labelMat = np.mat(classLabels).transpose()
    dataMat = np.mat(dataMat)
    m,n = np.shape(dataMat)

    maxCycles = 500
    weights = numpy.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMat*weights)
        error = (labelMat - h)
        weights = weights - alpha * dataMat.transpose()* error
    return weights
def stocGradAscentA(dataMatrix, classLabels, numIter=300):     
    '''dataMatrix is x, classLabels is y (output)'''
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones

    for j in range(numIter):
        for i, index  in enumerate(in_random_order(zip(dataMatrix, classLabels))):
            dataMatrix_i, classLabels_i = dataMatrix[index], classLabels[index]
          
            alpha = 4/(1.0+j+i)+0.0001
    
            h = sigmoid(sum(dataMatrix_i*weights))
            error = h - (classLabels_i)
  
            weights = weights - alpha * error * dataMatrix_i
    return weights
def simptest(weights, X_test, Y_test):
    intercept = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((intercept, X_test), axis=1) 
    errorCount = 0; numTestVec = 0.0
    
    for lineX, lineY in zip(X_test, Y_test):
        numTestVec += 1.0
        
        if int(classifyVector(lineX, weights))!= int(lineY):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print( "the error rate of this test is: %f" % errorRate)
    print("accuracy is %f" %  (1-errorRate))
    return errorRate
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

    


def test(X_train, Y_train, X_test, Y_test):

    trainingSet = X_train; trainingLabels = Y_train
    m, n = numpy.shape(trainingSet)
    trainWeights = sgd(numpy.array(trainingSet), trainingLabels)
  
    errorCount = 0; numTestVec = 0.0
    for lineX, lineY in zip(X_test, Y_test):
        numTestVec += 1.0
        if int(classifyVector(lineX, trainWeights))!= int(lineY):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print( "the error rate of this test is: %f" % errorRate)
    return trainWeights, errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += Test(X_train, Y_train, X_test, Y_test)
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
        

def learning_schedule(t):
    t0, t1 = 5.0, 50.0
    return t0/(t + t1)
def sgd(dataMatrix, classLabels):
    n_epochs = 200
    m,n = np.shape(dataMatrix)
    intercept = np.ones((dataMatrix.shape[0], 1))
    dataMatrix= np.concatenate((intercept, dataMatrix), axis=1) 
    weights = np.ones(n+1)   #initialize to all ones
 
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = dataMatrix[random_index:random_index+1]
            yi = classLabels[random_index:random_index+1]

            gradients = (1/m)*sum(sigmoid(sum(xi*weights))- yi )*xi    #2* xi.T.dot(xi.dot(weights)- yi)
            eta = learning_schedule(epoch*m + i)
            weights = weights - eta * gradients/m
    return weights
