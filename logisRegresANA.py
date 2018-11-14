import math
import numeric
import random
import autograd.numpy as np
from autograd import grad
import numpy
import pandas
def sigmoid(inX):
    #return 0.5*(np.tanh(x) + 1)
    return 1.0/(1+np.exp(-inX))
def sigmoid_prime(x):
    '''derivative of sigmoid/logistic function'''
    
    return sigmoid(x) * (1 - sigmoid(x))
def predict(weights, inputs):
    return sigmoid(np.dot(inputs, weights))
def logistic_log_likelihoodi(xi, yi, weights):
    #print('xi: ', xi,'yi: ', yi, 'weights: ', weights)
    #print('predict: ', predict(weights, xi))
    #print( math.log(1-predict(weights, xi)))
#    print('xi ' , xi)
#    print('yi ' , yi)
    prediction = predict(weights, xi)
    if (prediction == 'nan'): print('predict(weights, xi) ', predict(weights, xi))
    if yi == 1:
        #print('yi == 1', predict(weights, xi))
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
#def logistic_log_likelihoodi(xi, yi, weights):
#    print('xi ' , xi)
#    print('yi ' , yi)
#    print('predict(weights, xi) ', predict(weights, xi))
#    if yi == 1:
#        return math.log(predict(weights, xi))
    #Yi == 0:
#    else: return math.log((1 - predict(weights, xi)))
def logistic_log_likelihood(weights, X_train, Y_train):
    label_probabilities=np.array([logistic_log_likelihoodi(xi, yi, weights) for (xi, yi) in zip(X_train,Y_train)])
    print('- sum', -np.sum(label_probabilities))
    return -np.sum(label_probabilities)
def loss(weights, X_train,Y_train):  #algum erro aqui!!
    #preds = predict(weights, X_train)
    label_probabilities = 0
    
    for i, yi in enumerate(Y_train):   
        label_probabilities=label_probabilities+yi*np.dot(X_train[i], weights) -np.log(1+ np.exp(np.dot(X_train[i],weights)))
        #if (yi ==1) & (preds[i]!=0):
        #    label_probabilities = label_probabilities + np.log(preds[i])
        #elif (yi ==0) & (preds[i]!=1.):
        #    label_probabilities = label_probabilities + np.log(1-preds[i])
    #label_probabilities = np.log(preds) * Y_train + (np.log(1- preds))*(1- Y_train)   ####ERRO!!! log em cada termo!!
    #print('weights', weights)
    
    #print('label_probabilities ', label_probabilities)
    #print(numpy.log(label_probabilities))
    #if label_probabilities==0.0: print(label_probabilities)
    #print('- sum', -np.sum(label_probabilities))
    return -(label_probabilities)

def grad_loss(weights, X_train, Y_train):
    #x, y = real(z), imag(z)
    return grad(loss, 0)(weights, X_train, Y_train) #derivative wrt. weights
#def grad_loss(weights, X_train, Y_train):
#    return grad(logistic_log_likelihood,0)(weights, X_train, Y_train) #derivative wrt. weights
def steepest_descent_auto(X_train, Y_train, alpha =0.001):
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    intercept = np.ones((X.shape[0], 1))
    X_train   = np.concatenate((intercept, X_train), axis=1) 
    weights = numpy.ones(X_train.shape[1])
    #weights = np.random.randn(X_train.shape[1]) 
    print('initial weights ', weights)
    loss_values = []
    #gradientloss = grad(loss)
    #gradientloss = grad_loss(weights, X_train, Y_train)
    for i in range(500):
        #loss_values.append(logistic_log_likelihood(weights,X_train,Y_train))
        #loss_values.append(loss(weights,X_train,Y_train))
        #print('loss_values[-1]', loss_values[-1])
        #step = grad_loss(weights, X_train, Y_train) #gradientloss(weights,X_train,Y_train)
        deriv = sigmoid_prime(np.dot(X_train, weights))
        #print('deriv ', deriv)
        error  = Y_train- predict( weights, X_train)
        l1 = error * deriv
        alpha = 4/(1.0+i)+0.0001
        nextweights = weights + np.dot(X_train.T,l1)* alpha

        weights = nextweights      
    return weights#, loss_values[-1]
    #for _ in range(100):
    #        loss_values.append(loss(weights, X_train, Y_train))
    #        step = gradientloss(weights, X_train, Y_train)
    #        nextweights = weights - step * learning_rate
    #        if abs(loss_values[-1]- loss(nextweights,X_train,Y_train))<tolerance:
    #            return weights
    #        else:
    #            weights = nextweights
    
################################
def logistic_reg(X_train, y, epochs, lr):   #ruim...
    xb = numpy.c_[numpy.ones((X_train.shape[0],1)),X_train]
    #beta_hat =  np.linalg.inv(xb.T @ xb) @ xb.T @ y
    weights = numpy.ones(xb.shape[1]) #number of columns of xb
    for j in range(1,epochs):
        residual = sigmoid(xb @ weights) - y
        delta = (xb.T @ residual )* (1/xb.shape[0])
        weights = weights - lr* delta
    print(log_likelihood(xb, y, weights))
    return weights
def log_likelihood(xb, y, weights):
    scores = xb @ weights
    ll = (y * scores) - numpy.log(1+ numpy.exp(scores))
    return sum(ll)
##########################Neural network:
def neuralnetwork(sizes,X_train,Y_train,validation_x,validation_y,verbose=False,
                  epochs=30,mini_batch_size=10,lr=0.05,C='ce'):
    num_layers= len(sizes)
    listw = sizes[:-1] #not including the index -1 (last)
    listb = sizes[1:] # skips first element 
    #print('listb ', listb)
    #print('listw ', listw)
    #initialises with gaussian distribution for biases and weights
    biases = np.array([[random.gauss(0,1) for _ in range(b)] for _,b in enumerate(listb)])
    weights = np.array([[[random.gauss(0,1) for _ in range(w)] for _ in range(b)] for w,b in zip(listw,listb)])
    biases = np.array(list(map(np.array, biases)))
    weights = np.array(list(map(np.array, weights)))
    #print('biases initial ', biases)
    #print('initial weights ', weights)
    biases, weights = SGD(X_train, Y_train, epochs, mini_batch_size, lr, C, sizes, num_layers, 
        biases, weights, verbose, validation_x, validation_y)
    return biases, weights
def SGD(X_train, Y_train, epochs, mini_batch_size, lr, C, sizes, num_layers,
       biases, weights, verbose, validation_x, validation_y):
    #every epoch
    training_data =np.concatenate((X_train,np.array([Y_train]).T),axis=1) # zip(X_train,Y_train)
    for j,_ in enumerate(epochs):
        np.random.shuffle(training_data) #stochastic mini_batch (shuffle data)
        
        #Partition set into mini_batches
        mini_batches =  numpy.split(training_data, math.ceil(Y_train.shape[0]/mini_batch_size))#
                
        #feed-forward (and back) all mini_batches
        for _, minib in enumerate(mini_batches):
            biases, weights = update_mini_batch(minib, lr, C, sizes, num_layers,biases,weights)
            
        if(verbose):
            if(j % 1 ==0): 
                print('Epoch ', j, ' finished' )
                confusion = evaluate(X_train, Y_train, biases, weights)
    if (verbose): print('Training ended.')
    return biases, weights
def update_mini_batch(minibatch,lr, C,sizes,num_layers,biases,weights):
        #print('type(minibatch)', type(minibatch), minibatch.shape)
        print('mini_batch ', minibatch)
        nmb = (minibatch).shape[0]   
        listb = sizes[1:]
        listw = sizes[:-1]
        #initialise updates with zero arrays
        nabla_b = numpy.dot(0.0 , (biases))   #numpy.array([[0] for _ in listb])
        nabla_w = numpy.dot(0.0, (weights)) #numpy.array([[0 for _ in listw] for _ in listb])
        #print('weights ', weights)
        print('nabla_w from weights -shape ', nabla_w.shape)
        print('nabla_b from weights -shape ', nabla_b.shape)
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
            print(delta_nabla_w.shape, 'delta_nabla_w ', delta_nabla_w)
            
            print('shapes dos2', delta_nabla_w[0].shape, delta_nabla_w[1].shape)
            
            print(delta_nabla_b.shape, 'delta_nabla_b ', delta_nabla_b)
            print('shapes dos2', delta_nabla_b[0].shape, delta_nabla_b[1].shape)
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
    nabla_b_backprop = numpy.dot(0., biases)     #numpy.array([[0] for _ in listb])
    nabla_w_backprop = numpy.dot(0., weights)
    #Feed-forward (get predictions)
    activation = numpy.array(x)           #first activation is input vector x
    activations = [numpy.array(list(x))]
    print('activations: ',  activations)
    zs = []                               #to store computation in each neuron
    #print('weights ', weights)
    #print('shape of weights ', weights.shape)
    for f, b in enumerate(biases):
        w = numpy.array(weights[f])
        print('f', f, 'w in loop', w)
        print('activation : ', activation)
        w_a = numpy.dot(w , activation)
        print('w_a =  numpy.dot(w , activation): ', w_a )
        b_broadcast = numpy.array(b)   #??
        z = w_a + b
        print('b :', b)
        print('z = w_a + b : ', z)
        zs.append(z)
        activation = sigmoid(z)    #activation function
        activations.extend([activation])
        
    print('ACTIVATIONS : ', activations[0])
    print( activations[1])
    print( activations[2])
    #Backwards (update gradients using errors)
    #last layer
    print( 'in last layer, y ', y, 'activations[-1 ]', activations[-1])
    print('zs stored :' , zs)
    delta = cost_delta(method= C, z = zs[-1],  a=activations[-1], y = y)
    print('DELTA : ', delta)   
    nabla_b_backprop[-1] = delta
    print('in backprop: numpy.array(activations[-2]).T ', numpy.array(activations[-2]).T )
    if len(delta.shape)== 1: delta = delta[0]
    nabla_w_backprop[-1] = numpy.dot(delta , numpy.array(activations[-2]).T) #transpose
    print('in backprop: nabla_b_backprop[-1] ', nabla_b_backprop[-1] )
    print('in backprop: nabla_w_backprop[-1] ', nabla_w_backprop[-1] )
    

    #Second to second-to-last-layer
    #if no hidden layer reduces to multinomial logit
    if (num_layers>2):
        for k in range(1,(num_layers-1)):
            print("ENTRA NESTE LOOP, k = ", k)
            sp = sigmoid_prime(zs[-1-(k)])
            print('sp ', sp)
            print('numpy.array(weights[-1- (k-1)]).T ', numpy.array(weights[-1- (k-1)]).T)
            print('delta ', delta)
            delta = numpy.multiply(numpy.dot(numpy.array(weights[-1- (k-1)]).T ,delta).flatten() ,sp)  #(numpy.array(weights[-1- (k-1)]).T * delta) *sp
            print('delta in loop k', delta)
            nabla_b_backprop[-1 - k] = delta
            testyy = numpy.array(activations[-1 - (k+1)]).T
            print('testyy', testyy)
            nabla_w_backprop[-1 -(k)] = numpy.multiply(delta , testyy)
            
    return nabla_b_backprop, nabla_w_backprop
def feedforward(a, biases, weights):
    for f, b in enumerate(biases):
        #w_a = numpy.dot(weights[f], a )
        a = sigmoid(numpy.dot(weights[f], a) + b)
    return a
def get_predictions(test_Y, biases, weights):
    feedfor = [feedforward(testY, biases, weights) for testY in test_Y]   ### ???? O QUE ????
    return pandas.Series(feedfor).idxmax()
def cost_delta(method, z, a , y):
    if(method=='ce'): return (a-y)   #'ce' for cross-entropy loss function
def evaluate(X_test, Y_test, biases, weights):
        
    pred = get_predictions(X_test, biases, weights)
    truths = pandas.Series(Y_test).idxmax()
    #Accuracy
    correct = np.sum([1 for x, y in zip(pred,truths) if x==y ])
    total = Y_test.shape[0]
    print(correct/total)
    #Confusion
    #rows = []
    res = pandas.DataFrame({'Prediction': pred.T, 'Truth': truths.T})#, index=rows )  #pred truths as dataframe
    return res
########################################end: Neural network
def in_random_order(data):
    """generator that returns the elements of data in random order"""
    indexes = [i for i, _ in enumerate(data)] # create a list of indexes
    #print('indexes_antes',indexes)
    random.shuffle(indexes)                   # shuffle them
    return indexes
def gradDscent(dataMat, classLabels, alpha = 0.001):

    labelMat = numpy.mat(classLabels).transpose()
    dataMat = numpy.mat(dataMat)
    m,n = numpy.shape(dataMat)
    #alpha = 0.001
    maxCycles = 500
    weights = numpy.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMat*weights)
        error = (labelMat - h)
        weights = weights - alpha * dataMat.transpose()* error
    return weights
def stocGradAscentA(dataMatrix, classLabels, numIter=300):     
    '''dataMatrix is x, classLabels is y (output)'''
    m,n = numpy.shape(dataMatrix)
    weights = numpy.ones(n)   #initialize to all ones
    #weights = [random.gauss(0,1) for _ in range(n)]   #random number with 0 mean and 1 std
    #print(weights)
    for j in range(numIter):
        for i, index  in enumerate(in_random_order(zip(dataMatrix, classLabels))):
            dataMatrix_i, classLabels_i = dataMatrix[index], classLabels[index]
            #alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not go to 0 because of the constant
            alpha = 4/(1.0+j+i)+0.0001
            #print(dataMatrix_i)
            #print(sum(dataMatrix_i*weights))
            h = sigmoid(sum(dataMatrix_i*weights))
            error = h - (classLabels_i)
            #weights = weights + alpha * error * dataMatrix_i
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

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = numpy.array(dataMat)
    n = numpy.shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = numpy.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def test(X_train, Y_train, X_test, Y_test):
    #frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = X_train; trainingLabels = Y_train
    m, n = numpy.shape(trainingSet)
    trainWeights = sgd(numpy.array(trainingSet), trainingLabels)
    #stocGradAscentA(numpy.array(trainingSet), trainingLabels, 500)
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
        
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
def learning_schedule(t):
    t0, t1 = 4.0, 5.0
    alpha=0.0001
    return t0/(t + t1)
def sgd(dataMatrix, classLabels):
    n_epochs = 200

    m,n = numpy.shape(dataMatrix)
    intercept = np.ones((dataMatrix.shape[0], 1))
    dataMatrix= np.concatenate((intercept, dataMatrix), axis=1) 
    weights = numpy.ones(n+1)   #initialize to all ones
    #weights = [random.gauss(0,1) for _ in range(n)]   #random number with 0 mean and 1 std
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = numpy.random.randint(m)
            xi = dataMatrix[random_index:random_index+1]
            yi = classLabels[random_index:random_index+1]

            gradients = (1/m)*sum(sigmoid(sum(xi*weights))- yi )*xi    #2* xi.T.dot(xi.dot(weights)- yi)
            eta = learning_schedule(epoch*m + i)
            weights = weights - eta * gradients/m
    return weights
