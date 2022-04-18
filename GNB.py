"""
Need to add top layer comments
"""

class GNB:

	def __inti__(self, X, Y):
		self.X = X
		self.Y = Y


	# Summarize Feature
	def summarizeFeature(featureArr):
	# Everything should be numpy array type
	    # Data Mean
	    featureMean = sum(featureArr)/len(featureArr)
	    # Data STD - sample
	    featureSTD = sum((featureArr - featureMean) ** 2)/(len(featureArr)-1)
	    return [featureMean,featureSTD**.5]

	# Summarize Data
	def SummarizeData(features):
	    # 1st column in mean and 2nd column is std
	    summaries = np.empty(0)
	    for feature in np.transpose(features):
	        summaries = np.append(summaries, summarizeFeature(feature))
	    summaries = np.reshape(summaries,(-1,2))
	    return summaries

	# Calculate prior probability
	def GetPrior(Y):
	    # Get the number of classes present
	    classes = np.unique(Y)
	    n_classes = len(classes)
	    # Get total numbers for each class
	    n_in_c = np.empty(0)
	    for c in classes:
	        n_in_c = np.append(n_in_c, len(np.where(Y==c)[0]))
	    # Total number of targets
	    t = len(Y)
	    # Prior probabilities' calculation
	    priors = n_in_c / t
	    return priors

	# Calculate Gaussian probability density function 
	def GaussPDF(val, mean, std):
	    return (1/((2*math.pi)**.5)*std) * (math.e)**((-1/2)*(((val-mean)/std)**2))

	# Calculate likelihood of the test data with respect to each class
	def EstimateLikelihood(test, param):
	    n_classes = len(param)
	    likelihoods = {}
	#     print(param[1]['sumStats'][0])
	    for c in list(param.keys()):
	        lik = 1
	        for f in range(len(test)):
	            c_mean = param[c]['sumStats'][f][0]
	            c_std = param[c]['sumStats'][f][1]
	            lik = lik * GaussPDF(test[f], c_mean, c_std)
	        likelihoods[c] = lik
	    return likelihoods

	# Calculate the Posterior Probabilities
	def EstimatePosterior(likelihoods, param):
	    posteriors = {}
	    # Calculate Marginal Probability
	    margProb = 0
	    for c in list(param.keys()):
	        margProb += param[c]['prior'] * likelihoods[c]
	    # Calculate Posterior Probability
	    for c in list(param.keys()):
	        posteriors[c] = param[c]['prior'] * likelihoods[c] / margProb
	    return posteriors

	# Training the model
	def TrainModel(X, Y):
	    # Group the data into their respective classes
	    
	    # Get prior probability for each class
	    priors = GetPrior(Y)
	    print("Priors = ",priors)
	    # Get the number of classes present
	    classes = np.unique(Y)
	#     cSummary = np.zeros((len(classes), np.size(X,1), 2))
	    trainParam = {}
	    # Get summary data for each class 
	    for c in classes:
	        cIdx = np.where(Y==c)[0]
	        cX = X[cIdx,:]
	        cSummary = SummarizeData(cX) 
	        trainParam[c] = {
	            'prior':priors[np.where(classes==c)[0]],
	            'sumStats': cSummary
	        }
	    return trainParam

	# Testing a single data point
	def TestData(test, param ,y):
	    # Get the likelihoods for every class
	    likelihoods = EstimateLikelihood(test, param)
	    # Get the posterior probabilities for every class
	    posteriors = EstimatePosterior(likelihoods, param)
	    # Get the class for the Maximum Posterior Probability
	    maxClass = max(posteriors, key = posteriors.get)
	    return maxClass

	# Testing the model with multiple data points
	def TestModel(X, param, Y):
	    predictedClass = np.empty(0)
	    for test, y in zip(X,Y):
	        predictedClass = np.append(predictedClass, TestData(test, param,y))
	        
	    return predictedClass

	# Get the Accuracy of the Model
	def ModelAccuracy(predictedVal, trueVal):
	    correct = 0
	    for p, t in zip(predictedVal, trueVal):
	        if p == t:
	            correct += 1
	    acc = correct / len(trueVal)
	    return acc

    def ImplementGNB(self):
    	# Split Dataset
		splitPercent = 0.8
		size_trainData = int(len(self.X)*splitPercent)
		size_testData = len(self.X)-size_trainData

		# Randomly select indices
		trainIdx = random.sample(range(len(self.X)), size_trainData)

		# Get random training data
		trainX = self.X
		trainX = trainX[trainIdx,:]

		# Get the rest of the testing data
		testIdx = list(set(range(len(self.X))) - set(trainIdx))
		testX = self.X
		testX = testX[testIdx,:]

		# Split Y
		trainY = self.Y[trainIdx]
		testY = self.Y[testIdx]

		# Get Mean, STD for each feature is given in the form of nFeatures x 2 dimension
		fSummary = SummarizeData(trainX) 
		# Train the Model
		trainParam = TrainModel(trainX, trainY)
		# Test the Model
		predictedClass = TestModel(testX, trainParam, testY) #TestModel(trainX, trainParam,trainY) #TestModel(testX, trainParam, testY)
		# Get the Accuracy of the trained model
		acc = ModelAccuracy(predictedClass, testY)

		return acc

