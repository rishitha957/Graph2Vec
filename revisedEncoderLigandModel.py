#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import nltk
import pandas as pd
#from pysmiles import read_smiles
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect,GetHashedTopologicalTorsionFingerprintAsBitVect
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import callbacks
import keras
import numpy as np
import time
from keras import metrics
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Input
from keras.models import Model
from sklearn.neighbors import KDTree
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# In[2]:


class FingerPrintAutoEncoder(object):
	@staticmethod
	def buildandTrainEncoder(input):
		vectorLenght=len(input[0])
		inputData=Input(shape=(vectorLenght,))
		encoded = Dense(units=500, activation='relu')(inputData)
		decoded = Dense(units=vectorLenght, activation='relu')(encoded)
		autoencoder=Model(inputData, decoded)
		encoder = Model(inputData, encoded)
		encoder.summary()
		autoencoder.summary()
		autoencoder.compile(loss=keras.losses.mean_squared_error,
		optimizer=keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0))
		autoencoder.fit(input, input,
			epochs=50,batch_size=16)
		return encoder


# In[3]:


class TestDescriptionAutoEncoder(object):
	@staticmethod
	def buildandTraintextEncoder(input):
		vectorLenght=len(input[0])
		inputData=Input(shape=(vectorLenght,))
		encoded = Dense(units=500, activation='relu')(inputData)
		decoded = Dense(units=vectorLenght, activation='relu')(encoded)
		autoencoder=Model(inputData, decoded)
		encoder = Model(inputData, encoded)
		decoderInput=Input(shape=(500,))
		decoder_layer = autoencoder.layers[-1]
		decoder = Model(decoderInput,decoder_layer(decoderInput))
		encoder.summary()
		autoencoder.summary()
		autoencoder.compile(loss=keras.losses.mean_squared_error,
		optimizer=keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0))
		autoencoder.fit(input, input,
			epochs=50,batch_size=16)
		return encoder,decoder


# In[4]:


class FingerPrintTextDescriptionMapper(object):
	@staticmethod
	def buildMapper(sizeInputTransformedFingerPrint,sizeOutputTransformedTextData):
		model = Sequential()
		model.add(Dense(600, input_dim=sizeInputTransformedFingerPrint, activation='relu'))
		model.add(Dense(800, activation='relu'))
		model.add(Dense(600, activation='relu'))
		model.add(Dense(sizeOutputTransformedTextData))
		return model
	@staticmethod
	def trainMapper(inputTransformedFingerPrint,outputTransformedTextData,lr,epoch,batch_size):
		sizeOfInput=inputTransformedFingerPrint.shape[1]
		sizeOfOutput=outputTransformedTextData.shape[1]
		modelMapper= FingerPrintTextDescriptionMapper.buildMapper(sizeOfInput,sizeOfOutput)
		adam = Adam(lr=lr, 
					beta_1=0.9, 
					beta_2=0.999, 
					epsilon=None, 
					decay=0.0, 
					amsgrad=False)
		modelMapper.compile(optimizer=adam, 
							loss='mean_squared_error', 
							metrics=[metrics.mae])
		tfCallBack = callbacks.TensorBoard(log_dir='./graph', 
											histogram_freq = 0, 
											batch_size=batch_size, 
											write_graph=True, 
											write_grads=False, 
											write_images=True, 
											embeddings_freq=0, 
											embeddings_layer_names=None, 
											embeddings_metadata=None)
		modelMapper.fit(inputTransformedFingerPrint,
						outputTransformedTextData,
						epochs=epoch, 
						batch_size=batch_size, 
						verbose=2,
						shuffle = True, 
						callbacks = [tfCallBack])
		print("Model Train Complete")
		return modelMapper


# In[5]:


def removeBracketsAndunwnatedCHaracterInString(textData):
	textData=textData.replace("(","").replace(")","").replace(". ","")
	return textData


# In[6]:


def removeStopWordsInDescription(descriptionText,stopWords):
	#print(descriptionText)
	words=word_tokenize(descriptionText)
	processedDescriptionTestToken=[ w for w in words if not w in stopWords]
    #print(processedDescriptionTestToken)
	return processedDescriptionTestToken

def getLemmatizationForWordToken(tokenList):
	lemmaLizedWordTokenList=[]
	for word in tokenList:
		lemmaLizedWordTokenList.append(getLemmetizingOFAWordToken(word))
	return lemmaLizedWordTokenList


def generateNgramFormWorkToken(wordToken,n):
	nGramFormWordToken=ngrams(wordToken,n)
	ngramsFromToke=[' '.join(grams) for grams in nGramFormWordToken]
	return ngramsFromToke

def getLemmetizingOFAWordToken(word):
	lemmatizer=WordNetLemmatizer()
	lword=lemmatizer.lemmatize(word)
	return lword


def getAtomPairFingerPrintFromSmile(smile):
	try:
		molecule=Chem.MolFromSmiles(smile)
		# print("here1")
		atomPairFP=GetHashedAtomPairFingerprintAsBitVect(molecule)
		# print("Here2")
		return atomPairFP.ToBitString()
	except:
		raise Exception("Not able to Generate FigerPrint")


def figerPrintStringToVector(fingerPrint):
	fingerPrintvector=[ int(c) for c in fingerPrint]
	return fingerPrintvector


def generateNgramWordFeaturesFromBagOFnGramWords(bagofNgramWords):
	freqNgramWordToken=nltk.FreqDist(bagofNgramWords)
	mostlCommonWords=freqNgramWordToken.most_common(7000)
	bagOfWords1=[x[0] for x in mostlCommonWords]
	return bagOfWords1

def findFeatureFromWords(text,nGramWordFeatures):
	nGramWords=set(text)
	WordfeatureVector=[]
	for w in sorted(nGramWordFeatures):
		if w in nGramWords:
			WordfeatureVector.append(1)
		else:
			WordfeatureVector.append(0)
	return WordfeatureVector


def predictTextDescriptionFromFingerPrint(fingerPrint,fingerPrintEncoder,modelMapper):
	# fingerPrint=np.asarray(fingerPrint).reshape(1,len(fingerPrint))
	transformedfingerPrint=fingerPrintEncoder.predict(fingerPrint)
	predictedTransformedTestData=modelMapper.predict(transformedfingerPrint)
	return predictedTransformedTestData

def tanimoto_coefficient(p_vec, q_vec):
	return ;

def findNearestTextVector(moleculeTextVector,textDataSet):
	kdTree=KDTree(textDataSet, leaf_size = 3)
	dist, index=kdTree.query(moleculeTextVector, k=3)
	# print(index)
	return index[0]


# In[7]:

#Main Function
if __name__=="__main__":
	nGrameValue=1
	filename='drugInformationRefined.tsv'
	drugInformationData= pd.read_csv(filename,delimiter="\t")
	print(drugInformationData.columns.values)
	totalNumberOfData=len(drugInformationData)
	print("Total Number of data",totalNumberOfData)
	stopWords=set(stopwords.words("english"))
	print("Stop Words Created")
	createBagOfNGramsWords=[]
	DescriptionInformationForWholeDataSet=[]
	ListOFMorganFPForWholeDataSet=[]
	#DataPreparation
	for index in range((totalNumberOfData)):
	    try:
	        textdata=removeBracketsAndunwnatedCHaracterInString(drugInformationData.iloc[index,3])
	        filteredSentence=removeStopWordsInDescription(str(textdata),stopWords)
	        filteredLemalizedSentence=getLemmatizationForWordToken(filteredSentence)
	        nGrameTokenWord=generateNgramFormWorkToken(filteredLemalizedSentence,nGrameValue) 

	        atomPairFP=getAtomPairFingerPrintFromSmile(drugInformationData.iloc[index,1])
	        fingerPrint=figerPrintStringToVector(atomPairFP)
	        createBagOfNGramsWords.extend(nGrameTokenWord)
	        DescriptionInformationForWholeDataSet.append(nGrameTokenWord) 
	        ListOFMorganFPForWholeDataSet.append(fingerPrint)
	    except :
	        print('Exception')
	print("Data Preparation Complete")


	# In[8]:


	print(len(DescriptionInformationForWholeDataSet))
	nGramWordFeatures=generateNgramWordFeaturesFromBagOFnGramWords(createBagOfNGramsWords)
	print(len(nGramWordFeatures))
	featureVectorNgramText=[]
	print("Start Generating text Feature")
	for index in range(len(DescriptionInformationForWholeDataSet)):
	    featureVector=findFeatureFromWords(DescriptionInformationForWholeDataSet[index],nGramWordFeatures)
	    featureVectorNgramText.append(featureVector)

	print("Feature Generating Complete")

	print("Model Training")

	ListOFMorganFPForWholeDataSet=np.asarray(ListOFMorganFPForWholeDataSet).reshape(len(ListOFMorganFPForWholeDataSet),len(ListOFMorganFPForWholeDataSet[0]))
	featureVectorNgramText=np.asarray(featureVectorNgramText).reshape(len(featureVectorNgramText),len(featureVectorNgramText[0]))

	trainSetMorganFP, testSetMorganFP, trainSetNgramText, testSetNgramText = train_test_split(ListOFMorganFPForWholeDataSet, featureVectorNgramText, test_size=0.33, random_state=42)
	print("Dimension of Train Dataset: ",trainSetMorganFP, trainSetNgramText)
	fingerPrintEncoder=FingerPrintAutoEncoder.buildandTrainEncoder(trainSetMorganFP)	
	testDescriptionEncoder, testDescriptionDecoder=TestDescriptionAutoEncoder.buildandTraintextEncoder(trainSetNgramText)

	print("Model Training Complete")


	# In[9]:


	print("Prediction Start")
	transformedTrainFingerPrint=fingerPrintEncoder.predict(trainSetMorganFP)
	transformedTrainTextDescription= testDescriptionEncoder.predict(trainSetNgramText)
	transformedTestFingerPrint=fingerPrintEncoder.predict(testSetMorganFP)
	transformedTestTextDescription= testDescriptionEncoder.predict(testSetNgramText)
	print(transformedTrainFingerPrint.shape, transformedTestFingerPrint.shape)
	print(transformedTrainTextDescription.shape, transformedTestTextDescription.shape)
	print("Training Model Starts")


	# In[11]:


	learningRate=0.0001
	epoch=100
	batchSize=32

	modelMapper=FingerPrintTextDescriptionMapper.trainMapper(transformedTrainFingerPrint,transformedTrainTextDescription,learningRate,epoch,batchSize)

	predictedTranferedTextData=predictTextDescriptionFromFingerPrint(testSetMorganFP,fingerPrintEncoder,modelMapper)
	print(predictedTranferedTextData.shape)
	desciption=testDescriptionDecoder.predict(predictedTranferedTextData)
	print(desciption.shape)


	# In[18]:


	for i in range(predictedTranferedTextData.shape[0]):
	    ind = findNearestTextVector(predictedTranferedTextData[i:i+1], transformedTrainTextDescription)
	    for j in ind:
	        #print(transformedTestTextDescription[i].shape)
	        score = tanimoto_coefficient(transformedTestTextDescription[i], transformedTrainTextDescription[j])
	        print(score)
	    print()

