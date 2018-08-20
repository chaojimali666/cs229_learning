'''
	PostingList is From http://ml.apachecn.org/mlia/naive-bayes/
'''
from numpy import *
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def get_feature_vector(postingList):
	"""
	get the word dict
	"""
	x_dict = [word for wordlist in postingList for word in wordlist]
	x_dict = set(x_dict) #remove repeated element
	x_dict = list(x_dict)
	return x_dict

def get_document_vector(x_dict,document):
	"""
		get the vector with x_dict
		if the word in x_dict appears in the document,the value of x is 1
		else the value of x is 0
	"""
	x_vector = [0] * len(x_dict) #the initial value is 0
	for word in document:
		if word  in x_dict:
			x_vector[x_dict.index(word)] = 1
		else:
			print('Hello world ！')
	return x_vector

def train_data(x_dict,postingList,classVec):
    """
		get the parameters
		fi_y
		fi_jy1 while the words are abusive 
		fi_jy0 while normal 
    """
    fi_y = (sum([i for i in classVec if i == 1]))/(len(classVec))
    fi_jy1 = [1] * len(x_dict)
    fi_jy0 =  [1]*len(x_dict)
    y0 = 2
    y1 = 2
    for document in postingList:
        x_vector = get_document_vector(x_dict,document) 
        y = classVec[postingList.index(document)]
        if y == 1:
        	fi_jy1 = [fi_jy1[i]+x_vector[i] for i in range(len(x_vector))]
        	y1 = y1 +1
        elif y ==0:
        	fi_jy0 = [fi_jy0[i]+x_vector[i] for i in range(len(x_vector))]
        	y0 = y0 +1

    fi_jy0 = [fi/y0 for fi in fi_jy0]
    fi_jy1 = [fi/y1 for fi in fi_jy1]
    return fi_y,fi_jy1,fi_jy0














