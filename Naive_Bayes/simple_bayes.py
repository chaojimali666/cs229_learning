'''
	PostingList is From http://ml.apachecn.org/mlia/naive-bayes/

    Theory  refers to Andrew Ng's lecture note part 4
'''
from numpy import *
import re
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

def classifyNB(dataset,fi_y,fi_jy0,fi_jy1):
    fi_jy1 = np.array(fi_jy1)
    fi_jy0 = np.array(fi_jy0)
    p1 = (sum(dataset*fi_jy1)*fi_y)/((sum(dataset*fi_jy1))*fi_y+(sum(dataset*fi_jy0))*(1-fi_y))
    p0 = 1-p1
    if p1>p0:
        return 1
    else:
        return 0


def testingNB():
    postingList,classVec = loadDataSet()
    x_dict = get_feature_vector(postingList)
    fi_y,fi_jy1,fi_jy0 = train_data(x_dict,postingList,classVec)
    testEntry = ['love','my','dalmation']
    dataset = get_document_vector(x_dict,testEntry)
    dataset = array(dataset)
    p = classifyNB(dataset,fi_y,fi_jy0,fi_jy1)
    print(testEntry,'classify as :',p)
    testEntry = ['stupid','garbage']
    dataset = get_document_vector(x_dict,testEntry)
    dataset = array(dataset)
    p = classifyNB(dataset,fi_y,fi_jy0,fi_jy1)
    print(testEntry,'classify as :',p)
    testEntry = ['I','don\'t','want','to','be','a','stupid','guy']
    dataset = get_document_vector(x_dict,testEntry)
    dataset = array(dataset)
    p = classifyNB(dataset,fi_y,fi_jy0,fi_jy1)
    print(testEntry,'classify as :',p)
    testEntry = ['you','are','a','dog']
    dataset = get_document_vector(x_dict,testEntry)
    dataset = array(dataset)
    p = classifyNB(dataset,fi_y,fi_jy0,fi_jy1)
    print(testEntry,'classify as :',p)




def train_data_2(x_dict,postingList,classVec):
    """
    event model:calculate the frequency of the words appearing in spam(non-spam) email
    """
    fi_y = (sum([i for i in classVec if i == 1]))/(len(classVec))
    fi_jy1 = [1] * len(x_dict)
    fi_jy0 =  [1]*len(x_dict)
    y0 = len(x_dict)
    y1 = len(x_dict)
    for email in postingList:
        y = classVec[postingList.index(email)]
        if y == 1:
            y1 = y1 + len(email)
            for word in email:
                fi_jy1[x_dict.index(word)] += 1
        elif y == 0:
            y0 = y0 + len(email)
            for word in email:
                fi_jy0[x_dict.index(word)] += 1

    fi_jy0 = [fi/y0 for fi in fi_jy0]
    fi_jy1 = [fi/y1 for fi in fi_jy1]
    return fi_y,fi_jy1,fi_jy0
 

def string_parse(email_txt):
    regex = re.compile('\\W+')
    vocabulary_list  = regex.split(email_txt)
    vocabulary_list = [word.lower() for word in vocabulary_list if len(word) > 2]
    return vocabulary_list



def testNB2():
    postingList = []
    classVec = []

    #load data
    for i in range(1,26):
        email_spam = string_parse(open('email/spam/%d.txt'%(i)))
        postingList.append(email_spam)
        classVec.append(1)
        email_normal = string_parse(open('email/ham/%d.txt'%(i)))
        postingList.append(email_normal)
        classVec.append(0)
    x_dict = get_feature_vector(postingList)

    #split data into training/test set
    test_data = []
    test_value = []
    for i in range(10):
        index = int(random.uniform(1,50))
        test_data.append(postingList[index])
        test_value.append(classVec[index])
        del(postingList[index])
        del(classVec[index])

    fi_y,fi_jy1,fi_jy0 = train_data_2(x_dict,postingList,classVec)

    res_test = [classifyNB(dataset,fi_y,fi_jy0,fi_jy1) for dataset in test_data]

    prob = sum([abs(res_test[i]-test_value[i]) for i in range(len(res_test))])/len(res_test)

    print('the error rate of classify is %d'%(prob))





