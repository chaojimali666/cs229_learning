import numpy as np
import re

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

def get_feature_vector(postingList):
    """
    get the word dict
    """
    x_dict = [word for wordlist in postingList for word in wordlist]
    x_dict = set(x_dict) #remove repeated element
    x_dict = list(x_dict)
    return x_dict

def get_document_vector2(x_dict,document):
    """
        get the vector with x_dict
        if the word in x_dict appears in the document,the value of x is the index in x_dict
    """
    x_vector = [] 
    for word in document:
        if word  in x_dict:
            x_vector.append(x_dict.index(word))
        else:
            print('Hello world ！')
    return x_vector

def classifyNB2(x_dict,dataset,fi_y,fi_jy0,fi_jy1):
    fi_jy1 = np.array(fi_jy1)
    fi_jy0 = np.array(fi_jy0)
    
    s0 = 1 
    s1 = 1
    for word in dataset:
        if word not in x_dict:
            s0 = s0*1/len(x_dict)
            s1  = s1*1/len(x_dcit)
        else:
            s0 = s0 * fi_jy0[x_dict.index(word)]
            s1 = s1 * fi_jy1[x_dict.index(word)]
    p1 = (s1*fi_y)/(s1*fi_y+s0*(1-fi_y))
    p0 = 1-p1
    if p1>p0:
        return 1
    else:
        return 0

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
        email_spam = string_parse(open('email/spam/%d.txt'%(i)).read())
        postingList.append(email_spam)
        classVec.append(1)
        email_normal = string_parse(open('email/ham/%d.txt'%(i)).read())
        postingList.append(email_normal)
        classVec.append(0)
    x_dict = get_feature_vector(postingList)
    test_data = []
    test_value = []
        
    for i in range(10): 
        lenth = len(postingList)
        index = int(random.uniform(0,lenth))
        test_data.append(postingList[index])        
        test_value.append(classVec[index])
        del(postingList[index])
        del(classVec[index])

    fi_y,fi_jy1,fi_jy0 = train_data_2(x_dict,postingList,classVec)
    res_test = [classifyNB2(x_dict,dataset,fi_y,fi_jy0,fi_jy1) for dataset in test_data]
    prob = sum([abs(res_test[i]-test_value[i]) for i in range(len(res_test))])/len(res_test)
    #error_list = [ test_data[i] for i in range(len(test_data)) if res_test[i] != test_value[i]]
    print('the error rate of classify is %f '%(prob))
    return prob
    
a=[]
for i in range(100):
    a.append(testNB2())
print(sum(a)/100)