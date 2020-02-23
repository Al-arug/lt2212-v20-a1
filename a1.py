import os
import sys
import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def part1_load(folder1, folder2, n=1):
    allfiles = glob("{}/*.txt".format(folder1)) + glob("{}/*.txt".format(folder2))
    
    #spliting the file names for the first two columns
    classes_and_files=[i.split("/") for i in allfiles]
    B=pd.DataFrame({"files":[i[1]for i in classes_and_files],"classes":[i[0]for i in classes_and_files]})
    
    corpus = ""
    data_in_files={}
    for filename in allfiles:
        data_in_files[filename]=""
        with open(filename, "r") as thefile:
            for line in thefile:
                corpus += "\n" + line
                data_in_files[filename]+=line.lower()

    p= corpus.lower().split()
    #counting each word in the corpus using numpy, ouputs an array of tuples (word,count)
    unique, counts= np.unique(p, return_counts=True)
    counted_words=np.asarray((unique, counts)).T
    
    indexes= range(0,len(B["classes"]))
    for i in counted_words:
        if int(i[1])>n:     #if the word is mentioned more than n times in the etnire corpus
            counts=[]
            for position in indexes:      # count the occurance of a word in each file
                c=data_in_files[B.loc[position,"classes"]+"/"+B.loc[position,"files"]].lower().split()
                counts.append(c.count(i[0]))
            B[i[0]]=[int(i) for i in counts]

    return B 
df=loader_data("crude","grain",1000)
def part2_vis(df,m):
  
    assert isinstance(df, pd.DataFrame)

    # a copy of the dataframe using iloc, excluding the column of the filename
    R=df.iloc[:,1:]
    R= R.groupby(["classes"]).sum()
    i= R.append(R.agg(["sum"]))
    h=i.T.sort_values("sum",ascending=False).T
    t=h.iloc[: , 0:m]             #sorted dataframe exluding the total raw and including top m
    
    return t.T.plot(kind="bar")

def part3_tfidf(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)
    t=df.iloc[: ,2:]    # a copy without calsses and files
    number_of_allfiles=len(df)
    v= np.log(number_of_allfiles/t[t>0].count())
    b= t * np.array(v)   # tf times idf
    f=df.iloc[:,:2]      #files and classes
    c=pd.concat([f,b],axis=1)  #returning classes and files to the new tf-idf dataframe
    
    return c 
def classifier_data(T):
    
    X=T.iloc[:,2:]
    y=T["classes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75,test_size=0.25,random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred_class= logreg.predict(X_test)
    
    return metrics.accuracy_score(y_test, y_pred_class)
