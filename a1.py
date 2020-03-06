import os
import sys
import pandas as pd
import numpy as np
import numpy.random as npr
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def part1_load(folder1, folder2, n):
 
    allfiles = glob("{}/*.txt".format(folder1)) + glob("{}/*.txt".format(folder2))
    
    #spliting the file names for the first two columns
    classes_and_files=[i.split("/") for i in allfiles]
    B=pd.DataFrame({"files":[i[1]for i in classes_and_files],"classes":[i[0]for i in classes_and_files]})
    
    #saving data in a one_string corpus and also in a dict, with filename as keys and the content as value
    corpus = ""
    data_in_files={}
    for filename in allfiles:
        data_in_files[filename]=""
        with open(filename, "r") as thefile:
            for line in thefile:
                corpus += line
                data_in_files[filename]+=line
                
    p= corpus.lower().split()
    #counting each word in the corpus using numpy, ouputs an array of tuples (word,count)
    unique, counts= np.unique(p, return_counts=True)
    counted_words=np.asarray((unique, counts)).T
    
    indexes= range(0,len(B["classes"]))
    for i in counted_words:
        if int(i[1])>n:          #if the word is mentioned more than n times in the etnire corpus  
            counts=[]               
            for position in indexes:    # count the occurance of a word in each file 
                c=data_in_files[B.loc[position,"classes"]+"/"+B.loc[position,"files"]].lower().split()
                counts.append(c.count(i[0]))
            B[i[0]]=[int(i) for i in counts]
    

    return B


def part2_vis(df,m):
  
    assert isinstance(df, pd.DataFrame)
    # a copy of the dataframe using iloc, excluding the column of the filename
    R=df.iloc[1:,:]
    R= R.groupby(["classes"]).sum()         
    R= R.append(R.agg(["sum"]))              
    R=R.T.sort_values("sum",ascending=False).T    
    t=R.iloc[:-1 , 0:m]                      #sorted dataframe exluding the total raw and including top m
    v=t.T.plot(kind="bar")
    return v

def part3_tfidf(df):
    
    assert isinstance(df, pd.DataFrame)
    
    t=df.iloc[: ,2:]            # a copy without calsses and files 
    number_of_allfiles=len(df)  
    v= np.log(number_of_allfiles/t[t>0].count())
    b= t * np.array(v)            # tf times idf
    f=df.iloc[:,:2]              #classes and files
    c=pd.concat([f,b],axis=1)    #returning classes and files to the new tf-idf dataframe
    
    return c 

def test_data(T):

    X=T.iloc[:,2:]
    y=T["classes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred_class= logreg.predict(X_test)
    v=metrics.accuracy_score(y_test, y_pred_class)
    
    #h=y_test.value_counts()

    return v

