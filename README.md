# LT2212 V20 Assignment 1


Part1: 

I output and save all the files' contents in a one string corpus, and in a dictionary in the same loop. The dictionary contains all the files as keys, with the file content's as the values. The count all the occurrences of the words in the corpus to determine which words to include in the dataframe, if their count is more than n times. Then I iterate through the dict one by own using the index position of the files in data frame: [position,"classes"]+"/"+B.loc[position,"files"]], and count the occurrence of each word in the given file.  The data is lowered, and I have not removed punctuation and numbers, so non-alphabetic characters will appear in the data frame. 


Part2 : 

step1:I make a copy of the data frame, excluding the column of the file names. Step:2 I group all the the rows into two columns with sum total of each word in each class. Step:3 I add a new row of the total count of the word in both classes in order to sort them. Step 4: I sort the dataframe based on the total sum. Step5: A copy of the sorted data-frame with column limit of the most m occurred words. 


Part3: 

Step1: a copy of the data frame without the files and the classes. Step2: the length of dataframe- or the length of the columns which is equivalent to the the total number of all files. Step3: I get the idf. Step4: I get the tf-idf, the counting times idf. Step:5 a copy of the two columns, classes and files. Step6: I join the the classes and files to the new tf-idf to return the final and full data frame. 


Part4: 
The frequency of the top used words has changed dramatically, with more words that has a remarkable difference in representation from one class to the other. I assume this change is due to the relative ratio that that tf-idf points out, represented by narrower numbers or narrower classification of number of 0 or above 0. 


Bonus part- classiy: 

I used logistic regression to measure the accuracy. Step1: Choosing the feature. Step2: label. The accuracy of part1 to tf-if varied in the beginning. One time it was better for part 1 and other time was better accuracy for tf-idf, I kept changing the train and test size of the data, and also the random state, once zero and once 1. I could not figure out why it differs here. One strange thing I noticed that when I expanded the train size the accuracy went lower, which I assumed the case should be otherwise, meaning that If the training data is larger it should produce more accuracy on the test data. But perhaps I am doing something wrong. I also did not shuffle the the data. The accuracy results were between 0.95-0.96
