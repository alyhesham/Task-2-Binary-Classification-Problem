# Task-2-Binary-Classification-Problem

This python script handles a binary classification problem. Firstly, I drop variables 18 and 19 as 18 has more than 65 percent missing. Variable 19 had a correlation
of value 1 with the classLabel in the training but almost 0 in the validation so I removed it as it would shift the weights in my classifier and it produce
wrong results. I then dropped the rows that contained NA and a classLabel of yes as there was a bias towards the yes label with a ratio of 32:2 with the no class,
so it wouldn't affect my data drastically if i lost the rows with some missing data that had the label yes. After that I split the training data according to class 
in order to fix the NA values in categorical data in order to not have bias when filling the values with the most frequent category that didnt represent the same class.
In order to further prepare the data for classifying I applied an Over sampler for the minority which is the no class and an Under sampler for the majority which is the yes
class. I ended up with a ratio of 6:4 instead of 32:2. After that I imputed the quantitative variables by the median value of each column. There were many categorical data
so I ahd to one hot encode the training and the validation data to be prepared for the classifier. This resulted in n-1 cols for the categorical variables. I tried
different classifiers; I used KNN, logreg, RandomForest and SVM. They almost had close results, but RandomForest produced the best accuracy and precison. I have also hypertuned
the parameters of the classifier in order to find the best parameters to use. This model had 89-90% accuracy which means it incorrectly identied 20 out of the small validation set of 200 data entries.
A precision and recall of 0.89 but it correctly identifies the no label slightly better than the yes label.
