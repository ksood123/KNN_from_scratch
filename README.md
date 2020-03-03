# KNN_from_scratch
The following three imputations methods have been used:
1)  1-NN: In this method, we calculate distances of a testing instance from all data points and append the distances in a list. Then we sort the distances in decreasing order and then we make prediction based on a single neighbor it.
2) K-NN: In this method, we calculate distances of a testing instance from all data points and append the distances in a list. Then we sort the distances in decreasing order and then we select 'k' neighbors for making the prediction. These neighbors are then voted for their class labels and the testing instance is predicted to belong to that particular class that has highest number of votes.
3) Weighted KNN: Here we proceed in a similar manner as in K-NN till calculating the distances. But the class prediction is done in a slightly different manner using the following formula:
 
In this formula, the distances calculated are inverted and the sum of these inverted distances is calculated. Then each inverted distance is divided by this sum to obtain the relevant weights.
These weights represent the share of vote for a particular class. These distances are then sorted based on their weights. After that find the indices of 'k' distance measures. Then the index that appears most frequently is returned.
4) Feature Scaling: This is has been done using min/max and standardization scaler from Python's 'sklearn'  library.
Standardization uses following formula to feature scale data:
 
Min/max scaling uses following formula to scale the features:
 

5)  Euclidean Distance: The Euclidean distance between two points in either the plane or 3-dimensional space measures the length of a segment connecting the two points. It is the most obvious way of representing distance between two points. 
It is calculated using following formula:
 
6) Manhattan Distance: Manhattan Distance between two points (x1, y1) and (x2, y2) is:
|x1 – x2| + |y1 – y2|.

NOTE: SINCE CATEGORICAL VAIRABLES HAVE BEEN MAPPED TO NUMERICAL VALUES, THEREFORE, WE USE EUCLIDEAN AND MANHATTAN DISTANCE IN THEIR IMPUTATON AS WELL.

III. ACCURACY MEASURE
For calculating the accuracy of each imputation method, we compute the predictions and make imputations based on those predictions. Then these imputed values are then compared to the values that were actually present before introducing the missing values.
We then divide the number of correct predictions by the total number of missing values that were introduced in that particular iteration and append it to the accuracy list that is later used to filter out accuracy for continuous and categorical features.
