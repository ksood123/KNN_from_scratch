import pandas as pd
import numpy as np
import math, random, collections
import matplotlib.pyplot as plt

def createMissingValue(df,l,j): # This function creates missing values in a column specified by 'j' and returns the dataframe sod created
    df1=df.iloc[:,:]
    for i in l:
        df1.iloc[i,j]=math.nan
    return df1


def create_Train_Test_DataFrame(df,j,n_cols,l): # This function creates appropriate training and testing data frames by removing relevant rows and columns
    if(j==0):
        df_train=df.iloc[:,1:]
        df_test=df_train.drop(df_train.index[l]).reset_index(drop=True)
        return df_train,df_test
    elif(j==(n_cols-1)):
        df_train=df.iloc[:,:j]
        df_test=df_train.drop(df_train.index[l]).reset_index(drop=True)
        return df_train,df_test
    else:
        df1=df.iloc[:,:j]
        df2=df.iloc[:,j+1:]
        df_train=pd.concat([df1,df2],axis=1)
        df_test=df_train.drop(df_train.index[l]).reset_index(drop=True)
        return df_train,df_test

def Euclidean_distance(X,y,n): # It calculates Euclidean distance between two points
    dist=0
    for i in range(n):
        dist+=np.square((X[i])-y[i])
    return np.sqrt(dist)

def distancesum(X,n): # This returns the sum of distance of one axis
    X.sort()
    res=0 ; sum=0;
    for i in range(n):
        res+=(X[i]*i-sum) ; sum+=X[i]
    return res

def Manhattan_distance(X,y,n): # It calculates Manhattan distance
    return distancesum(X,n)+distancesum(y,n)


def model_prediction(df_t,testing_instance,k,str1,flag=False): # This makes predictions by calculating either type of distance- Manhattan or Euclidean
    distances=[] ; targets=[]
    length=df_t.shape[1]
    if(str1=='Euclidean'): # calculates Euclidean distance
        for i in range(df_t.shape[0]):
            distances.append([Euclidean_distance(list(df_t.iloc[i,:]),testing_instance,length),i])
    else:
        for i in range(df_t.shape[0]): # calculates Manhattan distance
            distances.append([Manhattan_distance(list(df_t.iloc[i,:]),testing_instance,length),i])

    if(flag): # If it is true, then the algorithm becomes Weighted KNN
        sum=0
        for i in range(len(distances)):
            if(distances[i][0]==0):
                distances[i][0]=0
            else:
                distances[i][0]=1/distances[i][0]
            sum+=distances[i][0]
        for i in range(len(distances)):
            distances[i][0]=distances[i][0]/sum
    distances=sorted(distances) # sorts distance according to distance value
    for i in range(k):
        index=distances[i][1]
        targets.append(index)
    return collections.Counter(targets).most_common(1)[0][0] # returns the maximum voted instance

def knn(df_train,df_test,k,str1,flag=False): # str1 tells distance method to be used and flag tells whether it is KNN or weighted KNN
    neighbors=[]
    for i in range(df_test.shape[0]):
        testing_instance=list(df_test.iloc[i,:])
        if(flag): # for weighted KNN
            neighbors.append(model_prediction(df_train, testing_instance, k, str1,True))
        else: # for normal KNN
            neighbors.append(model_prediction(df_train, testing_instance, k, str1))
    return neighbors # returns instance numbers which are most near to the  testing instances



df=pd.read_csv(r'C:\Users\SAT SAHIB\Documents\Lakehead\Big Data\Assignment-1\Answers\Loading Data\encoded_data.csv') # loads csv file into Data Frame
n_rows=df.shape[0] ; n_cols=df.shape[1]
mean_1=int(np.mean(list(df['Height (cms)'])))
mean_2=int(np.mean(list(df['Weight (kgs)'])))
mean_3=0
x=5
accuracy=[]
for i in range(3):
    random.seed((i+1)*11) # seed value is changed depending so that new random numbers are generated each time
    l=random.sample(range(0, n_rows-1), x) # list of row indices that  are to be randomly imputed
    for j in range(3):
        df1=createMissingValue(df,l,j) # data frame containing missing values
        print('The missing values in column '+str(j+1)+' are as follows:')
        print(df1[df1.index.isin(l)])
        df_train,df_test=create_Train_Test_DataFrame(df1,j,n_cols,l) # creates training and testing data frames
        #print(df_train.head())
        neighborsE1=knn(df_train,df_test,1,'Euclidean') # 1-NN with Euclidean distance
        neighborsE5=knn(df_train,df_test,5,'Euclidean') # 5-NN with Euclidean distance
        neighborsM1=knn(df_train,df_test,1,'Manhattan') # 1-NN with Manhattan distance
        neighborsM5=knn(df_train,df_test,5,'Manhattan') # 5-NN with Manhattan distance
        neighborsWE=knn(df_train,df_test,5,'Euclidean',True) # Weighted 5-NN with Euclidean distance
        neighborsWM=knn(df_train,df_test,5,'Manhattan',True) # Weighted 5-NN with Manhattan distance
        #print((neighborsE1))

        for k in range(len(l)): #imputed values to according to 1-NN Euclidean
                df1.iloc[l[k],j]=df1.iloc[neighborsE1[k],j]

        accuracy.append(np.sum([df.iloc[z,j]==df1.iloc[z,j] for z in l])/len(l))
        print('The imputed values in column '+str(j+1)+' using 1-NN and Euclidean distance are:')
        print(df1[df1.index.isin(l)])
        for k in range(len(l)): # Imputes values according to 5-NN Euclidean
                df1.iloc[l[k],j]=df1.iloc[neighborsE5[k],j]

        accuracy.append(np.sum([df.iloc[z,j]==df1.iloc[z,j] for z in l])/len(l))
        print('The imputed values in column '+str(j+1)+' using 5-NN and Euclidean distance are:')
        print(df1[df1.index.isin(l)])
        for k in range(len(l)): # Imputes values according to 1-NN Manhattan
                df1.iloc[l[k],j]=df1.iloc[neighborsM1[k],j]

        accuracy.append(np.sum([df.iloc[z,j]==df1.iloc[z,j] for z in l])/len(l))
        print('The imputed values in column '+str(j+1)+' using 1-NN and Manhattan distance are:')
        print(df1[df1.index.isin(l)])
        for k in range(len(l)): # Imputes values according to 5-NN Manhattan
                df1.iloc[l[k],j]=df1.iloc[neighborsM5[k],j]

        accuracy.append(np.sum([df.iloc[z,j]==df1.iloc[z,j] for z in l])/len(l))
        print('The imputed values in column '+str(j+1)+' using 5-NN and Manhattan distance are:')
        print(df1[df1.index.isin(l)])
        for k in range(len(l)): # Imputes values according to Weighted KNN using Euclidean distance
                df1.iloc[l[k],j]=df1.iloc[neighborsWE[k],j]

        accuracy.append(np.sum([df.iloc[z,j]==df1.iloc[z,j] for z in l])/len(l))
        print('The imputed values in column '+str(j+1)+' using weighted 5-NN and Euclidean distance are:')
        print(df1[df1.index.isin(l)])
        for k in range(len(l)): # Imputes values according to Weighted KNN using Manhattan distance
                df1.iloc[l[k],j]=df1.iloc[neighborsWM[k],j]

        accuracy.append(np.sum([df.iloc[z,j]==df1.iloc[z,j] for z in l])/len(l))
        print('The imputed values in column '+str(j+1)+' using weighted 5-NN and Manhattan distance are:')
        print(df1[df1.index.isin(l)])
    x=2*x
data=[accuracy[i:i + 6] for i in range(0, len(accuracy), 6)] # creates data frame for different accuracy measures
col_names=['E1','E5','M1','M5','WE','WM']
data_frame=pd.DataFrame(data=data,columns=col_names)
column_imputation=['Col1_5','Col2_5','Col3_5','Col1_10','Col2_10','Col3_10','Col1_20','Col2_20','Col3_20']
data_frame['Column with Imputation%']=column_imputation
print(data_frame.head())
# Selecting the accuracy measures of continuous features
df1=data_frame.iloc[:2,:]
df2=data_frame.iloc[3:5,:]
df3=data_frame.iloc[6:8,:]
df_continuous=df1.append([df2,df3],ignore_index=True)
print(df_continuous.head())
# Selecting the accuracy measures of categorical features
s1=data_frame.iloc[2,:]
s2=data_frame.iloc[5,:] ; s3=data_frame.iloc[8,:]
df_combined=pd.DataFrame()
df_categorical=df_combined.append([s1,s2,s3],ignore_index=True)
print(df_categorical.head())
results_continous=df_continuous.to_csv(r'C:\Users\SAT SAHIB\Documents\Lakehead\Big Data\Question1\results_continuous.csv',index=None)
results_categorical=df_categorical.to_csv(r'C:\Users\SAT SAHIB\Documents\Lakehead\Big Data\Question1\results_categorical.csv',index=None)
accuracy_csv=data_frame.to_csv(r'C:\Users\SAT SAHIB\Documents\Lakehead\Big Data\Question1\accuracy_csv.csv',index=None)
data_frame.plot() ; plt.show()