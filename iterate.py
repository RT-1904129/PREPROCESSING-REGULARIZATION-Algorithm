import csv
import numpy as np
import matplotlib.pyplot as plt

Lambda=0.1
threshold1_for_feature_selection=-0.4
threshold2_for_feature_selection=0.7
Threshold_value_of_difference_of_cost_values=0.0000001

# This function read the training csv file
def Import_data():
    X=np.genfromtxt("train_X_pr.csv",delimiter=',',dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_pr.csv",delimiter=',',dtype=np.float64)
    return X,Y

def replace_null_values_with_mean(X):
    mean_of_nan=np.nanmean(X,axis=0)
    index=np.where(np.isnan(X))
    X[index]=np.take(mean_of_nan,index[1])
    return X

def mean_normalize(X, column_indices):
    mini=np.min(X[:,column_indices],axis=0)
    maxi=np.max(X[:,column_indices],axis=0)
    mean=np.mean(X[:,column_indices],axis=0)
    X[:,column_indices]=(X[:,column_indices]-mean)/(maxi-mini)
    return X

def get_correlation_matrix(X):
    num_vars = len(X[0])
    m = len(X)
    correlation_matix = np.zeros((num_vars,num_vars))
    for i in range(0,num_vars):
        for j in range(i,num_vars):
            mean_i = np.mean(X[:,i])
            mean_j = np.mean(X[:,j])
            std_dev_i = np.std(X[:,i])
            std_dev_j = np.std(X[:,j])
            numerator = np.sum((X[:,i]-mean_i)*(X[:,j]-mean_j))
            denominator = (m)*(std_dev_i)*(std_dev_j)
            corr_i_j = numerator/denominator    
            correlation_matix[i][j] = corr_i_j
            correlation_matix[j][i] = corr_i_j
    return correlation_matix


def select_features(corr_mat, T1, T2):
    filter_feature=[0]
    m=len(corr_mat)
    for i in range(1,m):
        if(abs(corr_mat[i][0])>T1):
            filter_feature.append(i)
    removed_feature=[]
    n=len(filter_feature)
    select_features=list(filter_feature)
    for i in range(0,n):
        for j in range(i+1,n):
            f1=filter_feature[i]
            f2=filter_feature[j]
            if (f1 not in removed_feature) and (f2 not in removed_feature):
                if(abs(corr_mat[f1][f2])>T2):
                    selected_features.remove(f2)
                    removed_feature.append(f2)
                    
    return select_features

def data_processing(class_X) :
    X=replace_null_values_with_mean(class_X)
    for i in range(class_X.shape[1]):
        X=mean_normalize(X,i)
    
    correlation_matrix= get_correlation_matrix(X)
    selected_feature_list=select_features(correlation_matrix,threshold1_for_feature_selection,threshold2_for_feature_selection)
    X=X[:,selected_feature_list]
    return X

def sigmoid_function(Z):
    s=1.0/(1.0+np.exp(-Z))
    return s

def compute_cost(X, Y, W, b,Lambda):
    Z=np.dot(X,W.T)+b
    sigmoid_value=sigmoid_function(Z)
    sigmoid_value[sigmoid_value==1]=0.9999 #it is used to avoid nan value vy sum function as log(0) is not defined 
    sigmoid_value[sigmoid_value==0]=0.0001  #similarliy when a value is much larege or log(1) its result is zero.on that time we get nan
    sum=np.sum(np.multiply(Y,np.log(sigmoid_value))+np.multiply((1-Y),np.log(1-sigmoid_value)))
    regulization_cost=(Lambda/2)*np.sum(np.square(W))
    return(((-1.0)*sum+regulization_cost)/len(Y))

def compute_gradient_of_cost_function(X, Y, W, b,Lambda):
    Z=np.dot(X,W.T)+b
    sigmoid_value=sigmoid_function(Z)
    db=np.sum(sigmoid_value-Y)/len(Y)
    dw=(np.dot((sigmoid_value-Y).T,X)+Lambda*(W))/len(Y)
    return dw,db


def Optimize_weights_using_gradient_descent(X,Y,W,b,learning_rate):
    i=1;
    prev_cost_value=0
    while True:
        dw,db=compute_gradient_of_cost_function(X,Y,W,b,Lambda)
        W=W-(learning_rate*dw)
        b=b-(learning_rate*db)
        cost_value=compute_cost(X,Y,W,b,Lambda)
        if (i%100000)==0:
            print("i value ",i)
        if abs(cost_value-prev_cost_value)<(Threshold_value_of_difference_of_cost_values):
            print("final no of iteration",i)
            break
        prev_cost_value=cost_value
        i+=1
    return (W,b)


def train_model(X,Y,learning_rate):
    Y=Y.reshape(X.shape[0],1)
    W=np.zeros((1,X.shape[1]))
    b=0
    W,b=Optimize_weights_using_gradient_descent(X,Y,W,b,learning_rate)
    return (W,b)

    
if __name__=="__main__":
    X,Y=Import_data()
    X=data_processing(X)
    learning_rate=0.09
    weights,b_value=train_model(X,Y,learning_rate)

