import numpy as np
import csv
import sys

from validate import validate

selected_features=[0, 1, 2, 3, 4, 5, 6]


def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights

def replace_null_values_with_mean(X_test):
    mean_of_nan=np.nanmean(X_test,axis=0)
    index=np.where(np.isnan(X_test))
    X_test[index]=np.take(mean_of_nan,index[1])
    return X_test

def mean_normalize(X_test, column_indices):
    mini=np.min(X_test[:,column_indices],axis=0)
    maxi=np.max(X_test[:,column_indices],axis=0)
    mean=np.mean(X_test[:,column_indices],axis=0)
    X_test[:,column_indices]=(X_test[:,column_indices]-mean)/(maxi-mini)
    return X_test


def data_processing(X_test) :
    X_test=replace_null_values_with_mean(X_test)
    for i in range(X_test.shape[1]):
        X_test=mean_normalize(X_test,i)
    X_test=X_test[:,selected_features]
    return X_test
        
        
def sigmoid_function(Z):
    s=1.0/(1.0+np.exp(-Z))
    return s

def predict_target_values(test_X, weights):
    b_value=weights[0]
    W_value=weights[1:]
    Z=np.dot(test_X,W_value.T)+b_value
    sigmoid_value=sigmoid_function(Z)
    predicted_value=[]
    for i in range(len(test_X)):
        if(sigmoid_value[i]>=0.5):
            predicted_value.append(1)
        else:
            predicted_value.append(0)
    
    predicted_value=np.array(predicted_value)
    predicted_value=predicted_value.reshape(test_X.shape[0],1)
    return predicted_value
        
    

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    test_X=data_processing(test_X)
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_pr.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_pr.csv") 