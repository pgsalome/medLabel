
import numpy as np # Lib for multi-dimensional arrays and matrices handling
import pandas as pd # Lib for data manipulation and analysis

from sklearn import svm # SVM model for RFE Feature Selection
from sklearn.preprocessing import * # Data pre-processing and standardization
from sklearn.preprocessing import MinMaxScaler # Feature scaling
from sklearn.pipeline import Pipeline # To fit hyperparameters into 1 pipeline
from sklearn.model_selection import train_test_split # Splits data into indices of training and testing
from imblearn.over_sampling import SMOTE # Oversample data using SMOTE algorithm

import warnings # Lib for warning issue handling
warnings.filterwarnings('ignore') # Ignores all irrelevant warnings
from collections import Counter # To get / set count of elements
from os import path

%matplotlib inline
import matplotlib.pyplot as plt # Lib for interactive plots
plt.style.use('seaborn-white') # Sets theme of visualization (seaborn-ticks / whitegrid) are similar to white
import seaborn as sns # Matplotlib based lib - better interface for drawing attractive and informative statistical graphics
sns.set_palette(['#FC4B60','#06B1F0'])
random_seed = 63445

# Framework / Platform for building ML models
import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.layers import Input, Dense, Activation, GaussianNoise, LeakyReLU, Dropout
tensorflow.config.set_visible_devices([],'GPU')
print("TensorFlow Version:",tensorflow.__version__)

# from google.colab import drive
# drive.mount('/content/drive/')
data_path = '/home/e210/Downloads/gbm-20230903T122921Z-001/gbm'  #Change this path accordingly


# Helper function to print details
def print_dataframe_details(df, name):
    print(f"Size of {name} Dataset:", df.shape)
    nFeatures = len(df.columns) - 3  # ignoring ID, OS, and OS_IND columns
    print("Number of features:", nFeatures)
    if '_OS_IND' in df.columns:
        print(df['_OS_IND'].value_counts())  # Prints the count of 0 and 1 in the class column
    print("----")

print("Reading Multi-omics Data")

datasets = {
    'GENE_MiRNA': 'GENE_MiRNA.csv',
    'GENE_METHY': 'GENE_METHY.csv',
    'MiRNA_METHY': 'MiRNA_METHY.csv',
    'GENE_RAD': 'GENE_RAD.csv',
    'MIRNA_RAD': 'MIRNA_RAD.csv',
    'METHY_RAD': 'METHY_RAD.csv',
    'GENE_MIRNA_RAD': 'GENE_MIRNA_RAD.csv',
    'GENE_MIRNA_METHY': 'GENE_MIRNA_METHY.csv',
    'GENE_MIRNA_METHY_RAD': 'GENE_MIRNA_METHY_RAD.csv'
}

dataframes = {}  # Dictionary to store DataFrames

for name, file in datasets.items():
    df = pd.read_csv(path.join(data_path, file), index_col=0)
    dataframes[name] = df
    print_dataframe_details(df, name)


def near_zero_var(df, freq_cut=95/5, unique_cut=10):
    """
    Identify columns with near-zero variance.
    
    Parameters:
    - df: DataFrame to process.
    - freq_cut: The cutoff for the ratio of the most common value to the second most common value.
    - unique_cut: The minimum number of unique values the column must have.
    
    Returns:
    A list of column names with near-zero variance.
    """
    # Determine columns to drop
    to_drop = []
    
    for col in df.columns:
        # Compute the number of unique values and their frequencies
        freqs = df[col].value_counts()
        
        # If there are fewer unique values than unique_cut
        if len(freqs) < unique_cut:
            to_drop.append(col)
            continue
        
        # Calculate ratio of frequencies of top 2 most common values
        ratio = freqs.iloc[0] / freqs.iloc[1] if len(freqs) > 1 else float('inf')
        
        if ratio > freq_cut:
            to_drop.append(col)
    
    return to_drop

print("Multi-omics data imported successfully")
df = dataframes["GENE_MIRNA_RAD"]
nFeatures = len(df.columns) - 3
df

features = df.iloc[:,1:-1] #Retrieves all rows (1:), leaves last column (,1:-1)
target = df.iloc[:,-1] #Retrieves all rows (1:), retrieves only last column (,-1)

print("Number of features :", features.shape[1])
print("Number of targets :", target.shape[0])


#Setting all dataset into a range of 0 to 1
min_max_scaler = MinMaxScaler(feature_range =(0, 1))

## remove zero variance
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
selected_data = selector.fit_transform(features)
features = pd.DataFrame(selected_data, columns=features.columns[selector.get_support()])

# Scaled feature
features = pd.DataFrame(min_max_scaler.fit_transform(features))

#Function to Define SVM model accuracy
def accuracy(model, features, target):
    prediction = model.predict(features)
    print ("Accuracy of model:", accuracy_score(target, prediction) * 100, "%")

warnings.filterwarnings("ignore")

def plot_coefficient_distribution(coef, features):
    """
    Plots the coefficient distribution using a histogram.

    Parameters:
    - coef: The coefficients, expected to be a 2D array where coef[0] contains the coefficient values.
    - features: The dataframe or data structure containing the feature names.

    Returns:
    None. A plot is displayed.
    """
    
    nFeatures = len(features.columns)
    coefficients = [coef[0][i] for i in range(nFeatures)]
    
    # Plotting the distribution
    plt.figure(figsize=(15,7))
    plt.hist(coefficients, bins=30, edgecolor='black')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Feature Coefficients')
    plt.tight_layout()
    plt.show()

    
#Set Parameter
C = 1.0
rfeIndex = nFeatures
#Create SVM model using a linear kernel
model = svm.SVC(kernel='linear', C=C).fit(features,target)
coef = model.coef_

#Print co-efficients of features
for i in range(0, nFeatures):
    print (features.columns[i],":", coef[0][i])

#Find the minimum weight among features and eliminate the feature with the smallest weight
min = coef[0][0]
# Maximum number of elimination iterations
# Adjust this value based on how many eliminations you want to perform, to keep x features
x = 12000
max_elimination_iterations = nFeatures-x

for j in range(max_elimination_iterations):
    index = 0
    for i in range(0,rfeIndex): # Iterates until the final feature
        if min > coef[0][i]:
            index = index + 1
            min = coef[0][i]

    if len(features.columns) == 1:
        print ("After recursive elimination we have the", features.columns[index], "feature with a score of:", min)
    
    else:
        print ("Lowest feature weight is for", features.columns[index], "with a value of:", min)
        print ("Dropping feature", features.columns[index])
        features.drop(features.columns[index], axis = 1, inplace = True)
        rfeIndex = rfeIndex - 1
        nFeatures = nFeatures - 1

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")
ax = sns.countplot(data=df, x='_OS_IND')

# Annotate bars with counts
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

ax.set(xlabel='Class', ylabel='Frequency')

# Adjust y-axis range to show counts for both classes
plt.ylim(0, max(ax.patches[0].get_height(), ax.patches[1].get_height()) + 20)

plt.show()

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state=random_seed, stratify=target)
print ("X TRAIN DATA SHAPE: ", x_train.shape)
print ("X TEST DATA SHAPE: ", x_test.shape)
print ("Y TRAIN DATA SHAPE: ", y_train.shape)
print ("Y TEST DATA SHAPE: ", y_test.shape)

sm = SMOTE(k_neighbors=1, random_state=random_seed)
X, Y = sm.fit_resample(x_train, y_train)
print ('Shape of oversampled data: {}'.format(X.shape))
print ('Shape of Y: {}'.format(Y.shape))

# Stop training model when "monitor parameter" has stopped improving
earlyStopping = EarlyStopping(monitor='loss', patience=50) # patience is num of epochs to reach early stopping
# Save model after every epoch
checkpointer = ModelCheckpoint(filepath='MO_SDAE_Training.h5', verbose=1, save_best_only=True)
input_dims = (len(features)-1)

def dae (inputX, input_dims, output_dims, epoch, activation, loss, opti):

    model = Sequential()

    if input_dims > 5000:
        with tensorflow.device('/cpu:0'):
            print("Using CPU....")
            model.add(Dense(input_dims, input_dim = input_dims))
            model.add(GaussianNoise(0.5))
            model.add(Dense(output_dims, activation = activation, kernel_regularizer = regularizers.l1(0.01)))
            model.add(Dense(input_dims, activation= activation))
            model.compile(loss = loss, optimizer = opti)
            model.fit(inputX, inputX, epochs = epoch, batch_size = 4)
            model.summary()
    else:
        with tensorflow.device('/cpu:0'):
            print("Using GPU....")
            model.add(Dense(input_dims, input_dim = input_dims))
            model.add(GaussianNoise(0.5))
            model.add(Dense(output_dims, activation= activation, kernel_regularizer = regularizers.l1(0.01)))
            model.add(Dense(input_dims, activation= activation))
            model.compile(loss = loss, optimizer = opti)
            model.fit(inputX, inputX, epochs = epoch, batch_size = 1)
            model.summary()

    return model

autoencoder = dae(X,
                  input_dims = X.shape[1],
                  output_dims = 500,
                  epoch = 10,
                  activation = 'relu',
                  loss = 'mse',
                  opti = 'adam',
                 )

# Hyper-parameter
layers = [X.shape[1],10000,8000, 4000, 2000, 500] # Setting the size of your layer
epoch = 5
optimizer = 'adam'
activation = 'relu'
loss = 'mse'

# @title
def sdae_pretrain (inputX, layers, activation, epoch, optimizer, loss):

    encoder = []
    decoder = []
    ae = []

    for i in range(len(layers)-1):
            print("Now pretraining layer {} [{}-->{}]".format(i+1, layers[i], layers[i+1]))

            input_dims = layers[i]
            output_dims = layers[i+1]

            autoencoder = dae(inputX, input_dims, output_dims, epoch, activation, loss, optimizer)
            enc = Sequential()
            enc.add(Dense(output_dims, input_dim=input_dims))
            enc.compile(loss=loss, optimizer=optimizer)
            enc.layers[0].set_weights(autoencoder.layers[2].get_weights())
            inputX = enc.predict(inputX)
            print("check dimension : ", inputX.shape)
            enc.summary()
            encoder.append(autoencoder.layers[2].get_weights())
            decoder.append(autoencoder.layers[3].get_weights())
            ae.append(autoencoder)

    return ae

Train_SDAE = sdae_pretrain( X, layers = layers, activation = activation, epoch = epoch, optimizer = optimizer, loss = loss)

for i, m in enumerate(Train_SDAE):
    filename="pretrain_model" + str(i) + ".hd5"
    m.save(filename)

from tensorflow.keras import backend as K

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision2 = precision(y_true, y_pred)
    recall2 = recall(y_true, y_pred)
    return 2*((precision2 * recall2)/(precision2 + recall2 + K.epsilon()))

def fine_tuning(weights, inputX, inputXt, inputY, inputYt, layers, epoch, batch, optimizer, loss):

    encoder = []
    decoder = []

    for i in range(len(Train_SDAE)):

        encoder.append(Train_SDAE[i].layers[2].get_weights())
        decoder.append(Train_SDAE[i].layers[3].get_weights())

    with tensorflow.device('/device:gpu:0'): #I have to put this because the model size is too big for my GPU
        ft = Sequential()
        ft.add(Dense(layers[0], input_dim=layers[0]))
        ft.add(GaussianNoise(0.5))

        for i in range(len(layers)-1):
            ft.add(Dense(layers[i+1], activation = 'relu', weights = encoder[i], kernel_regularizer = regularizers.l2(0.01))) # Initial regularizer (l1_l2)

        for i in reversed(range(len(layers)-1)):
            ft.add(Dense(layers[i], activation = 'relu', weights = decoder[i]))
    ft.add(Dropout(0.2))
    ft.add(Dense(200, activation = 'relu'))
    ft.add(Dense(150, activation = 'relu', use_bias=True))
    ft.add(Dense(100, activation = 'relu', kernel_initializer= "glorot_uniform", bias_initializer= "zeros"))
    ft.add(Dense(50, activation= 'relu', kernel_initializer= "glorot_uniform", bias_initializer= "zeros"))
    ft.add(Dense(1, activation = 'sigmoid'))

    ft.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', recall, precision, f1])
    History = ft.fit(X, Y, epochs = epoch, batch_size = batch, validation_data = (x_test, y_test))
    ft.summary()

    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('SDAE Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

    return ft
Result = fine_tuning(Train_SDAE, X, x_test, Y, y_test, layers = layers, epoch = 50, batch = 4, optimizer = 'adam', loss = 'binary_crossentropy')
