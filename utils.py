import pandas as pd 
# Makes sure we see all columns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

class DataLoader():
    def __init__(self, path):
        self.data = None
        self.path = path

    def load_dataset(self):
        self.data = pd.read_csv(self.path)

    def preprocess_data(self):
        # Remove the other category in Gender as it only has one observation
        self.data.drop(self.data[self.data['gender'] == 'Other'].index, inplace = True)
        
        # One-hot encode all categorical columns
        categorical_cols = ["gender",
                            "ever_married",
                            "work_type",
                            "Residence_type",
                            "smoking_status"]
        encoded = pd.get_dummies(self.data[categorical_cols], 
                                prefix=categorical_cols)

        # Update data with new columns
        self.data = pd.concat([encoded, self.data], axis=1)
        self.data.drop(categorical_cols, axis=1, inplace=True)

        # Impute missing values for columns containing missing values

        # Find columns with missing values
        for c in range(len(self.data.columns)):
            if self.data.isnull().sum()[c] != 0:
                i = self.data.columns[c]
                # fill with median value
                self.data.loc[self.data.loc[:,i].isnull(),i]=self.data.loc[:,i].median()
        
        # Drop id as it is not relevant
        self.data.drop(["id"], axis=1, inplace=True)

        # Standardization 
        # Usually we would standardize here and convert it back later
        # But for simplification we will not standardize / normalize the features
        # Do this if it is an assumption of the models we use

    def get_data_split(self):
        X = self.data.iloc[:,:-1]
        y = self.data.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)
        # And validation data for ANN
        X_test, X_val, y_test, y_val = train_test_split(X_train, y_train, test_size=0.50, random_state=2) # 0.25 x 0.8 = 0.2
        return X_train, X_test, y_train, y_test, X_val, y_val
    
    # Oversampling to allow the algorithm to see more of the minority group while training
    def oversample(self, X_train, y_train):
        oversample = RandomOverSampler(sampling_strategy='minority')
        # Convert to numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        # Convert back to pandas
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        return x_over, y_over

