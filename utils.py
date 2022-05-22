import pandas as pd 
# Makes sure we see all columns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

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

    def get_data_split(self, ANN):
        X = self.data.iloc[:,:-1]
        y = self.data.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

        if ANN == True:
        # And validation data for ANN
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state= 8) # 0.25 x 0.8 = 0.2
            return X_train, X_test, X_val, y_train, y_test, y_val
        else: 

            return X_train, X_test, y_train, y_test
    
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
    
    def undersample(self, X_train, y_train):
        undersample = RandomUnderSampler()
        # Convert to numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = undersample.fit_resample(x_np, y_np)
        # Convert back to pandas
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        return x_over, y_over

    def SMOTE_oversample(self, X_train, y_train):
        oversample = SMOTE()
        X_train_resh, y_train_resh = oversample.fit_resample(X_train, y_train.ravel())
        return  X_train_resh, y_train_resh
    

from sklearn.metrics import confusion_matrix, accuracy_score

class Performance:
    def __init__(self):
        self = self

    def evaluate(self, test_labels, predictions):
        accuracy = accuracy_score(test_labels, predictions)
        cf = confusion_matrix(test_labels,predictions)
        TP = cf[1,1] # True positives
        FN = cf[1, 0] # False negatives
        recall = TP/(TP+FN)
        print('Model Performance')
        print('Accuracy = {:0.2f}%.'.format(accuracy*100))
        print('Recall strokes only = {:0.2f}%.'.format(recall*100))
    
        return accuracy


from interpret.blackbox import LimeTabular
from interpret import show
import shap
import dice_ml

# We make a class for the interpretability
class InterpretModel:
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def prepare_data(self):
        # We filter only the stroke incidents
        strokes_y_test = self.y_test.copy()
        strokes_X_test = self.X_test.copy()
        strokes_X_test.drop(strokes_y_test[strokes_y_test == 0].index, inplace = True)
        strokes_y_test.drop(strokes_y_test[strokes_y_test == 0].index, inplace = True)
        return strokes_X_test, strokes_y_test

    def LIME_method(self, model, X_training, X_testing, y_testing, start, end):
        lime = LimeTabular(predict_fn=model.predict_proba, 
                   data=X_training, 
                   random_state=1)        
        # Get local explanations for start:end predictions
        lime_local = lime.explain_local(X_testing[start:end], 
                                        y_testing[start:end], 
                                        name='LIME')

        return show(lime_local)

    def SHAP_method(self, model, X_testing, start, end):
        explainer = shap.TreeExplainer(model)
        # Calculate shapley values for test data
        shap_values = explainer.shap_values(X_testing[start:end])

        #Visualize local predictions
        shap.initjs()
        # Force plot
        prediction = model.predict(X_testing[start:end])[0]
        print(f"The {model} predicted: {prediction}")

        return shap.force_plot(explainer.expected_value[1], shap_values[1], X_testing[start:end]) # for values



    def CF_method(self, model, X_testing, start, end, data):

        # Dataset
        data_dice = dice_ml.Data(dataframe=data, 
                                # For perturbation strategy
                                continuous_features=['age', 
                                                    'avg_glucose_level',
                                                    'bmi'], 
                                outcome_name='stroke')
        # Model
        model_dice = dice_ml.Model(model=model, 
                                backend="sklearn")
        explainer = dice_ml.Dice(data_dice, 
                                model_dice, 
                                # Random sampling, genetic algorithm, kd-tree,...
                                method="random")
        # generate CF
        input_datapoint = X_testing[start:end]
        cf = explainer.generate_counterfactuals(input_datapoint, 
                                  total_CFs=3, 
                                  desired_class="opposite")   

        features_to_vary=['avg_glucose_level',
                        'bmi',
                        'smoking_status_smokes']
        permitted_range={'avg_glucose_level':[50,250],
                        'bmi':[18, 35], 
                        'smoking_status_smokes': ['0']} # only possible to change to non-smoker
        # Now generating explanations using the new feature weights
        cf = explainer.generate_counterfactuals(input_datapoint, 
                                        total_CFs=3, 
                                        desired_class="opposite",
                                        permitted_range=permitted_range,
                                        features_to_vary=features_to_vary)
        # Visualize it
        return cf.visualize_as_dataframe(show_only_changes=True)                

