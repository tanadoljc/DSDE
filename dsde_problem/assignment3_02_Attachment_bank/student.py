
import pandas as pd #e.g. pandas, sklearn, .....
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import numpy as np
import warnings # DO NOT modify this line
from sklearn.exceptions import ConvergenceWarning # DO NOT modify this line
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore", category=ConvergenceWarning) # DO NOT modify this line


class BankLogistic:
    def __init__(self, data_path): # DO NOT modify this line
        self.data_path = data_path
        self.df = pd.read_csv(data_path, sep=',')
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def Q1(self): # DO NOT modify this line
        """
        Problem 1:
            Load ‘bank-st.csv’ data from the “Attachment”
            How many rows of data are there in total?

        """
        # TODO: Paste your code here
        return self.df.shape[0]

    def Q2(self): # DO NOT modify this line
        """
        Problem 2:
            return the tuple of numeric variables and categorical variables are presented in the dataset.
        """
        # TODO: Paste your code here
        num = len(self.df.select_dtypes(include=['number']).columns)
        return (num, self.df.shape[1] - num)  
    
    def Q3(self): # DO NOT modify this line
        """
        Problem 3:
            return the tuple of the Class 0 (no) followed by Class 1 (yes) in 3 digits.
        """
        cnt = round(self.df['y'].value_counts(normalize=True),3)
        return (cnt['no'], cnt['yes'])
    

    def Q4(self): # DO NOT modify this line
        """
        Problem 4:
            Remove duplicate records from the data. What are the shape of the dataset afterward?
        """
        # TODO: Paste your code here
        
        self.df.drop_duplicates(inplace=True)
        return self.df.shape
        

    def Q5(self): # DO NOT modify this line
        """
        Problem 5:
            5. Replace unknown value with null
            6. Remove features with more than 99% flat values. 
                Hint: There is only one feature should be drop
            7. Split Data
            -	Split the dataset into training and testing sets with a 70:30 ratio.
            -	random_state=0
            -	stratify option
            return the tuple of shapes of X_train and X_test.

        """
        # TODO: Paste your code here
        self.df.drop_duplicates(inplace=True)
        self.df.replace('unknown', np.nan, inplace=True)

        flat_cols = [col for col in self.df.columns if self.df[col].value_counts(normalize=True).iloc[0] > 0.99]
        self.df.drop(columns=flat_cols, inplace=True)

        X = self.df.drop(columns=['y'])
        y = self.df['y']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        return (self.X_train.shape, self.X_test.shape)


    def Q6(self): 
        """
        Problem 6: 
            8. Impute missing
                -	For numeric variables: Impute missing values using the mean.
                -	For categorical variables: Impute missing values using the mode.
                Hint: Use statistics calculated from the training dataset to avoid data leakage.
            9. Categorical Encoder:
                Map the ordinal data for the education variable using the following order:
                education_order = {
                    'illiterate': 1,
                    'basic.4y': 2,
                    'basic.6y': 3,
                    'basic.9y': 4,
                    'high.school': 5,
                    'professional.course': 6,
                    'university.degree': 7} 
                Hint: Use One hot encoder or pd.dummy to encode ordinal category
            return the shape of X_train.

        """
        # TODO: Paste your code here
        self.Q5()

        num_imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        cat_imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        numeric_columns = self.X_train.select_dtypes(include=['number']).columns.tolist()
        test_numeric_columns = self.X_train.select_dtypes(include=['number']).columns.tolist()

        for col in self.X_train.columns:
            if col in numeric_columns:
                self.X_train[col] = num_imputer.fit_transform(self.X_train[[col]]).ravel()
                self.X_test[col] = num_imputer.fit_transform(self.X_test[[col]]).ravel()
            else:
                self.X_train[col] = cat_imputer.fit_transform(self.X_train[[col]]).ravel()
                self.X_test[col] = cat_imputer.fit_transform(self.X_test[[col]]).ravel()

        education_order = {
            'illiterate': 1,
            'basic.4y': 2,
            'basic.6y': 3,
            'basic.9y': 4,
            'high.school': 5,
            'professional.course': 6,
            'university.degree': 7
        }

        self.X_train['education'] = self.X_train['education'].map(education_order)
        self.X_test['education'] = self.X_test['education'].map(education_order)

        self.X_train = pd.get_dummies(self.X_train, columns=[col for col in self.X_train.columns if col not in numeric_columns and col !='education'], drop_first=False)
        self.X_test = pd.get_dummies(self.X_test, columns=[col for col in self.X_test.columns if col not in test_numeric_columns and col!='education'], drop_first=False)

        return self.X_train.shape

    
    def Q7(self):
        ''' Problem7: Use Logistic Regression as the model with 
            random_state=2025, 
            class_weight='balanced' and 
            max_iter=500. 
            Train the model using all the remaining available variables. 
            What is the macro F1 score of the model on the test data? in 2 digits
        '''
        # TODO: Paste your code here
        self.Q6()
        model = LogisticRegression(
            random_state=2025,
            class_weight='balanced',
            max_iter=500
        )

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        macro_f1 = f1_score(self.y_test, y_pred, average='macro')

        return  round(macro_f1, 2)
        


   