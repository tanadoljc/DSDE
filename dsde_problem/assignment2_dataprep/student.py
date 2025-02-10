import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

"""
    ASSIGNMENT 2 (STUDENT VERSION):
    Using pandas to explore Titanic data from Kaggle (titanic_to_student.csv) and answer the questions.
    (Note that the following functions already take the Titanic dataset as a DataFrame, so you don’t need to use read_csv.)

"""

def Q1(df):
    """
        Problem 1:
            How many rows are there in the "titanic_to_student.csv"?
    """
    # TODO: Code here
    
    return df.shape[0]


def Q2(df):
    '''
        Problem 2:
            Drop unqualified variables
            Drop variables with missing > 50%
            Drop categorical variables with flat values > 70% (variables with the same value in the same column)
            How many columns do we have left?
    '''
    df.dropna(axis = 1, thresh=0.5*df.shape[0], inplace=True)

    for col in df.columns:
        if df[col].value_counts().max() > 0.7 * df.shape[0]:
            df = df.drop(col, axis = 1)
    
    # TODO: Code here
    return df.shape[1]


def Q3(df):
    '''
       Problem 3:
            Remove all rows with missing targets (the variable "Survived")
            How many rows do we have left?
    '''
    # TODO: Code here
    df.dropna(subset=['Survived'], inplace=True)
    return df.shape[0]


def Q4(df):
    '''
       Problem 4:
            Handle outliers
            For the variable “Fare”, replace outlier values with the boundary values
            If value < (Q1 - 1.5IQR), replace with (Q1 - 1.5IQR)
            If value > (Q3 + 1.5IQR), replace with (Q3 + 1.5IQR)
            What is the mean of “Fare” after replacing the outliers (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    # TODO: Code here
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1

    df.loc[lambda df:df['Fare'] < (Q1 - 1.5*IQR), ['Fare']] = Q1 - 1.5*IQR
    df.loc[lambda df:df['Fare'] > (Q3 + 1.5*IQR), ['Fare']] = Q3 + 1.5*IQR

    # print(df['Fare'].iloc[13])
    return round(df['Fare'].mean(), 2)


def Q5(df):
    '''
       Problem 5:
            Impute missing value
            For number type column, impute missing values with mean
            What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    # TODO: Code here
    missing_column = df.select_dtypes(include=['number'])
    missing_column = missing_column.columns[missing_column.isnull().any()]
    num_imp = SimpleImputer(strategy='mean')
    df[missing_column] = num_imp.fit_transform(df[missing_column])

    return round(df['Age'].mean(), 2)


def Q6(df):
    '''
        Problem 6:
            Convert categorical to numeric values
            For the variable “Embarked”, perform the dummy coding.
            What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    # TODO: Code here
    df['Embarked'] = df['Embarked'].fillna('Unknown')
    oneHotEncoder = OneHotEncoder(sparse_output=False)
    oneHotEncoded = oneHotEncoder.fit_transform(df[['Embarked']])
    oneHotEncoded_df = pd.DataFrame(oneHotEncoded, columns=oneHotEncoder.get_feature_names_out(['Embarked']))
    # print(oneHotEncoded_df)
    return round(oneHotEncoded_df['Embarked_Q'].mean(), 2)


def Q7(df):
    '''
        Problem 7:
            Split train/test split with stratification using 70%:30% and random seed with 123
            Show a proportion between survived (1) and died (0) in all data sets (total data, train, test)
            What is the proportion of survivors (survived = 1) in the training data (round 2 decimal points)?
            Hint: Use function round(_, 2), and train_test_split() from sklearn.model_selection, 
            Don't forget to impute missing values with mean.
    '''
    # TODO: Code here
    df['Survived'] = df['Survived'].fillna(df['Survived'].mode()[0])

    # impute missing values with mean
    missing_column = df.select_dtypes(include=['number'])
    missing_column = missing_column.columns[missing_column.isnull().any()]
    df[missing_column] = df[missing_column].fillna(df[missing_column].mean())

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=123, stratify=df['Survived'])
    count = train_df['Survived'].value_counts(normalize='True')
    return round(count[1.0], 2)


df  = pd.read_csv('assignment2_dataprep/titanic_to_student.csv')
print(Q7(df))