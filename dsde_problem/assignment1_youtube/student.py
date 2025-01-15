import pandas as pd
import json

"""
    ASSIGNMENT 1 (STUDENT VERSION):
    Using pandas to explore youtube trending data from GB (GBvideos.csv and GB_category_id.json) and answer the questions.
"""
path = 'assignment1_youtube/USvideos.csv'
json_path = 'data/GB_category_id.json'

def Q1():
    """
        1. How many rows are there in the GBvideos.csv after removing duplications?
        - To access 'GBvideos.csv', use the path '/data/GBvideos.csv'.
    """
    # TODO: Paste your code here
    df = pd.read_csv(path)
    df_no_dup = df.drop_duplicates()
    return df_no_dup['video_id'].count()

def Q2(vdo_df):
    '''
        2. How many VDO that have "dislikes" more than "likes"? Make sure that you count only unique title!
            - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    # TODO: Paste your code here
    df_no_dup = vdo_df.drop_duplicates()
    condition = df_no_dup['dislikes'] > df_no_dup['likes']
    return len(df_no_dup[condition].drop_duplicates('title'))

def Q3(vdo_df):
    '''
        3. How many VDO that are trending on 22 Jan 2018 with comments more than 10,000 comments?
            - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - The trending date of vdo_df is represented as 'YY.DD.MM'. For example, January 22, 2018, is represented as '18.22.01'.
    '''
    # TODO: Paste your code here
    df = vdo_df.drop_duplicates()
    condition = (df['trending_date'] == '18.22.01') & (df['comment_count'] > 10000)
    return df[condition]['video_id'].count()

def Q4(vdo_df):
    '''
        4. Which trending date that has the minimum average number of comments per VDO?
            - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    # TODO:  Paste your code here
    df = vdo_df.drop_duplicates()
    grouped_df = df.groupby('trending_date')['comment_count'].mean().idxmin()
    return grouped_df

def Q5(vdo_df):
    '''
        5. Compare "Sports" and "Comedy", how many days that there are more total daily views of VDO in "Sports" category than in "Comedy" category?
            - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - You must load the additional data from 'GB_category_id.json' into memory before executing any operations.
            - To access 'GB_category_id.json', use the path '/data/GB_category_id.json'.
    '''
    # TODO:  Paste your code here
    with open(json_path,'r') as file:
        jsn = json.load(file) 

    df = vdo_df.drop_duplicates()

    arr = []
    for item in jsn['items']:
        if item['snippet']['assignable'] == True:
            arr.append({'category_id':int(item['id']), 'category':item['snippet']['title']})
    
    arr_df = pd.DataFrame(arr) 
    df = df.merge(arr_df,left_on='category_id', right_on='category_id')
    grouped_df = df.groupby(['trending_date', 'category'])['views'].sum().unstack(fill_value=0)
    return grouped_df[grouped_df['Sports']>grouped_df['Comedy']].shape[0]

df = pd.read_csv(path)
print(df.describe(include='all'))