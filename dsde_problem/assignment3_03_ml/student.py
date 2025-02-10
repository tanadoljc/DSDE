import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class Clustering:
    def __init__(self, file_path): # DO NOT modify this line
        #Add other parameters if needed
        self.file_path = file_path 
        self.df = pd.read_csv(self.file_path) #parameter for loading csv

    def Q1(self): # DO NOT modify this line
        """
        Step1-4
            1. Load the CSV file.
            2. Choose edible mushrooms only.
            3. Only the variables below have been selected to describe the distinctive
               characteristics of edible mushrooms:
               'cap-color-rate','stalk-color-above-ring-rate'
            4. Provide a proper data preprocessing as follows:
                - Fill missing with mean
                - Standardize variables with Standard Scaler
        """

        df_edible = self.df[self.df['label'] == 'e']

        df_selected = df_edible[['cap-color-rate','stalk-color-above-ring-rate']]

        df_selected = df_selected.fillna(df_selected.mean())

        scaler = StandardScaler()
        df_selected_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=['cap-color-rate','stalk-color-above-ring-rate']) 

        return df_selected_scaled.shape


    def Q2(self): # DO NOT modify this line
        """
        Step5-6
            5. K-means clustering with 5 clusters (n_clusters=5, random_state=0, n_init='auto')
            6. Show the maximum centroid of 2 features ('cap-color-rate' and 'stalk-color-above-ring-rate') in 2 digits.
        """
        # remove pass and replace with you code
        df_edible = self.df[self.df['label'] == 'e']

        df_selected = df_edible[['cap-color-rate','stalk-color-above-ring-rate']]

        df_selected = df_selected.fillna(df_selected.mean())

        scaler = StandardScaler()
        df_selected_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=['cap-color-rate','stalk-color-above-ring-rate']) 

        # --------------------
        kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto')
        kmeans.fit(df_selected_scaled)

        # Get centroids
        centroids = kmeans.cluster_centers_

        # Find the maximum centroid values
        max_centroid = centroids.max(axis=0)

        # Display the maximum centroid values rounded to 2 decimal places
        max_centroid_rounded = [round(float(val), 2) for val in max_centroid]
        return np.array(max_centroid_rounded, dtype=float)
        

    def Q3(self): # DO NOT modify this line
        """
        Step7
            7. Convert the centroid value to the original scale, and show the minimum centroid of 2 features in 2 digits.

        """
        df_edible = self.df[self.df['label'] == 'e']

        df_selected = df_edible[['cap-color-rate','stalk-color-above-ring-rate']]

        df_selected = df_selected.fillna(df_selected.mean())

        scaler = StandardScaler()
        df_selected_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=['cap-color-rate','stalk-color-above-ring-rate']) 

        # --------------------
        kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto')
        kmeans.fit(df_selected_scaled)

        centroids_scaled = kmeans.cluster_centers_
        
        # Convert centroids back to original scale
        centroids_original = scaler.inverse_transform(centroids_scaled)

        # Find the minimum centroid values
        min_centroid = centroids_original.min(axis=0)

        # Round to 2 decimal places
        min_centroid_rounded = np.round(min_centroid, 2)

        return np.array(min_centroid_rounded, dtype=float)