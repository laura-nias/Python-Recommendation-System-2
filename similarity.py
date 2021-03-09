#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from scipy.spatial.distance import jaccard, correlation
from sklearn.neighbors import NearestNeighbors

class Similarity_Metrics:
    def __init__(self, data, id_1, id_2):
            self.__data = data
            self.__id_1 = id_1
            self.__id_2 = id_2
            self.__vec_a = []
            self.__vec_b = []
            self.__artist_list = []
            
    def get_feature(self):
        return input("\nLIST OF FEATURES\n\nAll Features\nAcousticness\nDanceability\nEnergy\nInstrumentalness\nLiveness\nLoudness\nPopularity\nSpeechiness\nTempo\nValence\n\nWhich feature would you like to use? ").strip().lower()
    
    def get_all_features(self):
        return ['acousticness','danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'popularity', 'speechiness', 'tempo', 'valence']
        
    def manhattan_similarity(self):
        feature = self.get_feature()
        all_features = self.get_all_features()
        
        if(feature == "all features"):
            for i in all_features:
                self.__vec_a.append(float(self.__data[i][self.__id_1]))
                self.__vec_b.append(float(self.__data[i][self.__id_2]))
            
        elif(feature in all_features):
            self.__vec_a.append(float(self.__data[feature][self.__id_1]))
            self.__vec_b.append(float(self.__data[feature][self.__id_2]))
          
        else:
            print("ERROR: The input doesn't match any feature")
            self.manhattan_similarity()
        
        self.__vec_a = np.array(self.__vec_a)
        self.__vec_b = np.array(self.__vec_b)
            
        return np.sum(np.abs(self.__vec_a - self.__vec_b))
    
    def jaccard_similarity(self):
        feature = self.get_feature()
        all_features = self.get_all_features()
        
        if(feature == "all features"):
            for i in all_features:
                self.__vec_a.append(float(self.__data[i][self.__id_1]))
                self.__vec_b.append(float(self.__data[i][self.__id_2]))
                
            self.__vec_a = np.array(self.__vec_a)
            self.__vec_b = np.array(self.__vec_b)
            
            intersection = np.logical_and(self.__vec_a, self.__vec_b)
            union = np.logical_or(self.__vec_a, self.__vec_b)
            return intersection.sum() / float(union.sum())
      
        elif(feature in all_features):
            a = self.__data[feature][self.__id_1]
            b = self.__data[feature][self.__id_2]

            a = set(a)
            b = set(b)

            intersection = np.logical_and(a, b)
            union = np.logical_or(a, b)

            return intersection/float(union)
        else:
            print("ERROR: The input doesn't match any feature")
            self.jaccard_similarity()
            
    def euclidean_similarity(self):
        feature = self.get_feature()
        all_features = self.get_all_features()
        
        if(feature == "all features"):
            for i in all_features:
                self.__vec_a.append(float(self.__data[i][self.__id_1]))
                self.__vec_b.append(float(self.__data[i][self.__id_2]))
                
        elif(feature in all_features):
            self.__vec_a.append(float(self.__data[feature][self.__id_1]))
            self.__vec_b.append(float(self.__data[feature][self.__id_2]))
            
        else:
            print("ERROR: The input doesn't match any feature")
            self.euclidean_similarity()
            
        self.__vec_a = np.array(self.__vec_a)
        self.__vec_b = np.array(self.__vec_b)

        return np.linalg.norm(self.__vec_a - self.__vec_b)
    
    def cosine_similarity(self):
        feature = self.get_feature()
        all_features = self.get_all_features()
        
        if(feature == "all features"):
            for i in all_features:
                self.__vec_a.append(float(self.__data[i][self.__id_1]))
                self.__vec_b.append(float(self.__data[i][self.__id_2]))
  
        elif(feature in all_features):
            self.__vec_a.append(float(self.__data[feature][self.__id_1]))
            self.__vec_b.append(float(self.__data[feature][self.__id_2]))

        else:
            print("ERROR: The input doesn't match any feature")
            self.cosine_similarity()
       
        self.__vec_a = np.array(self.__vec_a)
        self.__vec_b = np.array(self.__vec_b)
        return np.dot(self.__vec_a, self.__vec_b) / (np.linalg.norm(self.__vec_a) * np.linalg.norm(self.__vec_b))
    
    def pearson_similarity(self):
        feature = self.get_feature()
        all_features = self.get_all_features()
        
        if(feature == "all features"):
            for i in all_features:
                self.__vec_a.append(float(self.__data[i][self.__id_1]))
                self.__vec_b.append(float(self.__data[i][self.__id_2]))
  
        elif(feature in all_features):
            self.__vec_a.append(float(self.__data[feature][self.__id_1]))
            self.__vec_b.append(float(self.__data[feature][self.__id_2]))

        else:
            print("ERROR: The input doesn't match any feature")
            self.pearson_similarity()
            
        self.__vec_a = np.array(self.__vec_a)
        self.__vec_b = np.array(self.__vec_b)

        try:
            return np.corrcoef(self.__vec_a, self.__vec_b)
        except ZeroDivisionError:
            return "Similarity cannot be computed for this feature"
        except IOError as e:
            errno, strerror = e.args
            print("I/O error({0}): {1}".format(errno,strerror))
        
    def recommendations_from_similarity(self, chosen_metric, n = 11):
        data = self.__data
        
        #drop columns that aren't needed - values that aren't numeric and/or not between 1 and 0
        new_data = data.drop(['row_id', 'artists', 'id', 'name', 'release_date', 'year', 'explicit', 'key', 'mode', 'duration_ms', 'tempo', 'loudness', 'popularity', 'instrumentalness'], axis = 1)
        #turn into numpy array
        new_data = new_data.to_numpy()
            
        #for euclidean, manhattan and cosine
        if(chosen_metric == "euclidean" or chosen_metric == "manhattan" or chosen_metric == "cosine"):
            #fit data for nearest neighbours - euclidean, manhattan or cosine
            nbrs = NearestNeighbors(n_neighbors=n, algorithm='auto', metric=chosen_metric).fit(new_data)
        elif(chosen_metric == "jaccard"):
            #fit data for nearest neighbours - jaccard
            nbrs = NearestNeighbors(n_neighbors=n, algorithm='auto', metric=jaccard).fit(new_data) 
        elif(chosen_metric == "pearson"):
            #fit data for nearest neighbours - based on pearson
            nbrs = NearestNeighbors(n_neighbors=n, algorithm='auto', metric=correlation).fit(new_data) 
            
        nbrs.kneighbors_graph(new_data[int(self.__id_1)].reshape(1,-1)).toarray()

        neighbor_index = nbrs.kneighbors(new_data[int(self.__id_1)].reshape(1,-1), return_distance=False)
        #delete first as it is the given id
        neighbor_index = np.delete(neighbor_index, 0)
        
        return neighbor_index
        
    def similar_artists_to_artist(self, metric):
        #call recommendations_from_similarity and print top 10 artists from select artist features
        values = []
        recommend = []
        #passed in 100 so there's a very small chance of getting less than 10 unique artists
        res = self.recommendations_from_similarity(metric, 100)
        
        #retrieve the artist name originally chosen
        try:
            chosen_artist = self.__data.iloc[self.__data.index[self.__id_1]]['artists']
        
            #make a list of unique artists - remove duplicates while retaining the order of similarity
            for i in range(0,len(res)):
                #don't include the artist originally chosen
                if(self.__data.iloc[self.__data.index[res[i]]]['artists'] != chosen_artist):
                    values.append(self.__data.iloc[self.__data.index[res[i]]])
                    self.__artist_list.append(self.__data.iloc[self.__data.index[res[i]]]['artists'])  
            #used dict so that it retains order
            final_list = list(dict.fromkeys(self.__artist_list))
        
        
            #print top 10 from the unique list
            print("\nTop 10 Artist Recommendations:")
            for q in range(0,10):
                print(final_list[q])
                recommend.append(values[q])

            #Calls the method to create graphs - uncomment to see graphs produced
            #self.accuracy(metric, recommend)
            
        except IOError as e:
            errno, strerror = e.args
            print("I/O error({0}): {1}".format(errno,strerror))

    def similar_music_to_music(self, metric):
        #call recommendations_from_similarity and print top 10 music tracks from select music features
        values = []
        
        res = self.recommendations_from_similarity(metric)
        #prints list in format 'name of song'
        print("\nTop 10 Song Recommendations:")
        for q in range(0,10):
            print(self.__data.iloc[self.__data.index[res[q]]]['name'])
            values.append(self.__data.iloc[self.__data.index[res[q]]])
        
        #Calls the method to create graphs - uncomment to see graphs produced
        #self.accuracy(metric, value)

    def similar_music_to_artist(self, metric):
        #call recommendations_from_similarity and print top 10 music tracks
        values = []
        
        res = self.recommendations_from_similarity(metric)
        #prints list in format 'name of song'
        print("\nTop 10 Song Recommendations:")
        for q in range(0,10):
            print(self.__data.iloc[self.__data.index[res[q]]]['name'])
            values.append(self.__data.iloc[self.__data.index[res[q]]])
        
        #Calls the method to create graphs - uncomment to see graphs produced
        #self.accuracy(metric, value)
            
    def accuracy(self, metric, recommend):
        #create accuracy charts
        
        #labels for bars
        labels = ['acousticness','danceability', 'energy','liveness','speechiness', 'valence']
        
        #values for bars
        orig = self.get_original_values(labels, recommend)
        rec_means = self.get_rec_values(labels, recommend)

        #set labels and width
        x = np.arange(len(labels))  # the label locations
        width = 0.4

        #create bars and key
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, orig, width, label='Original')
        rects2 = ax.bar(x + width/2, rec_means, width, label='Recommended')

        #create title and axis labels
        ax.set_ylabel('Score')
        ax.set_title('Recommendation Accuracy by Feature - {0}'.format(metric))
        ax.set_xlabel('Features')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        #make values appear above bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(round(height, 2)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        #print graph
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.xticks(fontsize=8)
        plt.show()
        
    def get_original_values(self, labels, recommend):
        #get values for chosen id
        vec_1 = []
        
        for i in labels:
            vec_1.append(self.__data[i][self.__id_1])
        return vec_1

    def get_rec_values(self, labels, recommend):
        #get values for recommended ids
        vec_2 = []
        value = []
        sum_values = 0
        
        for j in labels:
            result = []
            for i in range(0,len(recommend)):
                result.append(recommend[i][j])
            vec_2.append(result)
        
        for k in range(0, len(vec_2)):
            for l in range(0,10):
                sum_values = float(vec_2[k][l])
                l+=1
            
            sum_values = sum_values
            value.append(sum_values)
        return value
        
        
        
        
            
    

