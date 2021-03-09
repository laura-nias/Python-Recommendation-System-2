#!/usr/bin/env python
# coding: utf-8

import pandas as pd

class Load:
    def load_data(self):
        try:
            #read csv file and make dataframe
            data = pd.read_csv('data.csv')
            #remove [] and '' from artist values
            data.artists = data.artists.replace(['\[','\]','\''], ['','',''], regex=True)
            #remove duplicate rows based on artist and song titles, but keep first instance
            data = data.drop_duplicates(subset=['artists', 'name'], keep='first')
            #reset index after removing duplicates
            data = data.reset_index()
            #reset row_id so it matches index + 1 (so that it starts at 1, not 0)
            data.row_id = data.index + 1
            #drop index column as we are using row_id for this purpose
            data = data.drop(['index'], axis=1)
            return data
        except IOError as e:
            errno, strerror = e.args
            print("I/O error({0}): {1}".format(errno,strerror))

