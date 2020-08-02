import ast
import numpy as np
import pandas as pd

from sklearn.feature_extraction import FeatureHasher
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


class processData:
    """
                            Class Name: processData
                            Description: Preprocess the data.
                            Input: logger_obj, log_file
                            On Failure: Raise Exception

                            Written By: Murali Krishna Chintha
                            Version: 1.0
                            Revisions: None
    """

    def __init__(self, logger_object, log_file):

        self.loggerObj = logger_object
        self.log_file = log_file

    def cuisine_process(self, cuisine):
        """
                                    Method Name: cuisine_process
                                    Description: process the cuisines column of the data.
                                    Input: cuisine
                                    Input Type: string(str)

                                    Written By: Murali Krishna Chintha
                                    Version: 1.0
                                    Revisions: None
        """

        if not isinstance(cuisine,str):
            return 'Not Available'
        else:
            cuisine = cuisine.replace('[', '')
            cuisine = cuisine.replace(']', '')
            cuisine_list = []
            for i in cuisine.split(', '):
                cuisine_list.append(i.replace("\'", ''))

            return cuisine_list

    def review_to_words(self, review):
        """
                                            Method Name: review_to_words
                                            Description: converting every review in review column into a list of words.
                                            Input: review
                                            Input Type: string(str)

                                            Written By: Murali Krishna Chintha
                                            Version: 1.0
                                            Revisions: None
                """
        if not isinstance(review,str):
            return np.NaN

        else:
            try:
                output = ast.literal_eval(review)

                if len(output[0]) >= 1:
                    return output[0]
                else:
                    return np.NaN
            except:
                return np.NaN

    def sentiment_analyzer(self, review):
        """
             Method Name: sentiment_analyzer
             Description: Analyses the sentiment of the review.
             Input: review
             Input Type: string(str)
             Onput: 1 - Postive Sentiment, 0 - Negative Sentiment

             Written By: Murali Krishna Chintha
             Version: 1.0
             Revisions: None
        """


        review_tag = sia.polarity_scores(review)

        # For no reviews condition
        if review_tag['compound'] > 0.3:
            return 1  # For "Postive" sentiment
        else:
            return 0  # for 'Negative' sentiment

    def preprocess_training_data(self, dataframe):
        """
             Method Name: preprocess_training_data
             Description: Preprocess the training data.
             Input: dataframe
             Input Type: dataframe
             Onput: Returns a processsed dataframe.

             Written By: Murali Krishna Chintha
             Version: 1.0
             Revisions: None
        """

        try:

            self.log_file = self.loggerObj.write_log(self.log_file, 'Entered preprocess_training_data of dataProcessor class')
            self.log_file = self.loggerObj.write_log(self.log_file, 'Data preprocessing has been initiated.')

            # renaming the columns replacing ' ' with '_'.
            dataframe.rename(
                columns={'Name': 'name', 'City': 'city', 'Cuisine Style': 'cuisine_style', 'Ranking': 'ranking',
                         'Price Range': 'price_range', 'Number of Reviews': 'no_of_reviews',
                         'Reviews': 'reviews', 'URL_TA': 'url_ta', 'ID_TA': 'id_ta', 'Rating': 'rating'},
                inplace=True)
            # processing cuisine_style
            dataframe['cuisine_style'] = dataframe.apply(lambda row: self.cuisine_process(row['cuisine_style']), axis=1)
            # adding a new column to the dataframe
            dataframe['no_of_cuisines'] = dataframe.apply(
                lambda row: len(row['cuisine_style']) if row['cuisine_style'] != 'Not Available' else 0, axis=1)
            # processing price_range
            dataframe['price_range'] = dataframe['price_range'].map({'$': 'cheap',
                                                                     '$$ - $$$': 'medium',
                                                                     '$$$$': 'high',
                                                                     })

            dataframe['price_range'].fillna('medium', inplace=True)

            # dropping rows in rating with values -1
            drop_index = list(dataframe[dataframe['rating'] == -1].index)
            dataframe.drop(drop_index, inplace=True)

            dataframe['rating'] = dataframe['rating'].map(
                {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 1.5: 6, 2.5: 7, 3.5: 8, 4.5: 9})
            # converting reviews to list of reviews
            dataframe['reviews'] = dataframe.apply(lambda row: self.review_to_words(row['reviews']), axis=1)

            # considering only relevant features
            features = ['city', 'ranking', 'price_range', 'no_of_reviews', 'no_of_cuisines', 'rating','reviews']
            df_features = dataframe[features]
            # dropping the null values
            df_features = df_features.dropna()

            # onehotencoding price_range
            df_feat = pd.concat([df_features.drop(['price_range'], axis=1), pd.get_dummies(df_features['price_range'])],
                                axis=1)

            df_feat.reset_index(inplace=True, drop=True)


            # feature hashing city column
            fh = FeatureHasher(n_features=7, input_type='string')
            hashed_features = fh.fit_transform(df_feat['city'])
            hashed_features = hashed_features.toarray()
            hashed_df = pd.DataFrame(hashed_features,
                                     columns=['city_1', 'city_2', 'city_3', 'city_4', 'city_5', 'city_6', 'city_7'])
            df_hashed = pd.concat([df_feat.drop('city', axis=1), hashed_df],
                                  axis=1)

            df_hashed['review1'] = df_hashed.apply(lambda row: row['reviews'][0], axis=1)
            df_hashed['review2'] = df_hashed.apply(lambda row: row['reviews'][1] if len(row['reviews']) == 2 else np.NaN, axis=1)
            df_hashed = df_hashed.dropna()

            df_hashed['review1_sentiment'] = df_hashed.apply(lambda row: self.sentiment_analyzer(row['review1']), axis=1)
            df_hashed['review2_sentiment'] = df_hashed.apply(lambda row: self.sentiment_analyzer(row['review2']), axis=1)

            df_hashed.drop(['review1', 'review2', 'reviews'], axis=1, inplace=True)

            self.log_file = self.loggerObj.write_log(self.log_file,'Data Preprocessing has completed. Exiting the preprocess_training_data method of dataPreprocessor class.')

            return df_hashed

        except Exception as e:

            self.log_file = self.loggerObj.write_log(self.log_file,'An error occured in the preprocess_training_data method of dataPreprocessor class. The exception is ' + str(e))
            self.log_file = self.loggerObj.write_log(self.log_file,'Exiting the preprocess_training_data method of dataPreprocessor class.')
            self.log_file.to_csv("Logs\\Prediction Logs\\prediction_logs.csv")

            raise Exception

    def remove_columns(self, dataframe, columns):
        """
             Method Name: remove_columns
             Description: Drops the specified columns
             Input: dataframe, columns
             Input Type: dataframe - dataframe, columns - list
             Output: Returns a dataframe without the specified columns.

             Written By: Murali Krishna Chintha
             Version: 1.0
             Revisions: None
        """

        dataframe_dropped = dataframe.drop(columns, axis=1)
        self.log_file = self.loggerObj.write_log(self.log_file,
                                                 'Columns ' + columns + ' have been removed succesfully.')

        return dataframe_dropped

    def null_value_check(self, dataframe):
        """
             Method Name: null_value_check
             Description: Checks for null values in the dataset.
             Input: dataframe
             Input Type: dataframe - dataframe
             Output: Null Value dictionary containing the columns in which null values are present and its value, True/False

             Written By: Murali Krishna Chintha
             Version: 1.0
             Revisions: None
        """

        self.log_file = self.loggerObj.write_log(self.log_file,
                                                 'Entered null_value_check method of dataPreprocessor class.')
        self.log_file = self.loggerObj.write_log(self.log_file, 'Null value check has started.')

        try:
            null_value_dict = dict()

            for column, value in dataframe.isnull().sum().items():
                if value != 0:
                    null_value_dict[column] = value
            if len(null_value_dict) > 0:
                self.log_file = self.loggerObj.write_log(self.log_file,
                                                         'Null values been checked. There are no null values in the data.')
                self.log_file = self.loggerObj.write_log(self.log_file,
                                                         'Exiting the null_value_check method of the dataPreprocessor class.')
                return null_value_dict, True
            else:
                self.log_file = self.loggerObj.write_log(self.log_file,
                                                         'Null values been checked. There are no null values in the data.')
                self.log_file = self.loggerObj.write_log(self.log_file,
                                                         'Exiting the null_value_check method of the dataPreprocessor class.')
                return null_value_dict, False

        except Exception as e:
            self.log_file = self.loggerObj.write_log(self.log_file,
                                                     'An Exception has occured in the null_value_check method of dataPreprocessor class. The Exception is ' + str(
                                                         e))
            self.log_file = self.loggerObj.write_log(self.log_file,
                                                     'Exting the null_value_check method of the dataProcessor class.')
            self.log_file.to_csv("Logs\\Prediction Logs\\prediction_logs.csv")

            raise Exception

    def separate_label_feature(self, dataframe, label_column_name):
        """
             Method Name: separate_label_feature
             Description: Separates the input dataframe into two dataframe. One containing the only the specified column.
             Input: dataframe, label_column_name
             Input Type: dataframe - dataframe, label_column_name - string
             Output: Returns two dataframes X,Y

             Written By: Murali Krishna Chintha
             Version: 1.0
             Revisions: None
        """

        try:
            self.log_file = self.loggerObj.write_log(self.log_file, 'Seperating labels has started.')
            self.log_file = self.loggerObj.write_log(self.log_file,
                                                     'Entered seperate_label_feature of the dataPreprocessor class.')

            X = dataframe.drop(labels=label_column_name, axis=1)
            Y = dataframe[label_column_name]

            self.log_file = self.loggerObj.write_log(self.log_file, 'Data has been seperated succesfully.')
            self.log_file = self.loggerObj.write_log(self.log_file,
                                                     'Exiting the seperate_label_feature method of the dataPreprocessor class.')

            return X, Y
        except Exception as e:
            self.log_file = self.loggerObj.write_log(self.log_file,
                                                     'An exception occured in seperate_label_feature method of the dataPreprocessor class. The Exception is ' + str(
                                                         e))
            self.log_file = self.loggerObj.write_log(self.log_file,
                                                     'Exiting the seperate_label_feature method of the dataPreprocessor class.')
            self.log_file.to_csv("Logs\\Prediction Logs\\prediction_logs.csv")

            raise Exception()

    def preprocess_prediction_data(self, dataframe):
        """
             Method Name: preprocess_prediction_data
             Description: Preprocesses the prediction data.
             Input: dataframe
             Input Type: dataframe - dataframe
             Output: Returns dataframe

             Written By: Murali Krishna Chintha
             Version: 1.0
             Revisions: None
        """

        try:

            self.log_file = self.loggerObj.write_log(self.log_file,
                                                     'Entered preprocess_prediction_data of dataProcessor class')
            self.log_file = self.loggerObj.write_log(self.log_file, 'Data preprocessing has been initiated.')

            # renaming the columns replacing ' ' with '_'.
            dataframe.rename(
                columns={'Name': 'name', 'City': 'city', 'Cuisine Style': 'cuisine_style', 'Ranking': 'ranking',
                         'Price Range': 'price_range', 'Number of Reviews': 'no_of_reviews',
                         'Reviews': 'reviews', 'URL_TA': 'url_ta', 'ID_TA': 'id_ta'},
                inplace=True)
            # processing cuisine_style
            dataframe['cuisine_style'] = dataframe.apply(lambda row: self.cuisine_process(row['cuisine_style']), axis=1)
            # adding a new column to the dataframe
            dataframe['no_of_cuisines'] = dataframe.apply(
                lambda row: len(row['cuisine_style']) if row['cuisine_style'] != 'Not Available' else 0, axis=1)
            # processing price_range
            dataframe['price_range'] = dataframe['price_range'].map({'$': 'cheap',
                                                                     '$$ - $$$': 'medium',
                                                                     '$$$$': 'high',
                                                                     })

            dataframe['price_range'].fillna('medium', inplace=True)

            # converting reviews to list of reviews
            dataframe['reviews'] = dataframe.apply(lambda row: self.review_to_words(row['reviews']), axis=1)

            # considering only relevant features
            features = ['name','city', 'ranking', 'price_range', 'no_of_reviews', 'no_of_cuisines','reviews']
            df_features = dataframe[features]
            # dropping the null values
            df_features = df_features.dropna()

            # onehotencoding price_range
            df_feat = pd.concat([df_features.drop(['price_range'], axis=1), pd.get_dummies(df_features['price_range'])],
                                axis=1)

            df_feat.reset_index(inplace=True, drop=True)

            # feature hashing city column
            fh = FeatureHasher(n_features=7, input_type='string')
            hashed_features = fh.fit_transform(df_feat['city'])
            hashed_features = hashed_features.toarray()
            hashed_df = pd.DataFrame(hashed_features,
                                     columns=['city_1', 'city_2', 'city_3', 'city_4', 'city_5', 'city_6', 'city_7'])
            df_hashed = pd.concat([df_feat.drop('city', axis=1), hashed_df],
                                  axis=1)

            df_hashed['review1'] = df_hashed.apply(lambda row: row['reviews'][0], axis=1)
            df_hashed['review2'] = df_hashed.apply(lambda row: row['reviews'][1] if len(row['reviews']) == 2 else np.NaN, axis=1)
            df_hashed = df_hashed.dropna()

            df_hashed['review1_sentiment'] = df_hashed.apply(lambda row: self.sentiment_analyzer(row['review1']),
                                                             axis=1)
            df_hashed['review2_sentiment'] = df_hashed.apply(lambda row: self.sentiment_analyzer(row['review2']),
                                                             axis=1)

            df_hashed.drop(['review1', 'review2', 'reviews'], axis=1, inplace=True)

            self.log_file = self.loggerObj.write_log(self.log_file,'Data Preprocessing has completed. Exiting the preprocess_prediction_data method of dataPreprocessor class.')

            return df_hashed

        except Exception as e:

            self.log_file = self.loggerObj.write_log(self.log_file, "Exception occured in preprocess_prediction_data method of dataPreprocessor class. Exception is " + str(e))
            self.log_file = self.loggerObj.write_log(self.log_file,'Exiting the preprocess_prediction_data method of dataPreprocessor class.')

            self.log_file.to_csv("Logs\\Prediction Logs\\prediction_logs.csv")

            raise Exception

    def preprocess_single_predict_manual(self, feature_list):
        """
                     Method Name: preprocess_single_predict_manual
                     Description: Preprocesses the prediction data entered manually.
                     Input: feature_list
                     Input Type: list
                     Output: Returns dataframe

                     Written By: Murali Krishna Chintha
                     Version: 1.0
                     Revisions: None
                """

        try:
            self.log_file = self.loggerObj.write_log(self.log_file,'Entered preprocess_single_predict_manual of dataProcessor class')
            self.log_file = self.loggerObj.write_log(self.log_file, 'Data preprocessing has been initiated.')

            columns = ['name', 'city', 'ranking', 'no_of_reviews', 'no_of_cuisines', 'review1', 'review2', 'cheap',
                       'high', 'medium']

            features = feature_list[:-1]
            price = feature_list[-1]
            if price == 'cheap':
                features.extend([1, 0, 0])
            elif price == 'medium':
                features.extend([0, 0, 1])
            elif price == 'high':
                features.extend([0, 1, 0])

            feature_dic = dict(zip(columns, features))

            df = pd.DataFrame(feature_dic, index=[0])

            # feature hashing city column
            fh = FeatureHasher(n_features=7, input_type='string')
            hashed_features = fh.fit_transform(df['city'])
            hashed_features = hashed_features.toarray()
            hashed_df = pd.DataFrame(hashed_features,
                                     columns=['city_1', 'city_2', 'city_3', 'city_4', 'city_5', 'city_6', 'city_7'])
            df_hashed = pd.concat([df.drop('city', axis=1), hashed_df],
                                  axis=1)

            df_hashed['review1_sentiment'] = df_hashed.apply(lambda row: self.sentiment_analyzer(row['review1']),
                                               axis=1)
            df_hashed['review2_sentiment'] = df_hashed.apply(lambda row: self.sentiment_analyzer(row['review2']),
                                               axis=1)

            df_hashed.drop(['review1', 'review2'], axis=1, inplace=True)

            self.log_file = self.loggerObj.write_log(self.log_file,'Data Preprocessing has completed. Exiting the preprocess_single_predict_manual method of dataPreprocessor class.')

            return df_hashed

        except Exception as e:

            self.log_file = self.loggerObj.write_log(self.log_file, "Exception occured in preprocess_single_predict_manual method of dataPreprocessor class. Exception is " + str(e))
            self.log_file = self.loggerObj.write_log(self.log_file,'Exiting the preprocess_single_predict_manual method of dataPreprocessor class.')

            self.log_file.to_csv("Logs\\Prediction Logs\\prediction_logs.csv")

            raise Exception

