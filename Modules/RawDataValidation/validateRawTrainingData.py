from Modules.appLogger import application_logger
import pandas as pd
import numpy as np


class validate_raw_data():

    def __init__(self, logger_object,log_file):
        self.log_file = log_file
        self.logger_object = logger_object

    def column_validate(self,dataframe):
        """
                                Module Name: column_validate
                                Description: Validates the number of columns and the column names.
                                Input: dataframe
                                Output: True/False
                                On Failure: Raise Exception

                                Written By: Murali Krishna Chintha
                                Version: 1.0
                                Revisions: None
        """
        try:
            self.log_file = self.logger_object.write_log(self.log_file, 'Data Validation has Stared.')

            columns = list(dataframe.columns)
            length_of_columns = len(columns)

            actual_columns = ['Name','City', 'Cuisine Style', 'Ranking', 'Rating', 'Price Range',
                              'Number of Reviews', 'Reviews', 'URL_TA', 'ID_TA']

            object_column_list = ['Name', 'City', 'Cuisine Style', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA']
            float_column_list = ['Ranking', 'Rating', 'Number of Reviews']

            object_columns = list(dataframe.dtypes[dataframe.dtypes == np.object].index)
            float_columns = list(dataframe.dtypes[dataframe.dtypes == np.float64].index)

            if object_columns == object_column_list and float_columns == float_column_list:
                data_type_validate = True
            else:
                data_type_validate = False


            if columns == actual_columns and data_type_validate:
                self.logger_object.write_log(self.log_file, 'Columns in the data are valid.')
                self.log_file = self.logger_object.write_log(self.log_file,
                                                                            'Exited column_validate module of validateRawTrainingData class.')

                return True
            elif columns == actual_columns and not data_type_validate:
                self.logger_object.write_log(self.log_file,
                                      'Columns in the dataset are valid but the column data types are not valid.')
                self.log_file = self.logger_object.write_log(self.log_file,
                                                                            'Exited column_validate module of validateRawTrainingData class.')

                return False
            else:
                self.logger_object.write_log(self.log_file, 'Columns in the data are not valid.')
                self.log_file = self.logger_object.write_log(self.log_file,
                                                                            'Exited column_validate module of validateRawTrainingData class.')

                return False


        except Exception as e:
            self.log_file = self.logger_object.write_log(self.log_file, 'An Exception has occured in column_validate of validateRawTrainingData class. The Exception is ' + str(e))
            self.log_file = self.logger_object.write_log(self.log_file,'Exting column_validate of validateRawTrainingData class.')
            self.log_file.to_csv('Logs\\Training Validation\\train_validation_logs.csv')



    def entire_column_null_value_check(self, dataframe):
        """
                                Module Name: entire_column_null_value_check
                                Description: Validates the number of columns and the column names.
                                Input: dataframe
                                Output: True/False
                                On Failure: Raise Exception

                                Written By: Murali Krishna Chintha
                                Version: 1.0
                                Revisions: None
        """
        try:
            self.log_file = self.logger_object.write_log(self.log_file, 'Null Values Check has started.')

            null_values_counts = dataframe.isnull().sum()
            length_of_dataframe = dataframe.shape[0]

            for column,null_value_count in null_values_counts.items():
                if null_value_count == 0.75 * length_of_dataframe:
                    self.log_file = self.logger_object.write_log(self.log_file,
                                                                     'The Column ' + str(column) + ' has only values. The Dataset Cannot be Accepted.')
                    return True, column
                else:
                    self.log_file = self.logger_object.write_log(self.log_file,
                                                                     'No column in the Dataset has complete null values.')

                    return False, np.NaN
            self.log_file = self.logger_object.write_log(self.log_file, 'Entire Row null value check has completed.')
        except Exception as e:
            self.log_file = self.logger_object.write_log(self.log_file,'An Exception has occured in entire_column_null_value_check of validateRawTrainingData class. The Exception is ' + str(e))
            self.log_file = self.logger_object.write_log(self.log_file,'Exting entire_column_null_value_check of validateRawPredicitonData class.')
            self.log_file.to_csv('Logs\\Training Validation\\train_validation_logs.csv')

