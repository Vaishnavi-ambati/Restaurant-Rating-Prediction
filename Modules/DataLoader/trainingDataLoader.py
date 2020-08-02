import pandas as pd



class data_loader:

    def __init__(self,logger_obj, log_file):

        self.logger_obj = logger_obj
        self.data_load_logs = log_file

    def load_data(self, filename):
        """
             Method Name: load_data
             Description: Loads the data from the specified location.
             Input: None
             Output: Returns Returns a dataframe.

             Written By: Murali Krishna Chintha
             Version: 1.0
             Revisions: None
        """


        self.data_load_logs = self.logger_obj.write_log(self.data_load_logs, 'Data Loading has initiated.')

        try:
            self.data = pd.read_csv('instance\\files\\' + filename)
            self.data.drop('Unnamed: 0', axis = 1, inplace = True)
            self.data_load_logs = self.logger_obj.write_log(self.data_load_logs, 'Data has been loaded succesfully')
            self.data_load_logs = self.logger_obj.write_log(self.data_load_logs, 'Exiting the load_data method of trainingDataLoader class.')

            return self.data
        except Exception as e:

            self.data_load_logs = self.logger_obj.write_log(self.data_load_logs, 'Data has not loaded. An exception occured in data_loader method of trainingDataLoader class. Exception message: ' + str(e))
            self.data_load_logs = self.logger_obj.write_log(self.data_load_logs, 'Exiting the load_data method of trainingDataLoader class.')
            self.data_load_logs.to_csv("Logs\\Training Validation\\training_validation_logs.csv")

            raise Exception
