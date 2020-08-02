import pandas as pd

class predictionDataLoader():
    """
                            Class Name: predictionDataLoader
                            Description: Loads the prediction data.
                            Output: Returns a CSV file containing the prediction data
                            On Failure: Raise Exception

                            Written By: Vaishnavi Ambati
                            Version: 1.0
                            Revisions: None
    """

    def __init__(self, logger_obj, log_file):
        self.loggerObj = logger_obj
        self.prediction_data_logs = log_file

    def load_prediction_data(self, filename):
        """
                                Method Name: load_prediction_data
                                Description: Loads the prediction data.
                                Output: Returns a CSV file containing the prediction data
                                On Failure: Raise Exception

                                Written By: Vaishnavi Ambati
                                Version: 1.0
                                Revisions: None
        """

        #put the path and the filename
        self.prediction_data_logs = self.loggerObj.write_log(self.prediction_data_logs,'Entered load_prediction_data method of predictionDataLoader class.')
        self.prediction_data_logs = self.loggerObj.write_log(self.prediction_data_logs, 'Loading of prediction Data has Started.')
        try:
            #loading the data
            data = pd.read_csv('instance\\files\\' + filename)

            self.prediction_data_logs = self.loggerObj.write_log(self.prediction_data_logs,'Data has been loadded succesfully')
            self.prediction_data_logs = self.loggerObj.write_log(self.prediction_data_logs,'Exiting load_prediction_data method of predictionDataLoader class.')
            return data

        except Exception as e:
            self.prediction_data_logs = self.loggerObj.write_log(self.prediction_data_logs,'Data has not loaded. The Exception occured is ' + str(e))
            self.prediction_data_logs = self.loggerObj.write_log(self.prediction_data_logs,'Exiting load_prediction_data method of predictionDataLoader class.')
            self.prediction_data_logs.to_csv('Logs\\Prediction Validation\\prediction_validation_logs.csv', index = False)

            raise Exception
