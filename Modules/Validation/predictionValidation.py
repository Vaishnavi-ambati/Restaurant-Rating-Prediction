from Modules.appLogger import application_logger
import pandas as pd
from Modules.RawDataValidation import validateRawPredictionData
from Modules.DataLoader import predictionDataLoader


class prediction_validation:
    """
                            Class Name: prediction_validation
                            Description: Validates the prediction data.
                            Output: CSV file containing the logs of Validation.
                            On Failure: Raise Exception

                            Written By: Vaishnavi Ambati
                            Version: 1.0
                            Revisions: None
    """
    def __init__(self):
        # self variables
        try:
            self.prediction_validation_logs = pd.read_csv('Logs\\Prediction Validation\\prediction_validation_logs.csv')
        except:
            self.prediction_validation_logs = pd.DataFrame(columns=['date', 'time', 'logs'])
        self.log_writer = application_logger.logger()
        self.raw_validate = validateRawPredictionData.validate_raw_data(self.log_writer,self.prediction_validation_logs)
        self.data_loader = predictionDataLoader.predictionDataLoader(self.log_writer, self.prediction_validation_logs)

    def validate_predict_data(self, filename):
        """
                                Method Name: validate_predict_data
                                Description: Validates the prediction data.
                                Output: CSV file containing the logs of Validation.
                                On Failure: Raise Exception

                                Written By: Vaishnavi Ambati
                                Version: 1.0
                                Revisions: None
        """
        try:
            self.prediction_validation_logs = self.log_writer.write_log(self.prediction_validation_logs,'Prediction Data has been loaded Successfully.')
            self.prediction_validation_logs = self.log_writer.write_log(self.prediction_validation_logs,'Prediction Data Validation has started.')

            dataframe = self.data_loader.load_prediction_data(filename)

            # columns names and no. of columns check
            validate = self.raw_validate.column_validate(dataframe)

            if validate:
                self.prediction_validation_logs = self.log_writer.write_log(self.prediction_validation_logs,'Columns are Valid in the Training Data.')
            else:
                self.prediction_validation_logs = self.log_writer.write_log(self.prediction_validation_logs,'Columns are not valid in the Training Data.')

            # entire column null values check.
            null_values, column = self.raw_validate.precentage_column_null_value_check(dataframe)

            if not null_values:
                self.prediction_validation_logs = self.log_writer.write_log(self.prediction_validation_logs,'No column in the dataset has complete null values.')
            else:
                self.prediction_validation_logs = self.log_writer.write_log(self.prediction_validation_logs,str(column) + 'has complete null values. Dataset cannot be accepted.')
            self.prediction_validation_logs = self.log_writer.write_log(self.prediction_validation_logs, 'prediction data validation is complete.')

            self.prediction_validation_logs = self.log_writer.write_log(self.prediction_validation_logs, 'Exited validate_predict_data method of predictionValidation class.')
            self.prediction_validation_logs.to_csv('Logs\\Prediction Validation\\prediction_validation_logs.csv', index=False)

            if validate:
                return True
            else:
                return False


        except Exception as e:

            self.prediction_validation_logs = self.log_writer.write_log(self.prediction_validation_logs,'Exception occured in validate_predict_data method of predictionValidation class. The exception is ' + str(e))
            self.prediction_validation_logs = self.log_writer.write_log(self.prediction_validation_logs, 'Exited validate_predict_data method of predictionValidation class.')
            self.prediction_validation_logs.to_csv('Logs\\Prediction Validation\\prediction_validation_logs.csv', index=False)

            raise Exception
