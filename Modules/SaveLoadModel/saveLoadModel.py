import os
import shutil
import pickle

class saveLoadModel:
    """
                            Class Name: saveLoadModel

                            Written By: Murali Krishna Chintha
                            Version: 1.0
                            Revisions: None
    """

    def __init__(self, loggerObj, log_file):
        self.loggerObj = loggerObj
        self.log_file = log_file
        self.model_directory = 'models/'

    def save_model(self,model,filename):
        """
                                Module Name: save_model
                                Description: Saves the model into a pickle file.
                                Input: model, filename
                                Output: Pickle file containing the model.
                                On Failure: Raise Exception

                                Written By: Murali Krishna Chintha
                                Version: 1.0
                                Revisions: None
        """

        try:
            self.log_file = self.loggerObj.write_log(self.log_file,'Entered save_model module of the saveLoadModel class.')
            path = os.path.join(self.model_directory,filename)
            if os.path.isdir(path):
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path) #
            with open(path +'/' + filename+'.sav',
                      'wb') as f:
                pickle.dump(model, f) # save the model to file
            self.log_file = self.loggerObj.write_log(self.log_file,'Model has been saved Successfully.')
            self.log_file = self.loggerObj.write_log(self.log_file,'Exiting save_model')

            return 'success'

        except Exception as e:
            self.log_file = self.loggerObj.write_log(self.log_file,'An exception has occured in save_model module of the saveLoadModel class.')
            self.log_file = self.loggerObj.write_log(self.log_file,'Exiting save_model module')
            self.log_file.to_csv("Logs\\TrainingLogs\\training_logs.csv")
            raise Exception()

    def model_name_finder(self):
        """
                                Module Name: model_name_finder
                                Description: Processes the name of the model
                                Input: None
                                Output: model name
                                On Failure: Raise Exception

                                Written By: Murali Krishna Chintha
                                Version: 1.0
                                Revisions: None
        """
        try:
            model_name = os.listdir(path=self.model_directory)[0]

            return model_name

        except Exception as e:
            self.log_file = self.loggerObj.write_log(self.log_file,"Model has not loaded. The exception is " + str(e))
            self.log_file.to_csv("Logs\\Prediction Logs\\prediction_logs.csv")

            raise Exception()

    def load_model(self):
        """
                                Module Name: load_model
                                Description: loads the model from a pickle file.
                                Input: filename
                                Output: model
                                On Failure: Raise Exception

                                Written By: Murali Krishna Chintha
                                Version: 1.0
                                Revisions: None
        """

        self.log_file = self.loggerObj.write_log(self.log_file, "Model loading has started")
        self.log_file = self.loggerObj.write_log(self.log_file, "Entered load_model method of saveLoadModel class.")
        filename = self.model_name_finder()

        try:
            with open(self.model_directory + filename + '/' + filename + '.sav',
                      'rb') as f:
                model = pickle.load(f)
                self.log_file = self.loggerObj.write_log(self.log_file, "Model" + filename +"has loaded Succesfully.")
                self.log_file = self.loggerObj.write_log(self.log_file, "Exiting load_model method of saveLoadModel class.")
                return model

        except Exception as e:

            self.log_file = self.loggerObj.write_log(self.log_file, "Model"+ filename +" has not loaded. The exception is "+ str(e))
            self.log_file = self.loggerObj.write_log(self.log_file, "Exiting load_model method of saveLoadModel class.")
            self.log_file.to_csv("Logs\\Prediction Logs\\prediction_logs.csv")

            raise Exception()
