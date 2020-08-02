#logger to log every step of training and prediction.
#takes fileobject and text as input and logs the given text in a text file at the given fielobjects loction.
from datetime import datetime


class logger():

    def __init__(self):
        pass

    def write_log(self, dataframe, log_message):
        self.now = datetime.now()
        self.date = str(self.now.strftime("%Y-%m-%d"))
        self.time = self.now.strftime('%H:%M:%S')

        dataframe.loc[len(dataframe)] = [self.date, self.time, log_message]

        return dataframe