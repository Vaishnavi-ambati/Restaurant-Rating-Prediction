import pandas as pd
import MySQLdb
import mysql.connector
from mysql.connector import Error


class db_insertion:
    """
                            Class Name: prediction_validation
                            Description: Validates the prediction data.
                            Output: CSV file containing the logs of Validation.
                            On Failure: Raise Exception

                            Written By: Vaishnavi Ambati
                            Version: 1.0
                            Revisions: None
    """

    def __init__(self, path, tablename):
        self.path = path
        self.tablename = tablename

    def db_insert_query(self):
        """
                                Method Name: db_insert_query
                                Description: Inserts given data into database.
                                Output: Success or Failure message
                                On Failure: Raise Exception

                                Written By: Vaishnavi Ambati
                                Version: 1.0
                                Revisions: None
        """
        try:
            #type in your password here
            connection = mysql.connector.connect(host="127.0.0.1", user="root", password="",
                                                 database="restaurantDB")

            cursor = connection.cursor()

            logs = pd.read_csv(self.path)
            # print(logs)
            for ind in logs.index:
                LogDate = str(logs['date'][ind])
                LogTime = str(logs['time'][ind])
                Log = str(logs['logs'][ind])
                cursor.execute('INSERT INTO ' + self.tablename + '(LogDate,LogTime,Log) VALUES ("{}","{}","{}")'.format(LogDate,LogTime,Log))

            connection.commit()
            cursor.close()

            return "Successfuly inserted into Database. "


        except Exception as e:

            print('Exception occurred in db_insert_query method of db_insertion class. The exception is ' + str(e))

            return "Insertion failed. "


