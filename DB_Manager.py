import sqlite3

class DBManager():
    def __init__(self):
        # self.connection = sqlite3.connect("cse2050_students_db.db")
        
        self.connection = sqlite3.connect('NerualDatabase.db')
        # Create table if their isnt one
        sql_create_table = """ CREATE TABLE IF NOT EXISTS History (
                                        ID INTERGER,
                                        Title VARCHAR(100),
                                        Model VARCHAR(100),
                                        Activation VARCHAR(100),
                                        L1 FLOAT,
                                        L2 FLOAT,
                                        Dropout VARCHAR(100),
                                        Momentum FLOAT,
                                        Alpha FLOAT,
                                        Epochs INTEGER,
                                        Note VARCHAR(500),
                                        TrainingValAccuracy VARCHAR(100),
                                        TrainingAccuracy VARCHAR(100),
                                        Date DATE
                                    ); """
        self.cursor = self.connection.cursor()
        self.cursor.executescript(sql_create_table)
        
    def close_database(self):
        self.connection.close()