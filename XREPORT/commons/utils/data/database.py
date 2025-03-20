import os
import sqlite3
import pandas as pd

from XREPORT.commons.constants import DATA_PATH
from XREPORT.commons.logger import logger

# [DATABASE]
###############################################################################
class XREPORTDatabase:

    def __init__(self, configuration):             
        self.db_path = os.path.join(DATA_PATH, 'XREPORT_database.db') 
        self.configuration = configuration 

    #--------------------------------------------------------------------------
    def load_source_data(self): 
        # Connect to the database and inject a select all query
        # convert the extracted data directly into a pandas dataframe          
        conn = sqlite3.connect(self.db_path)        
        data = pd.read_sql_query(f"SELECT * FROM SOURCE_DATA", conn)
        conn.close()  

        return data

    #--------------------------------------------------------------------------
    def load_preprocessed_data(self): 
        # Connect to the database and inject a select all query
        # convert the extracted data directly into a pandas dataframe          
        conn = sqlite3.connect(self.db_path)        
        data = pd.read_sql_query(f"SELECT * FROM PROCESSED_DATA", conn)
        conn.close()  

        return data       

    #--------------------------------------------------------------------------
    def save_source_data(self, data : pd.DataFrame): 
        # connect to sqlite database and save the preprocessed data as table
        conn = sqlite3.connect(self.db_path)         
        data.to_sql('SOURCE_DATA', conn, if_exists='replace')
        conn.commit()
        conn.close() 
        
    #--------------------------------------------------------------------------
    def save_preprocessed_data(self, processed_data : pd.DataFrame): 
        # Connect to the database and inject a select all query
        # convert the extracted data directly into a pandas dataframe          
        conn = sqlite3.connect(self.db_path)        
        processed_data.to_sql('PROCESSED_DATA', conn, if_exists='replace')
        conn.close()

    #--------------------------------------------------------------------------
    def save_image_statistics(self, data : pd.DataFrame): 
        # connect to sqlite database and save the preprocessed data as table
        conn = sqlite3.connect(self.db_path)         
        data.to_sql('IMAGE_STATISTICS', conn, if_exists='replace')
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_checkpoints_summary(self, data : pd.DataFrame): 
        # connect to sqlite database and save the preprocessed data as table
        conn = sqlite3.connect(self.db_path)         
        data.to_sql('CHECKPOINTS_SUMMARY', conn, if_exists='replace')
        conn.commit()
        conn.close()   

 
    
    