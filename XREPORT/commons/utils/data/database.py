import os
import sqlite3
import pandas as pd

from XREPORT.commons.constants import PROCESSED_PATH
from XREPORT.commons.logger import logger

# [DATABASE]
###############################################################################
class XREPORTDatabase:

    def __init__(self, configuration):             
        self.db_path = os.path.join(PROCESSED_PATH, 'XREPORT_processed_dataset.db') 
        self.configuration = configuration        

    #--------------------------------------------------------------------------
    def save_to_database(self, data : pd.DataFrame): 
        # connect to sqlite database and save the preprocessed data as table
        conn = sqlite3.connect(self.db_path)         
        data.to_sql('XREPORT_dataset', conn, if_exists='replace')
        conn.commit()
        conn.close() 
        
    #--------------------------------------------------------------------------
    def load_from_database(self): 
        # Connect to the database and inject a select all query
        # convert the extracted data directly into a pandas dataframe          
        conn = sqlite3.connect(self.db_path)        
        data = pd.read_sql_query(f"SELECT * FROM XREPORT_dataset", conn)
        conn.close()  

        return data
    
    