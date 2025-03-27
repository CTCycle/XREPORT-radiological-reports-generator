import os
import sqlite3
import pandas as pd

from XREPORT.commons.constants import DATA_PATH, SOURCE_PATH
from XREPORT.commons.logger import logger

# [DATABASE]
###############################################################################
class XREPORTDatabase:

    def __init__(self, configuration):             
        self.db_path = os.path.join(DATA_PATH, 'XREPORT_database.db') 
        self.source_path = os.path.join(SOURCE_PATH, 'XREPORT_dataset.csv')
        self.configuration = configuration 
        self.initialize_database()
        self.update_database()

    #--------------------------------------------------------------------------       
    def initialize_database(self):        
        # Connect to the SQLite database and create the database if does not exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        create_source_data_table = '''
        CREATE TABLE IF NOT EXISTS SOURCE_DATA (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image TEXT,
            text TEXT            
        );
        '''

        create_processed_data_table = '''
        CREATE TABLE IF NOT EXISTS PROCESSED_DATA (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            tokens TEXT
        );
        '''

        create_inference_data_table = '''
        CREATE TABLE IF NOT EXISTS GENERATED_REPORTS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image TEXT,
            report TEXT
            checkpoint TEXT
        );
        '''
      
        create_image_statistics_table = '''
        CREATE TABLE IF NOT EXISTS IMAGE_STATISTICS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            height INTEGER,
            width INTEGER,
            mean REAL,
            median REAL,
            std REAL,
            min REAL,
            max REAL,
            pixel_range REAL,
            noise_std REAL,
            noise_ratio REAL
        );
        '''       
        
        create_checkpoints_summary_table = '''
        CREATE TABLE IF NOT EXISTS CHECKPOINTS_SUMMARY (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_name TEXT,
            sample_size REAL,
            validation_size REAL,
            seed INTEGER,
            precision_bits INTEGER,
            epochs INTEGER,
            additional_epochs INTEGER,
            batch_size INTEGER,
            split_seed INTEGER,
            image_augmentation TEXT,
            image_height INTEGER,
            image_width INTEGER,
            image_channels INTEGER,
            jit_compile TEXT,
            jit_backend TEXT,
            device TEXT,
            device_id TEXT,
            number_of_processors INTEGER,
            use_tensorboard TEXT,
            lr_scheduler_initial_lr REAL,
            lr_scheduler_constant_steps REAL,
            lr_scheduler_decay_steps REAL
        );
        '''
        
        cursor.execute(create_source_data_table)  
        cursor.execute(create_processed_data_table)  
        cursor.execute(create_inference_data_table) 
        cursor.execute(create_image_statistics_table)        
        cursor.execute(create_checkpoints_summary_table)

        conn.commit()
        conn.close()

    #--------------------------------------------------------------------------
    def update_database(self):               
        dataset = pd.read_csv(self.source_path, sep=';', encoding='utf-8')        
        self.save_source_data(dataset)

    #--------------------------------------------------------------------------
    def load_source_data(self): 
        # Connect to the database and select entire source data table               
        conn = sqlite3.connect(self.db_path)        
        data = pd.read_sql_query(f"SELECT * FROM SOURCE_DATA", conn)
        conn.close()  

        return data

    #--------------------------------------------------------------------------
    def load_preprocessed_data(self): 
        # Connect to the database and select entire processed data table    
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
    def save_inference_statistics(self, data : pd.DataFrame): 
        # connect to sqlite database and save the preprocessed data as table
        conn = sqlite3.connect(self.db_path)         
        data.to_sql('GENERATED_REPORTS', conn, if_exists='replace')
        conn.commit()
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

 
    
    