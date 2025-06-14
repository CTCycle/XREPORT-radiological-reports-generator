import os
import sqlite3
import pandas as pd

from XREPORT.commons.constants import DATA_PATH, SOURCE_PATH, INFERENCE_PATH
from XREPORT.commons.logger import logger


###############################################################################
class RadiographyDataTable:

    def __init__(self):
        self.name = 'RADIOGRAPHY_DATA'
        self.dtypes = {
            'image': 'VARCHAR',
            'text': 'VARCHAR'}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (
            image VARCHAR,
            text VARCHAR            
        );
        '''

        cursor.execute(query)
   
    
###############################################################################
class TrainDataTable:

    def __init__(self):
        self.name = 'TRAIN_DATA'
        self.dtypes = {
            'image': 'VARCHAR',
            'tokens': 'VARCHAR'}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (
            image VARCHAR,
            tokens VARCHAR            
        );
        '''

        cursor.execute(query)  


###############################################################################
class ValidationDataTable:

    def __init__(self):
        self.name = 'VALIDATION_DATA'
        self.dtypes = {
            'image': 'VARCHAR',
            'tokens': 'VARCHAR'}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (
            image VARCHAR,
            tokens VARCHAR            
        );
        '''

        cursor.execute(query)     
    
    
###############################################################################
class GeneratedReportsTable:

    def __init__(self):
        self.name = 'GENERATED_REPORTS'
        self.dtypes = {
            'image': 'VARCHAR',
            'report': 'VARCHAR',
            'checkpoint': 'VARCHAR'}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (
            image VARCHAR,
            report VARCHAR 
            checkpoint VARCHAR            
        );
        '''

        cursor.execute(query)        
    

###############################################################################
class ImageStatisticsTable:

    def __init__(self):
        self.name = 'IMAGE_STATISTICS'
        self.dtypes = {
            'name': 'VARCHAR',
            'height': 'INTEGER',
            'width': 'INTEGER',
            'mean': 'FLOAT',
            'median': 'FLOAT',
            'std': 'FLOAT',
            'min': 'FLOAT',
            'max': 'FLOAT',
            'pixel_range': 'FLOAT',
            'noise_std': 'FLOAT',
            'noise_ratio': 'FLOAT'}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (            
            name VARCHAR,
            height INTEGER,
            width INTEGER,
            mean FLOAT,
            median FLOAT,
            std FLOAT,
            min FLOAT,
            max FLOAT,
            pixel_range FLOAT,
            noise_std FLOAT,
            noise_ratio FLOAT
        );
        '''

        cursor.execute(query)

###############################################################################
class TextStatisticsTable:

    def __init__(self):
        self.name = 'TEXT_STATISTICS'
        self.dtypes = {
            'name': 'VARCHAR',
            'words_count': 'INTEGER',}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (            
            name VARCHAR,
            words_count INTEGER
        );
        '''

        cursor.execute(query)
       
    
###############################################################################
class CheckpointSummaryTable:

    def __init__(self):
        self.name = 'CHECKPOINTS_SUMMARY'
        self.dtypes = {
            'checkpoint_name': 'VARCHAR',
            'sample_size': 'FLOAT',
            'validation_size': 'FLOAT',
            'seed': 'INTEGER',
            'precision_bits': 'INTEGER',
            'epochs': 'INTEGER',
            'additional_epochs': 'INTEGER',
            'batch_size': 'INTEGER',
            'split_seed': 'INTEGER',
            'image_augmentation': 'VARCHAR',
            'image_height': 'INTEGER',
            'image_width': 'INTEGER',
            'image_channels': 'INTEGER',
            'jit_compile': 'VARCHAR',
            'jit_backend': 'VARCHAR',
            'device': 'VARCHAR',
            'device_id': 'VARCHAR',
            'number_of_processors': 'INTEGER',
            'use_tensorboard': 'VARCHAR',
            'lr_scheduler_initial_lr': 'FLOAT',
            'lr_scheduler_constant_steps': 'FLOAT',
            'lr_scheduler_decay_steps': 'FLOAT'}    

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (            
            checkpoint_name VARCHAR,
            sample_size FLOAT,
            validation_size FLOAT,
            seed INTEGER,
            precision_bits INTEGER,
            epochs INTEGER,
            additional_epochs INTEGER,
            batch_size INTEGER,
            split_seed INTEGER,
            image_augmentation VARCHAR,
            image_height INTEGER,
            image_width INTEGER,
            image_channels INTEGER,
            jit_compile VARCHAR,
            jit_backend VARCHAR,
            device VARCHAR,
            device_id VARCHAR,
            number_of_processors INTEGER,
            use_tensorboard VARCHAR,
            lr_scheduler_initial_lr FLOAT,
            lr_scheduler_constant_steps FLOAT,
            lr_scheduler_decay_steps FLOAT
            );
            ''' 
         
        cursor.execute(query)       
    

# [DATABASE]
###############################################################################
class XREPORTDatabase:

    def __init__(self, configuration):             
        self.db_path = os.path.join(DATA_PATH, 'XREPORT_database.db') 
        self.source_path = os.path.join(SOURCE_PATH, 'XREPORT_dataset.csv')   
        self.configuration = configuration 
        self.source_data = RadiographyDataTable()
        self.train_data = TrainDataTable()
        self.validation_data = ValidationDataTable()
        self.inference_data = GeneratedReportsTable()
        self.image_stats = ImageStatisticsTable()
        self.text_stats = TextStatisticsTable()
        self.checkpoints_summary = CheckpointSummaryTable()    
        self.initialize_database()
        self.update_database()

    #--------------------------------------------------------------------------       
    def initialize_database(self):        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        self.source_data.create_table(cursor)  
        self.train_data.create_table(cursor)
        self.validation_data.create_table(cursor)
        self.inference_data.create_table(cursor)
        self.image_stats.create_table(cursor)  
        self.checkpoints_summary.create_table(cursor)   
        conn.commit()
        conn.close()

    #--------------------------------------------------------------------------
    def update_database(self): 
        logger.debug(f'Updating database from {self.source_path}')              
        source_dataset = pd.read_csv(self.source_path, sep=';', encoding='utf-8')                 
        self.save_source_data_table(source_dataset)        

    #--------------------------------------------------------------------------
    def load_source_data_table(self):                  
        conn = sqlite3.connect(self.db_path)        
        data = pd.read_sql_query(
            f"SELECT * FROM {self.source_data.name}", conn)
        conn.close()  

        return data

    #--------------------------------------------------------------------------
    def load_train_and_validation_tables(self):       
        conn = sqlite3.connect(self.db_path)        
        train_data = pd.read_sql_query(
            f"SELECT * FROM {self.train_data.name}", conn)
        validation_data = pd.read_sql_query(
            f"SELECT * FROM {self.validation_data.name}", conn)
        conn.close()  

        return train_data, validation_data         

    #--------------------------------------------------------------------------
    def save_source_data_table(self, data : pd.DataFrame):        
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(self.source_data.name, conn, if_exists='replace', index=False,
                    dtype=self.source_data.get_dtypes())
        conn.commit()
        conn.close() 
        
    #--------------------------------------------------------------------------
    def save_train_and_validation_tables(self, train_data : pd.DataFrame, validation_data : pd.DataFrame):         
        conn = sqlite3.connect(self.db_path)         
        train_data.to_sql(
            self.train_data.name, conn, if_exists='replace', index=False,
            dtype=self.train_data.get_dtypes())  
        validation_data.to_sql(
            self.validation_data.name, conn, if_exists='replace', index=False,
            dtype=self.validation_data.get_dtypes())    
        conn.commit()
        conn.close()

    #--------------------------------------------------------------------------
    def save_inference_data_table(self, data : pd.DataFrame):         
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(self.inference_data.name, conn, if_exists='replace', index=False,
                    dtype=self.inference_data.get_dtypes())
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_image_statistics_table(self, data : pd.DataFrame):      
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(
            self.image_stats.name, conn, if_exists='replace', index=False,
            dtype=self.image_stats.get_dtypes())
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_text_statistics_table(self, data : pd.DataFrame):      
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(
            self.text_stats.name, conn, if_exists='replace', index=False,
            dtype=self.text_stats.get_dtypes())
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_checkpoints_summary_table(self, data : pd.DataFrame):         
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(
            self.checkpoints_summary.name, conn, if_exists='replace', index=False,
            dtype=self.checkpoints_summary.get_dtypes())
        conn.commit()
        conn.close() 

 
    
    