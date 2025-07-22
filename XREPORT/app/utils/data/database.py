import os
import pandas as pd
import sqlalchemy
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.dialects.sqlite import insert

from XREPORT.app.constants import DATA_PATH, SOURCE_PATH
from XREPORT.app.logger import logger

Base = declarative_base()

###############################################################################
class RadiographyData(Base):
    __tablename__ = 'RADIOGRAPHY_DATA'
    image = Column(String, primary_key=True)
    text = Column(String)
    __table_args__ = (
        UniqueConstraint('image'),
    )
   
    
###############################################################################
class TrainData(Base):
    __tablename__ = 'TRAIN_DATA'
    image = Column(String, primary_key=True)
    tokens = Column(String)
    __table_args__ = (
        UniqueConstraint('image'),
    )


###############################################################################
class ValidationData(Base):
    __tablename__ = 'VALIDATION_DATA'
    image = Column(String, primary_key=True)
    tokens = Column(String)
    __table_args__ = (
        UniqueConstraint('image'),
    )    
    
    
###############################################################################
class GeneratedReport(Base):
    __tablename__ = 'GENERATED_REPORTS'
    image = Column(String, primary_key=True)
    report = Column(String)
    checkpoint = Column(String, primary_key=True)
    __table_args__ = (
        UniqueConstraint('image', 'checkpoint'),
    )


###############################################################################
class ImageStatistics(Base):
    __tablename__ = 'IMAGE_STATISTICS'
    name = Column(String, primary_key=True)
    height = Column(Integer)
    width = Column(Integer)
    mean = Column(Float)
    median = Column(Float)
    std = Column(Float)
    min = Column(Float)
    max = Column(Float)
    pixel_range = Column(Float)
    noise_std = Column(Float)
    noise_ratio = Column(Float)
    __table_args__ = (
        UniqueConstraint('name'),
    )


###############################################################################
class TextStatistics(Base):
    __tablename__ = 'TEXT_STATISTICS'
    name = Column(String, primary_key=True)
    words_count = Column(Integer)
    __table_args__ = (
        UniqueConstraint('name'),
    )
       
    
###############################################################################
class CheckpointSummary(Base):
    __tablename__ = 'CHECKPOINTS_SUMMARY'
    checkpoint_name = Column(String, primary_key=True)
    sample_size = Column(Float)
    validation_size = Column(Float)
    seed = Column(Integer)
    precision = Column(Integer)
    epochs = Column(Integer)
    batch_size = Column(Integer)
    split_seed = Column(Integer)
    image_augmentation = Column(String)
    image_height = Column(Integer)
    image_width = Column(Integer)
    image_channels = Column(Integer)
    jit_compile = Column(String) 
    has_tensorboard_logs = Column(String)
    initial_LR = Column(Float)
    constant_steps_LR = Column(Float)
    decay_steps_LR = Column(Float)
    target_LR = Column(Float)
    initial_neurons = Column(Float)
    dropout_rate = Column(Float)
    train_loss = Column(Float)
    val_loss = Column(Float)
    train_cosine_similarity = Column(Float)
    val_cosine_similarity = Column(Float)
    __table_args__ = (
        UniqueConstraint('checkpoint_name'),
    )
    

# [DATABASE]
###############################################################################
class XREPORTDatabase:

    def __init__(self):             
        self.db_path = os.path.join(DATA_PATH, 'XREPORT_database.db')
        self.source_path = os.path.join(SOURCE_PATH, 'XREPORT_dataset.csv')
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False, future=True)
        self.Session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = 2000
    
    #--------------------------------------------------------------------------       
    def initialize_database(self):
        Base.metadata.create_all(self.engine)  

    #--------------------------------------------------------------------------       
    def update_database_from_source(self): 
        source_dataset = pd.read_csv(self.source_path, sep=';', encoding='utf-8')                 
        self.save_source_data(source_dataset)

        return source_dataset         

    #--------------------------------------------------------------------------
    def upsert_dataframe(self, df: pd.DataFrame, table_cls):
        table = table_cls.__table__
        session = self.Session()
        try:
            unique_cols = []
            for uc in table.constraints:
                if isinstance(uc, UniqueConstraint):
                    unique_cols = uc.columns.keys()
                    break
            if not unique_cols:
                raise ValueError(f"No unique constraint found for {table_cls.__name__}")

            # Batch insertions for speed
            records = df.to_dict(orient='records')
            for i in range(0, len(records), self.insert_batch_size):
                batch = records[i:i + self.insert_batch_size]
                stmt = insert(table).values(batch)
                # Columns to update on conflict
                update_cols = {c: getattr(stmt.excluded, c) for c in batch[0] if c not in unique_cols}
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols,
                    set_=update_cols
                )
                session.execute(stmt)
                session.commit()
            session.commit()
        finally:
            session.close()       

    #--------------------------------------------------------------------------
    def load_source_dataset(self):
        with self.engine.connect() as conn:
            data = pd.read_sql_table('RADIOGRAPHY_DATA', conn)
        return data

    #--------------------------------------------------------------------------
    def load_train_and_validation_tables(self):       
        with self.engine.connect() as conn:
            train_data = pd.read_sql_table("TRAIN_DATA", conn)
            validation_data = pd.read_sql_table("VALIDATION_DATA", conn)

        return train_data, validation_data      

    #--------------------------------------------------------------------------
    def save_source_data(self, data : pd.DataFrame):
        with self.engine.begin() as conn:
            conn.execute(sqlalchemy.text(f"DELETE FROM RADIOGRAPHY_DATA"))        
        data.to_sql("RADIOGRAPHY_DATA", self.engine, if_exists='append', index=False) 
        
    #--------------------------------------------------------------------------
    def save_train_and_validation(self, train_data : pd.DataFrame, validation_data : pd.DataFrame):         
        with self.engine.begin() as conn:
            train_data.to_sql("TRAIN_DATA", conn, if_exists='replace', index=False)
            validation_data.to_sql("VALIDATION_DATA", conn, if_exists='replace', index=False)

    #--------------------------------------------------------------------------
    def save_predictions_table(self, data : pd.DataFrame):         
        with self.engine.begin() as conn:
            data.to_sql("GENERATED_REPORTS", conn, if_exists='replace', index=False)

    #--------------------------------------------------------------------------
    def save_image_statistics(self, data : pd.DataFrame):      
        with self.engine.begin() as conn:
            conn.execute(sqlalchemy.text(f"DELETE FROM IMAGE_STATISTICS"))        
        data.to_sql('IMAGE_STATISTICS', self.engine, if_exists='append', index=False)

    #--------------------------------------------------------------------------
    def save_text_statistics(self, data : pd.DataFrame):      
        with self.engine.begin() as conn:
            conn.execute(sqlalchemy.text(f"DELETE FROM TEXT_STATISTICS"))        
        data.to_sql('TEXT_STATISTICS', self.engine, if_exists='append', index=False)

    #--------------------------------------------------------------------------
    def save_checkpoints_summary(self, data : pd.DataFrame):         
        self.upsert_dataframe(data, CheckpointSummary)
    

 
    
    