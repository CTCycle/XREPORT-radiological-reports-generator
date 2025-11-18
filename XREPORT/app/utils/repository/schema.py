from __future__ import annotations

from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


###############################################################################
class RadiographyData(Base):
    __tablename__ = "RADIOGRAPHY_DATA"
    image = Column(String, primary_key=True)
    text = Column(String)
    __table_args__ = (UniqueConstraint("image"),)


###############################################################################
class TrainingData(Base):
    __tablename__ = "TRAINING_DATASET"
    image = Column(String, primary_key=True)
    text = Column(String)
    tokens = Column(String)
    split = Column(String)
    __table_args__ = (UniqueConstraint("image"),)


###############################################################################
class GeneratedReport(Base):
    __tablename__ = "GENERATED_REPORTS"
    image = Column(String, primary_key=True)
    report = Column(String)
    checkpoint = Column(String, primary_key=True)
    __table_args__ = (UniqueConstraint("image", "checkpoint"),)


###############################################################################
class ImageStatistics(Base):
    __tablename__ = "IMAGE_STATISTICS"
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
    __table_args__ = (UniqueConstraint("name"),)


###############################################################################
class TextStatistics(Base):
    __tablename__ = "TEXT_STATISTICS"
    name = Column(String, primary_key=True)
    words_count = Column(Integer)
    __table_args__ = (UniqueConstraint("name"),)


###############################################################################
class CheckpointSummary(Base):
    __tablename__ = "CHECKPOINTS_SUMMARY"
    checkpoint = Column(String, primary_key=True)
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
    post_warmup_LR = Column(Float)
    warmup_steps = Column(Float)
    temperature = Column(Float)
    tokenizer = Column(String)
    max_report_size = Column(Integer)
    attention_heads = Column(Integer)
    n_encoders = Column(Integer)
    n_decoders = Column(Integer)
    embedding_dimensions = Column(Integer)
    frozen_img_encoder = Column(String)
    train_loss = Column(Float)
    val_loss = Column(Float)
    train_accuracy = Column(Float)
    val_accuracy = Column(Float)
    __table_args__ = (UniqueConstraint("checkpoint"),)
