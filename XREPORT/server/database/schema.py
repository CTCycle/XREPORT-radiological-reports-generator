from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


###############################################################################
class Table(Base):
    __tablename__ = "TABLE"
    id = Column(BigInteger, primary_key=True)
    param = Column(String(200))   
    __table_args__ = (UniqueConstraint("id"),)


