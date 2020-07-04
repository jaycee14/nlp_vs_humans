"""Script to be run via docker-compose run to extract to stdout the values in the database

docker-compose run voila_server /opt/conda/bin/python extract_script.py > result.json
"""

from datetime import datetime


import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

host_addr = "database:5432"

user = os.environ['POSTGRES_USER']
passwd = os.environ['POSTGRES_PASSWORD']
db = os.environ['POSTGRES_DB']

url = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(user=user,
                                                            pw=passwd,
                                                            url=host_addr,
                                                            db=db, )

Base = declarative_base()


class Features(Base):
    __tablename__ = 'features'

    id = Column(Integer, primary_key=True)
    feature_text = Column(String)
    type = Column(String)


class Labels(Base):
    __tablename__ = 'labels'

    id = Column(Integer, primary_key=True)
    feature_id = Column(Integer)
    label_text = Column(String)
    entry_id = Column(Integer)


class Entries(Base):
    __tablename__ = 'entries'

    id = Column(Integer, primary_key=True)
    entry_date = Column(DateTime, default=datetime.utcnow)


if __name__ == '__main__':
    engine = create_engine(url)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    query = session.query(Labels.id, Labels.feature_id, Labels.label_text, Labels.entry_id, Features.feature_text,
                          Features.type).outerjoin(Features, Labels.feature_id == Features.id)

    df = pd.read_sql(query.statement, query.session.bind)

    json_out = df.to_json(orient='records')

    print(json_out)
