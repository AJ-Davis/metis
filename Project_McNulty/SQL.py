# Code to load data onto AWS from command line
# scp -i ~/.ssh/aws_key.txt NSDUH-2015-DS0001-data-excel.tsv ubuntu@18.222.180.144:~/.
# Too many columns to CREATE TABLE manually

import pandas as pd

# import stata file through pandas
df = pd.read_stata('SAMHDA.dta')

from psycopg2 import connect
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

params = {
    'host': '18.222.180.144',
    'user': 'ubuntu',
    'port': 5432
}

# Connect and create database, disconnect, and reconnect to the right database
connection = connect(**params, dbname='ubuntu')
connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
connection.cursor().execute('CREATE DATABASE samhda;')
connection.close()

from sqlalchemy import create_engine
connection_string = f'postgres://ubuntu:{params["host"]}@{params["host"]}:{params["port"]}/samhda'
engine = create_engine(connection_string)
df.to_sql('survey', engine, index=False)

connection = connect(**params, dbname='samhda')
cursor = connection.cursor()
cursor.execute("SELECT * FROM survey;")
cursor.fetchall()

query = """
SELECT * FROM survey;
"""

cursor.execute(query)
cursor.fetchall()
