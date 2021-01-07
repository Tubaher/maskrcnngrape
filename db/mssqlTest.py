from os import getenv
import pymssql
import datetime

'''
    CREATE DATABASE UVAS
    USE uvas
    CREATE TABLE deteccion(
        client_id integer NOT NULL,
        campo_id integer NOT NULL,
        cultivo_id integer NOT NULL,
        hilera_id integer NOT NULL,
        racimo_id integer NOT NULL,
        racimo_area integer NOT NULL,
        racimo_lng character varying(10),
        racimo_lat character varying(10),
        racimo_sepa character varying(50),
        timestamp datetime
    );
'''

def read(conn) :
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM deteccion')
    row = cursor.fetchone()
    while row:
        print("hilera=%s, racimo=%s, area=%s, lng=%s, lat=%s, lng=%s" % (row[4],row[5],row[6],row[7],row[8],row[9]))
        row = cursor.fetchone()

def insert_list(conn, lista):
    print(lista)
    cursor = conn.cursor()
    cursor.executemany("INSERT INTO deteccion VALUES (%d, %d, %d, %d, %d, %d, %s, %s, %s, %s)", lista)
    conn.commit()




# MAIN
server = '54.233.255.233'
user = 'SA'
password = 'digevo'

conn = pymssql.connect(server, user, password, "AustralFalconUvas")

lista = [(1, 1, 1, 1, 1, 450, "10.5", "11.6", "carmenere", datetime.datetime.now()),
     (1, 1, 1, 1, 2, 250, "10.5", "11.6", "carmenere", datetime.datetime.now()),
     (1, 1, 1, 1, 3, 350, "10.5", "11.6", "carmenere", datetime.datetime.now())]

insert_list(conn, lista)
read(conn)
conn.close()
