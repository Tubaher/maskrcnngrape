#!/usr/bin/env python
# coding: utf-8

import sys
import pg8000 as pg
#from IPython.core.display import display, HTML

#display(HTML("<style>.container { width:100% !important; }</style>"))
print(sys.executable)

hostname = '10.10.10.166'
username = 'digevo'
password = 'dgv2019'
database = 'digevo'

'''
table pg_test (
    id serial PRIMARY KEY,
    client_id int NOT NULL,
    campo_id int NOT NULL,
    cultivo_id int NOT NULL,
    hilera_id int NOT NULL,
    racimo_id int NOT NULL,
    racimo_area int NOT NULL,
    racimo_lng float8,
    racimo_lat float8,
    racimo_sepa VARCHAR(50),
    timestamp timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
);
'''

def select( conn ) :
    cur = conn.cursor()
    cur.execute("SELECT * FROM pg_test")
    fetch = cur.fetchall()
    for row in fetch:
        print(row)

def insert_one(conn, data) :
    cur = conn.cursor()
    cur.execute("INSERT INTO pg_test (client_id, campo_id, cultivo_id, hilera_id, racimo_id, racimo_area, racimo_lng, racimo_lat, racimo_sepa) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", data)
    conn.commit()

def insert_list(conn, datas) :
    cur = conn.cursor()
    for data in datas:
        cur.execute("INSERT INTO pg_test (client_id, campo_id, cultivo_id, hilera_id, racimo_id, racimo_area, racimo_lng, racimo_lat, racimo_sepa) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", data)
    conn.commit()

connection = pg.connect(user=username, host=hostname, database=database, password=password)

#data = ("1", "1", "1", "1", "3", "100", "10.20", "10.30", "Merlot")
#insert_one(connection, data)
insert_list(connection, [("1", "1", "1", "1", "4", "100", "10.20", "10.30", "Merlot"), ("1", "1", "1", "1", "2", "100", "10.20", "10.30", "Merlot"), ("1", "1", "1", "1", "3", "100", "10.20", "10.30", "Merlot")])
select(connection)

connection.close()
