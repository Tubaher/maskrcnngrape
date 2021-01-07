Este documento esta destinado a guiar en la configuracion de la DB pensada para el proyecto Austral falcon uvas en un servidor Ubuntu 18.

1. Instalar y configurar PostgreSQL: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-18-04
	Idealmente mantener la convencion del mismo nombre de ususario, db y tabla que puede encontrar en el pg_test.py
2. Crear tabla como puede encontrar el en archivo .sql (se puede abrir con editor de txt, para echarle un ojo a lo que ejecutara)
	Ej: si mantiene los nombre: psql -d digevo -f ~/Escritorio/python/uva/pg_test.sql
3. Crear un env de python o agregar al ya existente las siguientes intalaciones:
	- pg8000
	*** actualizacion, añadi al requirements del git lo que correspondia para ejecutar, no olvidar ejecutar:
	- pip install -r requirements.txt (para actualizar su ambiente)

4. Ejecute pgTest.py una vez corrido el ambiente, se insertaran 3 valores a la tabla indicada y a la DIRECCION IP INDICADA.
