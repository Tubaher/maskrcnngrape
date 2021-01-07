Para ejecutar este proyecto necesitas:
	1. Tener instalado python version: 3.6.9 (minima testeada por nosotros hasta ahora).
	2. Crear un ambiente python con requirements instalados.
	3. Archivo .h5 con los pesos del modelo...(idealmente el ultimo entrenado).
	4. Algun video para testear(dentro del contexto trabajado, claro -.-).


(Pasos a seguir ubuntu 18.04)

1: Instalar python3 en ubuntu (consola): 
	- sudo apt-get update
	- sudo apt-get install python3
	- sudo apt-get install python3-pip
	- sudo apt-get install python3-venv
	- (verificar versiones con): "python3 -V", "pip3 -V"

2: Para crear, ejecutar e instalar requirements en un ambiente de python3 (consola):
	- python3 -m venv env
	- source env/bin/activate
	- pip install -r requirements2}3.txt
	- (tomarse un cafesito mientras instala todo :B)
	*** (si hace falta alguna instalacion) ***
	- pip install "nani"
	- pip freeze > requirements2.txt (para actualizar requeriments en el git, no olvide hacer push)

3. Archivo .h5:
	- Solicite archivo .h5 del modelo a algun Spartano del team, en este git no se guardaran archivos tan pesados.

4. Video:	
	- Solicite a Jerges Ricardo un buen video para testear la deteccion.

Finalmente desde la raiz del git (consola):
	- cd samples/uvas
	- python splash_uvas.py --weights=../../mask_rcnn_uvas_0029.h5 --video=../../videos/DJI_0211.MOV
	- (reze y espera la magia)



Issues (Encontrados hasta ahora y con soluciones, sientase libre de añadir nuevos siempre y cuando tengan la solucion -.-):

	- (issues) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support. If you are on Ubuntu or Debian, 
	  install libgtk2.0-dev and pkg-config, then re-run cmake or configure script

	+ (solucion)(consola): (https://stackoverflow.com/questions/14655969/opencv-error-the-function-is-not-implemented)
		- sudo apt-get install libgtk2.0-dev pkg-config libqt4-dev opencv-contrib-python 
