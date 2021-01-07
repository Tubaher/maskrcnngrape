### Ejecutar el codigo con un ambiente de conda ##Fecha: 07/01/2021

Para ejecutar este proyecto necesitas:
	1. Tener instalado python version: 3.6.9 (minima testeada por nosotros hasta ahora).
	2. Crear un ambiente python con requirements instalados.
	3. Archivo .h5 con los pesos del modelo...(idealmente el ultimo entrenado).
	4. Algun video para testear(dentro del contexto trabajado, claro -.-).

### Pasos detallado

1. Create the environment with conda and python 3.6.9

`conda create -n grapes python=3.6.9`

2. Instalar los requirimientos 3

`pip install -r requirements3.txt`

3. Descargar Archivo .h5  desde [drive](https://drive.google.com/drive/folders/1BVnFb5XKCctHdzKL2XMRAoYWUNlufd8o?usp=sharing):
	- Solicite archivo .h5 del modelo a algun Spartano del team, en este git no se guardaran archivos tan pesados.

4. Descargar Archivo checkpoint ckpt.t7 desde [drive](https://drive.google.com/drive/folders/1BVnFb5XKCctHdzKL2XMRAoYWUNlufd8o?usp=sharing) en la carpeta pretrained_weights
    - Copiar el archivo en la siguiente dirección `maskrcnngrape/samples/uvas/deep_sort_pytorch/deep_sort/deep/checkpoint/`

5. Descargar Video desde [drive](https://drive.google.com/drive/folders/1BVnFb5XKCctHdzKL2XMRAoYWUNlufd8o?usp=sharing):	
	- O solicite a Jerges Ricardo un buen video para testear la deteccion.

Finalmente desde la raiz del git (consola):
    
    cd samples/uvas
	python splash_uvas.py --weights=../../mask_rcnn_uvas_0029.h5 --video=../../videos/DJI_0211.MOV
    
(reze y espera la magia)
---

### Notas y Problemas

Encontrados hasta ahora y con soluciones, sientase libre de añadir nuevos siempre y cuando tengan la solucion:

1. (issues) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script

	- (solucion)(consola): (https://stackoverflow.com/questions/14655969/opencv-error-the-function-is-not-implemented)
    - `sudo apt-get install libgtk2.0-dev pkg-config`

2. (issues) Nota: tomar en cuenta que funcione este repositorio. Las librerias de cuda toolkit deben 10.0 y el cudnn > 7.5

3. Para entrenar las configuraciones del modelo se puede utilizar todas las GPUS disponibles cambiando el valor de
`IMAGES_PER_GPU`. Sin embargo para correr la inferencia se recomienda una sola GPU `IMAGES_PER_GPU=1` en  InferenceConfig del archivo [splash_uvas.py](samples/uvas/splash_uvas.py).
