# Cuadrados Mínimos Lineales: TP3 de Métodos Numéricos

## Instrucciones

En `data/` están los datasets. En `tools/` están las instrucciones para descargar los archivos (provistas por
la cátedra)

En `src/` está el código de C++, en particular en `src/airline.cpp` está el entrypoint de pybind. El código
se compila con `compile.sh` y genera el `.so` en `notebooks/`.

En `notebooks/` está el código de test, visualización y experimentación.

Necesitamos bajar las librerías `pybind` y `eigen` (el "numpy" de C++), para eso bajamos los submódulos
como primer paso.

Versión de Python >= 3.6.5, librerías requeridas en `requirements.txt` (se recomienda
correr en una virtualenv).
