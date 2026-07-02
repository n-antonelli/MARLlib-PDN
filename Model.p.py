import pickle
from pprint import pprint

ruta = "marllib/patch/dpn/var_voltage_control/data/case33_3min_final/model.p"

# Abrir en modo de lectura binaria ('rb')
with open(ruta, "rb") as f:
    datos_red = pickle.load(f)

# Esto te imprimirá en la consola las claves y variables reales del caso 33
pprint(datos_red)