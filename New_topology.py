import pandapower as pp
import pickle
import pandas as pd
import json

# 1. Cargar la red original desde el binario model.p
ruta_original = "marllib/patch/dpn/var_voltage_control/data/case33_3min_final/model.p"
with open(ruta_original, "rb") as f:
    net = pickle.load(f)

# 1. Convertimos el diccionario interno 'DF' en un DataFrame real de Pandas
df_line = pd.DataFrame(**net["line"]["DF"])
print("--- TOPOLOGÍA DE LÍNEAS ACTUALES ---")
print(df_line[["from_bus", "to_bus", "length_km", "r_ohm_per_km"]])

# =========================================================================
# OPCIÓN A: Cambiar la sección/impedancia de una línea (Ej: Refuerzo de red)
# =========================================================================
# Supongamos que queremos cambiar la resistencia de la línea con índice 5
# net.line.at[5, 'r_ohm_per_km'] = 0.15  # Reducimos la resistencia eléctrica
# net.line.at[5, 'x_ohm_per_km'] = 0.12

# =========================================================================
# OPCIÓN B: Añadir una nueva línea (Crear una red mallada o bucle)
# =========================================================================
# El case33 original es radial (forma de árbol). Si conectas, por ejemplo,
# la barra 18 con la barra 33, creas un anillo/malla, cambiando los voltajes.
# pp.create_line(net, from_bus=17, to_bus=32, length_km=1.0, std_type="149-AL1/24-ST1A 110.0")

# =========================================================================
# OPCIÓN C: Modificar switches
# =========================================================================
# 1- Modificar estado switch
# El case33 suele tener "tie-lines" (líneas de enlace abiertas). Si cambias
# el estado de un switch de Abierto (False) a Cerrado (True), modificas la topología.
# net.switch.at[0, 'closed'] = True  # Cierra un interruptor específico
# 2- Crear nuevo switch
# Crear un switch cerrado (closed=True) que conecta la barra 'bus' con la línea 'element'
# Convertir la tabla de switches a un DataFrame de Pandas
df_switch = pd.DataFrame(**net["switch"]["DF"])

# Crear el nuevo switch con los datos que querías
# Como la tabla original está vacía (index []), lo agregamos en la posición 0
nuevo_switch = {
    'bus': 0,           # Nodo al que se conecta
    'element': 9,       # Índice de la línea (o elemento) al que se conecta
    'et': 'l',          # 'l' significa que conecta a una línea (line)
    'type': None,
    'closed': True,     # Estado inicial: Cerrado
    'name': 'Mi_Switch_1',
    'z_ohm': 0.0        # Sin impedancia adicional
}
# Inyectamos la fila en el DataFrame
df_switch.loc[len(df_switch)] = nuevo_switch

# Eliminar todos los switches
df_switch = df_switch.iloc[0:0]
# Eliminar un switch en particular (0)
# df_switch = df_switch.drop(index=0)

# Volver a convertir al formato de diccionario anidado que usa MARLlib
net["switch"]["DF"] = json.loads(df_switch.to_json(orient='split'))
# =========================================================================

# =========================================================================
# OPCIÓN D: Dar de baja a una línea
# =========================================================================
# Desconectar por completo la línea con índice 5 (abre el circuito en esa rama)
# net.line.at[5, 'in_service'] = False



# 2. Guardar la NUEVA topología reemplazando el archivo que leerá MARLlib
# Consejo: Haz una copia de seguridad de tu model.p original antes de correr esto
ruta_destino = "marllib/patch/dpn/var_voltage_control/data/case33_3min_final/model.p"
with open(ruta_destino, "wb") as f:
    pickle.dump(net, f)

print("¡Nueva topología guardada con éxito en model.p!")