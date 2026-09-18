import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import time
import ast
import matplotlib.ticker as ticker

actual_date = datetime.now().date()
algo = "PPO"
ventana = 10
algoritmo = "mappo"
dirección = {
    "mappo" : ['shared_policy','total_loss'],
    "ippo" : ['shared_policy','total_loss'],
    "vdppo" : ['shared_policy','total_loss'],
    "vda2c" : ['shared_policy','grad_gnorm'],
}

device = 'oficina'
mode = 'train'

if mode == 'train':
    time_data = '2026-09-17.12-20' # 07.08-58' # TRAIN
    # '2026-08-24.10-48' Q_BASE_MVAR=5, "q_weight": 0.1
    # '2026-08-25.16-55' Q_BASE_MVAR=2, "q_weight": 0.2
    # '2026-08-26.12-43' Q_BASE_MVAR=5, "q_weight": 0.1, dq_dv_weight=0.1
elif mode == 'eval':
    time_data = '2026-09-17.09-10' # EVAL
    # '2026-08-26.12-12' Q_BASE_MVAR=5, "q_weight": 0.1
    # '2026-08-26.12-16' Q_BASE_MVAR=2, "q_weight": 0.2
    # '2026-08-27.09-35' Q_BASE_MVAR=5, "q_weight": 0.1, dq_dv_weight=0.1
else:
    print('Definir bien mode')
topology = '33_3'

train_path = f'mappo_{topology}_{time_data}'

eval_path = f'mappo_{topology}_{time_data}'
cantidad_agentes = 4

if device == 'oficina':
    url = "C:\\PDN_runs\\Pruebas"
# else:
#     url = 'C:\\Users\\Nicolas\\Documents\\UNSL\\Programas\\MARLlib-PDN'

steps = np.arange(480)
time_index = pd.date_range(start="00:00", periods=480, freq="3min")
time_labels = time_index.strftime("%H:%M")
tick_positions = np.arange(0, 480, 40)  # cada 48 steps = cada 2.4 hs ~ cada 2hs

ventana = 50
def calcular_promedio_movil(datos, ventana, mode):
    """
    Calcula el promedio móvil.
    En los bordes donde no cabe la ventana, MANTIENE los datos originales.
    """
    datos = np.asarray(datos)
    n = len(datos)

    # Si hay menos datos que la ventana, devolvemos tal cual
    if n < ventana:
        return datos

    # 1. Crear el kernel de promedio
    ventana_kernel = np.ones(ventana) / ventana

    # 2. Calcular la parte central suavizada ('valid')
    # Esto reduce el tamaño del array
    datos_validos = np.convolve(datos, ventana_kernel, mode='valid')

    # 3. Calcular el offset (cuántos datos quedan a la izquierda)
    # Al centrar la ventana, sobran 'ventana // 2' elementos a cada lado aprox.
    offset = ventana // 2

    # 4. Construir el resultado final
    # a) Inicio: Tomamos los datos ORIGINALES desde el 0 hasta el offset
    parte_inicio = datos[:offset]

    # b) Final: Tomamos los datos ORIGINALES desde el final de la parte válida
    # Calculamos dónde termina la parte válida dentro del array original
    indice_final = offset + len(datos_validos)
    if mode == 'eval':
        parte_final = datos[indice_final:]
    else:
        parte_final = datos[indice_final: -ventana]

    # c) Concatenar: Originales + Promedio + Originales
    datos_suavizados = np.concatenate([parte_inicio, datos_validos, parte_final])

    return datos_suavizados

init = 10
final = 6200
cantidad_valores_mostrados = 500

if mode == 'train':
    with open(f'{url}'
              f'\\{train_path}'
              '\\result.json',
              'r') as file:
        train_data = []
        for episode in file:
            train_data.append(json.loads(episode))

    date_format = '%Y-%m-%d.%H-%M'
    folder = Path(r'C:\PDN_runs\Pruebas\results')

    # Convertir el string inicial a objeto datetime
    dt = datetime.strptime(time_data, date_format)

    # Ruta del archivo inicial
    file_path = folder / f'physical_log_{dt.strftime(date_format)}.csv'

    # Sumar 1 segundo iterativamente hasta encontrar el archivo
    while not file_path.exists():
        dt += timedelta(minutes=1)
        time_data = dt.strftime(date_format)
        file_path = folder / f'physical_log_{time_data}.csv'
        print(file_path)
        time.sleep(1)


    df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=False,)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["episode"])
    df["episode"] = df["episode"].astype(int)
    # df = pd.read_csv(f'C:\\PDN_runs\\Pruebas\\results\\physical_log_2026-09-01.08-48-13.csv', error_bad_lines=False, warn_bad_lines=False,)

    agents = train_data[0]['config']['model']['custom_model_config']['policy_mapping_info']['case33_3min']['team_prefix']
    agents = [f'agent_zone_{i+1}' for i in range(cantidad_agentes)]
    pol_agents = {f'agent_zone_{i+1}': f'pol_agent_zone_{i+1}' for i in range(cantidad_agentes)}
    figl, axl = plt.subplots(1,len(agents)+1, figsize=(12, 6))
    figr, axr = plt.subplots(1,len(agents)+1, figsize=(16, 6))
    figv, axv = plt.subplots(3, 1, figsize=(12, 10))
    figp, axp = plt.subplots(figsize=(12, 6))
    figp1, axp1 = plt.subplots(figsize=(12, 6))
    figp2, axp2 = plt.subplots(figsize=(12, 6))

    loss_episode = []
    loss_episode_ag0 = []
    loss_episode_ag1 = []
    loss_episode_ag2 = []
    loss_episode_ag3 = []
    loss_episode_ag4 = []


    reward_policy = []
    reward_episode = []
    reward_episode_ag0 = []
    reward_episode_ag1 = []
    reward_episode_ag2 = []
    reward_episode_ag3 = []
    reward_episode_ag4 = []

    vvio = []

    powerp_1 = []
    powerp_2 = []
    powerp_3 = []

    figp1, axp1 = plt.subplots()
    p_consumed_pv = []
    p_consumed_ev = []
    p_consumed_building_building = []
    p_consumed_building_pv = []
    p_consumed_building_storage = []

    for episode in range(1, len(train_data)): #len(data)): # TODO: Está hecho para 3 agentes
        loss_episode_ag0.append(train_data[episode]['info']['learner'][pol_agents[agents[0]]]['learner_stats']['total_loss'])
        loss_episode_ag1.append(train_data[episode]['info']['learner'][pol_agents[agents[1]]]['learner_stats']['total_loss'])
        loss_episode_ag2.append(train_data[episode]['info']['learner'][pol_agents[agents[2]]]['learner_stats']['total_loss'])
        loss_episode_ag3.append(train_data[episode]['info']['learner'][pol_agents[agents[3]]]['learner_stats']['total_loss'])
        reward_episode_ag0.append(train_data[episode]['policy_reward_mean'][pol_agents[agents[0]]])
        reward_episode_ag1.append(train_data[episode]['policy_reward_mean'][pol_agents[agents[1]]])
        reward_episode_ag2.append(train_data[episode]['policy_reward_mean'][pol_agents[agents[2]]])
        reward_episode_ag3.append(train_data[episode]['policy_reward_mean'][pol_agents[agents[3]]])
        reward_episode.append(train_data[episode]['episode_reward_mean'])

    loss_episode_ag = {agents[0]: loss_episode_ag0,
                    agents[1]: loss_episode_ag1,
                    agents[2]: loss_episode_ag2,
                    agents[3]: loss_episode_ag3,
                    # agents[4]: loss_episode_ag4,
                    }

    loss_episode = [a + b + c + d for a, b, c, d in zip(loss_episode_ag0, loss_episode_ag1, loss_episode_ag2, loss_episode_ag3)]

    for i in range(len(agents) + 1):

        if i == len(agents):
            # loss_episode = calcular_promedio_movil(loss_episode, ventana, mode)
            axl[i].set_title(f'Total loss')
            axl[i].plot(range(0, len(loss_episode)), np.asarray(loss_episode), label='total_rew')
            axl[i].legend(loc='best')
        else:
            # loss_episode_ag[agents[i]] = calcular_promedio_movil(loss_episode_ag[agents[i]], ventana, mode)
            axl[i].set_title(f'Loss - {agents[i]}')
            axl[i].plot(range(0, len(loss_episode_ag[agents[i]])), np.asarray(loss_episode_ag[agents[i]]), label=f'{agents[i]}')
            axl[i].legend(loc='best')
        axl[i].grid(True, alpha=0.3)

    reward_episode_ag = {agents[0]: reward_episode_ag0,
                         agents[1]: reward_episode_ag1,
                         agents[2]: reward_episode_ag2,
                         agents[3]: reward_episode_ag3,
                         # agents[4]: reward_episode_ag4,
                         }

    #figl.savefig(f'Loss_{actual_date}_{algo}.png', dpi=600)
    figl.show()
    for i in range(len(agents)+1):
        if i == len(agents):
            axr[i].plot(range(0, len(reward_episode)), np.asarray(reward_episode), label='total_rew')
        else:
            axr[i].set_title(f'{algoritmo} reward- {agents[i]}')
            axr[i].plot(range(0, len(reward_episode_ag[agents[i]])), np.asarray(reward_episode_ag[agents[i]]), label=f'rew_{agents[i]}')
        axr[i].grid(True, alpha=0.3)
    figr.show()

    # Crear un diccionario que aplique 'mean' a todas las columnas por defecto
    agg_dict = {col: "mean" for col in df.columns if col not in ["episode", "step", "percentage_of_v_out_of_control"]}
    # Sobrescribir las métricas donde te interesan los picos del episodio
    agg_dict["v_max"] = "max"  # Máximo absoluto alcanzado en el episodio
    agg_dict["v_min"] = "min"  # Mínimo absoluto alcanzado en el episodio
    agg_dict["percentage_of_v_out_of_control"] = "max"
    # Agrupar y filtrar
    ep = df.groupby("episode").agg(agg_dict)
    ep = ep[init:final]
    ventana = (final-init)//cantidad_valores_mostrados
    v_mean_bus = calcular_promedio_movil(ep["v_mean_bus"], ventana, mode)
    v_min = calcular_promedio_movil(ep["v_min"], ventana, mode)
    v_max = calcular_promedio_movil(ep["v_max"], ventana, mode)
    power_p_bus = calcular_promedio_movil(ep["power_p_bus"], ventana, mode)
    power_q_bus = calcular_promedio_movil(ep["power_q_bus"], ventana, mode)
    line_loading_perc = calcular_promedio_movil(ep["line_loading_perc"], ventana, mode)
    perd_p_line_total = calcular_promedio_movil(ep["perd_p_line_total"], ventana, mode)
    perd_q_line_total = calcular_promedio_movil(ep["perd_q_line_total"], ventana, mode)

    axv[0].set_title("Voltaje de línea")
    axv[1].set_title("Potencia de línea")
    axv[2].set_title("Pérdida de línea")
    axv2 = axv[2].twinx()
    # axv2 = axv.twinx()
    # axv[0].plot(range(0, len(ep["v_mean_bus"][5:])), np.asarray(ep["v_mean_bus"][5:]), label='v_mean_bus')
    # axv[1].plot(range(0, len(ep["q_gen_total"][5:])), np.asarray(ep["q_gen_total"][5:]), label='q_gen_total', color='green')
    # axv[2].plot(range(0, len(ep["line_loading_perc"][5:])), np.asarray(ep["line_loading_perc"][5:]), label='line_loading_perc', color='red')
    axv[0].plot(range(0, len(v_mean_bus)), v_mean_bus, label='v_mean_bus')
    axv[0].plot(range(0, len(v_min)), v_min, label='v_min')
    axv[0].plot(range(0, len(v_max)), v_max, label='v_max')
    axv[0].axhline(y=0.95, color='red', linestyle='--', linewidth=1)
    axv[0].axhline(y=1.05, color='red', linestyle='--', linewidth=1)
    axv[1].plot(range(0, len(power_p_bus)), power_p_bus, label='power_p_bus')
    axv[1].plot(range(0, len(power_q_bus)), power_q_bus, label='power_q_bus')
    # axv[1].plot(range(0, len(np.asarray(ep["va_degree_bus"]))), np.asarray(ep["va_degree_bus"]), label='va_degree_bus')
    axv2.plot(range(0, len(line_loading_perc)), line_loading_perc, label='porcentaje pérdidas línea', color='red')
    axv[2].plot(range(0, len(perd_p_line_total)), perd_p_line_total, label='pérdidas P')
    axv[2].plot(range(0, len(perd_q_line_total)), perd_q_line_total, label='pérdidas Q')
    axv[0].legend(loc='upper left')
    axv[1].legend(loc='best')
    axv[2].legend(loc='upper left')
    axv2.legend(loc='upper right')
    axv[0].grid(True, alpha=0.3)
    axv[0].set_ylim(0.65,1.25)
    # axv[1].set_ylim(-0.015, 0.01)
    # axv[2].set_ylim(0.0, 0.5)
    # axv2.set_ylim(0.0, 0.0001)
    figv.show()

    load_p = calcular_promedio_movil(ep["load_p"], ventana, mode)
    load_q = calcular_promedio_movil(ep["load_q"], ventana, mode)
    axp1.set_title('Consumos')
    axp1.plot(range(0, len(load_p)), load_p, label='P consumida total [MW]')
    axp1.plot(range(0, len(load_q)), load_q, label='Q consumida total [MVAR]')
    axp1.legend(loc='best')
    figp1.show()

    p_gen_total = calcular_promedio_movil(ep["p_gen_total"], ventana, mode)
    q_gen_total = calcular_promedio_movil(ep["q_gen_total"], ventana, mode)
    axp2.set_title('Generación fotovoltáica')
    axp2.plot(range(0, len(p_gen_total)), p_gen_total, label='P generada total [MW]')
    axp2.plot(range(0, len(q_gen_total)), q_gen_total, label='Q generada total [MVAR]')
    axp2.legend(loc='best')
    axp2.grid(True, alpha=0.3)
    figp2.show()

    percentage_of_v_out_of_control = calcular_promedio_movil(ep["percentage_of_v_out_of_control"], ventana, mode)*100
    axp.set_title('percentage_of_v_out_of_control')
    axp.plot(range(0, len(percentage_of_v_out_of_control)), percentage_of_v_out_of_control, label='v_out_con')
    axp.legend(loc='best')
    axp.grid(True, alpha=0.3)
    figp.show()

# ------ evaluación ------

elif mode == 'eval':
    with open(f'{url}'
              f'\\{eval_path}'
              '\\result.json',
              'r') as file:
        eval_data = []
        for episode in file:
            eval_data.append(json.loads(episode))
    df = pd.read_csv(f'C:\\PDN_runs\\Pruebas\\results\\physical_log_{time_data}.csv') #physical_log_2026-07-27_11-36-17
    eval_df = pd.read_csv(f'C:\\PDN_runs\\Pruebas\\results\\eval_average_voltage_{time_data}.csv')
    # agents = eval_data[0]["rewards"].keys()
    # agents = [f'agent_zone_{i + 1}' for i in range(cantidad_agentes)]
    agents = [f'agent_zone_{i + 1}' for i in range(cantidad_agentes)]
    figli, axli = plt.subplots(3, 1, figsize=(12, 6))
    figlo, axlo = plt.subplots(figsize=(12, 6))
    figgf, axgf = plt.subplots(figsize=(12, 6))
    figex, axex = plt.subplots(figsize=(12, 6))

    # Por step dentro de un episodio específico
    ep = df[df["episode"] == 1]  # 1 para summer, 0  para winter
    ep = ep.iloc[0:480]

    eval_ep = eval_df

    axli[0].set_title("Voltaje de línea")
    axli[1].set_title("Potencia de línea")
    axli[2].set_title("Pérdida de línea")
    values = np.asarray(ep["v_mean_bus"].values)
    largo = len(np.asarray(ep["v_mean_bus"].values))
    axli[0].plot(range(0, len(np.asarray(ep["v_mean_bus"]))), np.asarray(ep["v_mean_bus"].values), label='v_mean_bus')
    axli[0].plot(range(0, len(np.asarray(ep["v_min"]))), np.asarray(ep["v_min"].values), label='v_min')
    axli[0].plot(range(0, len(np.asarray(ep["v_max"]))), np.asarray(ep["v_max"].values), label='v_max')
    axli[0].axhline(y=0.95, color='red', linestyle='--', linewidth=1)
    axli[0].axhline(y=1.05, color='red', linestyle='--', linewidth=1)
    axli[1].plot(range(0, len(np.asarray(ep["power_p_bus"]))), np.asarray(ep["power_p_bus"]), label='power_p_bus')
    axli[1].plot(range(0, len(np.asarray(ep["power_q_bus"]))), np.asarray(ep["power_q_bus"]), label='power_q_bus')
    # axli[1].plot(range(0, len(np.asarray(ep["va_degree_bus"]))), np.asarray(ep["va_degree_bus"]), label='va_degree_bus')
    axli[2].plot(range(0, len(np.asarray(ep["line_loading_perc"]))), np.asarray(ep["line_loading_perc"]), label='porcentaje pérdidas línea')
    axli[2].plot(range(0, len(np.asarray(ep["perd_p_line_total"]))), np.asarray(ep["perd_p_line_total"]), label='pérdidas P')
    axli[2].plot(range(0, len(np.asarray(ep["perd_q_line_total"]))), np.asarray(ep["perd_q_line_total"]), label='pérdidas Q')
    axli[0].set_xticks([])
    axli[1].set_xticks([])
    axli[2].set_xticks(tick_positions)
    axli[2].set_xticklabels(time_labels[tick_positions], rotation=45)
    axli[0].legend(loc='best')
    axli[1].legend(loc='best')
    axli[2].legend(loc='best')
    figli.show()

    axlo.set_title('Consumos')
    axlo.plot(range(0, len(np.asarray(ep["load_p"]))), np.asarray(ep["load_p"]), label='P consumida total [MW]')
    axlo.plot(range(0, len(np.asarray(ep["load_q"]))), np.asarray(ep["load_q"]), label='Q consumida total [MVAR]')
    axlo.set_xticks(tick_positions)
    axlo.set_xticklabels(time_labels[tick_positions], rotation=45)
    axlo.legend(loc='best')
    figlo.show()

    axgf.set_title('Generación fotovoltáica')
    axgf.plot(range(0, len(np.asarray(ep["p_gen_total"]))), np.asarray(ep["p_gen_total"]), label='P generada total [MW]')
    axgf.plot(range(0, len(np.asarray(ep["q_gen_total"]))), np.asarray(ep["q_gen_total"]), label='Q generada total [MVAR]')
    axgf.set_xticks(tick_positions)
    axgf.set_xticklabels(time_labels[tick_positions], rotation=45)
    axgf.legend(loc='best')
    figgf.show()

    axex.set_title('extras')
    axex.plot(range(0, len(np.asarray(ep["percentage_of_v_out_of_control"]))), np.asarray(ep["percentage_of_v_out_of_control"])*100, label='v_out_con')
    # axex.plot(range(0, len(np.asarray(ep["frec"]))), np.asarray(ep["frec"]), label='frec')
    axex.set_xticks(tick_positions)
    axex.set_xticklabels(time_labels[tick_positions], rotation=45)
    axex.legend(loc='best')
    figex.show()

    # Datos de las líneas
    # "v_mean_bus"
    # "v_min"
    # "v_max"
    # "va_degree_bus"
    # "power_p_bus"
    # "power_q_bus"
    # # Datos de las pérdidas en la línea
    # "perd_p_line_total"
    # "perd_q_line_total"
    # "line_loading_perc"
    # # Datos de la generación fotovoltáica
    # "p_gen_total"
    # "q_gen_total"
    # # Custom metrics
    # "average_voltage"
    # "total_line_loss"
    # "q_loss"
    # # Datos de la carga
    # "load_p"
    # "load_q"
    # Extras
    # "percentage_of_v_out_of_control"
    # "frec"

    # 1. Definir los buses (ejemplo: IEEE 33 buses)
    def extract_bus_array(data_cell):
        # Si la celda se leyó desde CSV como string, se convierte a diccionario
        if isinstance(data_cell, str):
            data_cell = ast.literal_eval(data_cell)
        # Retorna los 33 valores ordenados como numpy array
        return np.array(list(data_cell.values()))


    # Episodio Verano
    avg_v_verano = extract_bus_array(eval_ep["average_voltage_by_node"].iloc[0])
    min_v_verano = extract_bus_array(eval_ep["min_voltage_by_node"].iloc[0])
    max_v_verano = extract_bus_array(eval_ep["max_voltage_by_node"].iloc[0])
    # Episodio Invierno
    # avg_v_invierno = extract_bus_array(eval_ep["average_voltage_by_node"].iloc[1])
    # min_v_invierno = extract_bus_array(eval_ep["min_voltage_by_node"].iloc[1])
    # max_v_invierno = extract_bus_array(eval_ep["max_voltage_by_node"].iloc[1])

    num_buses = len(avg_v_verano)
    buses = [str(i) for i in range(1, num_buses + 1)]
    angles = np.linspace(0, 2 * np.pi, num_buses, endpoint=False).tolist()

    # Cerrar los polígonos uniendo el último nodo con el primero
    angles += angles[:1]
    avg_v_verano = np.append(avg_v_verano, avg_v_verano[0])
    min_v_verano = np.append(min_v_verano, min_v_verano[0])
    max_v_verano = np.append(max_v_verano, max_v_verano[0])
    # avg_v_invierno = np.append(avg_v_invierno, avg_v_invierno[0])

    # Generar la gráfica polar comparativa
    fig1, ax1 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)

    # Configurar límites radiales
    r_min = min(min_v_verano.min(), min_v_verano.min()) - 0.02
    r_max = max(max_v_verano.max(), max_v_verano.max()) + 0.02
    ax1.set_rlim(r_min, r_max)

    # --- PINTAR BANDAS DE VOLTAJE ---
    theta_contour = np.linspace(0, 2 * np.pi, 500)

    # Zona normal (0.95 a 1.05 pu) en verde
    ax1.fill_between(
        theta_contour, 0.95, 1.05, color="green", alpha=0.15, zorder=0
    )

    # Zona fuera de rango (< 0.95 y > 1.05 pu) en rojo
    ax1.fill_between(theta_contour, r_min, 0.95, color="red", alpha=0.10, zorder=0)
    ax1.fill_between(theta_contour, 1.05, r_max, color="red", alpha=0.10, zorder=0)

    # Trazar las series de cada estación
    ax1.plot(
        angles,
        avg_v_verano,
        label="AVG - Verano",
        color="blue",
        linewidth=1.8,
        zorder=2,
    )
    ax1.plot(
        angles,
        min_v_verano,
        label="MIN - Verano",
        color="green",
        linewidth=1.8,
        zorder=2,
    )
    ax1.plot(
        angles,
        max_v_verano,
        label="MAX - Verano",
        color="red",
        linewidth=1.8,
        zorder=2,
    )

    # Configurar etiquetas del eje y formato
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(buses, fontsize=8)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.legend(loc="lower right", bbox_to_anchor=(1.25, 0.1))
    plt.title("Perfil Promedio de Tensión por Nodo (Verano vs Invierno)", pad=20)
    plt.tight_layout()
    plt.show()