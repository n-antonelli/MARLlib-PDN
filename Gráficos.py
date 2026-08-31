import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# def reward_function(v):
#     if np.abs(v) < 1:
#         return np.exp(-1 / (1 - v**4))
#     elif 1 < v < 3:
#         return np.exp(-1 / (1 - (v - 2)**4))
#     else:
#         return 0.0
#
# # Vectorizar la función para usarla con arrays de numpy
# v_func = np.vectorize(reward_function)
#
# # Generar datos para el gráfico
# v_vals = np.linspace(-1.5, 3.5, 400)
# reward_vals = v_func(v_vals)
#
# plt.figure(figsize=(8, 4))
# plt.plot(v_vals, reward_vals, label="Reward Shaping", color='blue', lw=2)
# plt.axhline(0, color='black', lw=0.5, ls='--')
# plt.title("Visualización de la Función de Recompensa (Doble Bump)")
# plt.xlabel("Estado (v)")
# plt.ylabel("Recompensa")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.show()


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
    time_data = '2026-08-28.11-46' # TRAIN
    # '2026-08-24.10-48' Q_BASE_MVAR=5, "q_weight": 0.1
    # '2026-08-25.16-55' Q_BASE_MVAR=2, "q_weight": 0.2
    # '2026-08-26.12-43' Q_BASE_MVAR=5, "q_weight": 0.1, dq_dv_weight=0.1
else:
    time_data = '2026-08-28.09-03' # EVAL
    # '2026-08-26.12-12' Q_BASE_MVAR=5, "q_weight": 0.1
    # '2026-08-26.12-16' Q_BASE_MVAR=2, "q_weight": 0.2
    # '2026-08-27.09-35' Q_BASE_MVAR=5, "q_weight": 0.1, dq_dv_weight=0.1
topology = '33_3'

train_path = f'mappo_{topology}_{time_data}'
# line_losses
# 'MAPPOTrainer_voltage_case33_3min_final_f66a3_00000_0_2026-05-18_08-23-15' physical_log20260518_082343
# q_losses
# 'MAPPOTrainer_voltage_case33_3min_final_aa900_00000_0_2026-05-19_09-02-53' physical_log20260519_090314
# rewards escalados
# line_losses
# 'MAPPOTrainer_voltage_case33_3min_final_8b089_00000_0_2026-05-21_08-38-10'
# 'IPPOTrainer_voltage_case33_3min_final_f1633_00000_0_2026-05-26_09-03-44'
# q_losses
# 'MAPPOTrainer_voltage_case33_3min_final_301f6_00000_0_2026-05-20_11-57-14'
# 'IPPOTrainer_voltage_case33_3min_final_c6216_00000_0_2026-05-28_08-38-40'
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
final = 4000
cantidad_valores_mostrados = 500

if mode == 'train':
    with open(f'{url}'
              f'\\{train_path}'
              '\\result.json',
              'r') as file:
        train_data = []
        for episode in file:
            train_data.append(json.loads(episode))
    df = pd.read_csv(f'C:\\PDN_runs\\Pruebas\\results\\physical_log_{time_data}-52.csv', error_bad_lines=False, warn_bad_lines=False,)
    # df = pd.read_csv(f'C:\\PDN_runs\\Pruebas\\results\\physical_log_2026-08-26.12-44.csv', error_bad_lines=False, warn_bad_lines=False,)

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

    # pol_pv
    # pol_building
    # ['learner']['pol_ev']['learner_stats']['total_loss']
    # print((data[1]['info']['learner']['pol_ev']['learner_stats']['total_loss']))
    # print(data[episode]['info']['learner'])
    #Sacar tipos de agentes (nombres)
    for episode in range(1, len(train_data)): #len(data)): # TODO: Está hecho para 3 agentes
        loss_episode_ag0.append(train_data[episode]['info']['learner'][pol_agents[agents[0]]]['learner_stats']['total_loss'])
        loss_episode_ag1.append(train_data[episode]['info']['learner'][pol_agents[agents[1]]]['learner_stats']['total_loss'])
        loss_episode_ag2.append(train_data[episode]['info']['learner'][pol_agents[agents[2]]]['learner_stats']['total_loss'])
        loss_episode_ag3.append(train_data[episode]['info']['learner'][pol_agents[agents[3]]]['learner_stats']['total_loss'])
        # loss_episode_ag4.append(train_data[episode]['info']['learner'][pol_agents[agents[4]]]['learner_stats']['total_loss'])
        #loss_episode.append(sum(loss_episode_ag2[episode],loss_episode_ag1[episode], loss_episode_ag0[episode]))

        # if 'shared_policy' in train_data[episode]['policy_reward_max']:
        #     reward_episode.append(train_data[episode]['policy_reward_max']['shared_policy'])
        # else:
        #     pass

        reward_episode_ag0.append(train_data[episode]['policy_reward_mean'][pol_agents[agents[0]]])
        reward_episode_ag1.append(train_data[episode]['policy_reward_mean'][pol_agents[agents[1]]])
        reward_episode_ag2.append(train_data[episode]['policy_reward_mean'][pol_agents[agents[2]]])
        reward_episode_ag3.append(train_data[episode]['policy_reward_mean'][pol_agents[agents[3]]])
        # reward_episode_ag4.append(train_data[episode]['policy_reward_mean'][pol_agents[agents[4]]])
        reward_episode.append(train_data[episode]['episode_reward_mean'])
        # reward_episode_ag1.append(train_data[episode]['policy_reward_mean'][pol_agents[agents[1]]])
        # reward_episode_ag2.append(train_data[episode]['policy_reward_mean'][pol_agents[agents[2]]])

        # vvio.append(train_data[episode]["custom_metrics"]["vvio"]) # TODO: No andan para IPPO
        # powerp_1.append(train_data[episode]["custom_metrics"][f"powerp_mean_{agents[0]}_mean"])
        # powerp_2.append(train_data[episode]["custom_metrics"][f"powerp_mean_{agents[1]}_mean"])
        # powerp_3.append(train_data[episode]["custom_metrics"][f"powerp_mean_{agents[2]}_mean"])

        # # print(train_data[episode]["custom_metrics"])
        # p_consumed_pv.append(train_data[episode]["custom_metrics"]["p_consumed_pv_mean"])
        # p_consumed_ev.append(train_data[episode]["custom_metrics"]["p_consumed_ev-charging_mean"])
        # p_consumed_building_building.append(train_data[episode]["custom_metrics"]["p_consumed_building_building_mean"])
        # p_consumed_building_pv.append(train_data[episode]["custom_metrics"]["p_consumed_building_pv_mean"])
        # p_consumed_building_storage.append(train_data[episode]["custom_metrics"]["p_consumed_building_storage_mean"])

    loss_episode_ag = {agents[0]: loss_episode_ag0,
                    agents[1]: loss_episode_ag1,
                    agents[2]: loss_episode_ag2,
                    agents[3]: loss_episode_ag3,
                    # agents[4]: loss_episode_ag4,
                    }

    loss_episode = [a + b + c + d for a, b, c, d in zip(loss_episode_ag0, loss_episode_ag1, loss_episode_ag2, loss_episode_ag3)]
    # axl.plot(range(0, len(loss_episode)), np.asarray(loss_episode), label=f'loss')
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


    reward_episode_ag = {agents[0]: reward_episode_ag0,
                         agents[1]: reward_episode_ag1,
                         agents[2]: reward_episode_ag2,
                         agents[3]: reward_episode_ag3,
                         # agents[4]: reward_episode_ag4,
                         }
    # axl.legend(loc='best')
    # axl.set_xlim(50, )
    # axl.set_ylim(0, 100000)
    #figl.savefig(f'Loss_{actual_date}_{algo}.png', dpi=600)
    figl.show()
    # axr.plot(range(0, len(reward_episode)), np.asarray(reward_episode), label=f'rew')
    for i in range(len(agents)+1):
        if i == len(agents):
            axr[i].plot(range(0, len(reward_episode)), np.asarray(reward_episode), label='total_rew')
        else:
            axr[i].set_title(f'{algoritmo} reward- {agents[i]}')
            axr[i].plot(range(0, len(reward_episode_ag[agents[i]])), np.asarray(reward_episode_ag[agents[i]]), label=f'rew_{agents[i]}')

    # axr.legend(loc='best')
    # axr.set_xlim(50, )
    # axr.set_ylim(-1000, 0)
    #figr.savefig(f'curvas/Reward{actual_date}_{algo}.png', dpi=600)
    figr.show()

    # axv.set_title(f'{algoritmo} vvio')
    # axv.plot(range(0, len(vvio)), np.asarray(vvio), label=f'vvio')
    # figv.show()
    #
    # # Primer eje Y (izquierdo) con powerp_1 y powerp_2
    # axp.set_title(f'{algoritmo} Power P')
    # axp.plot(range(0, len(powerp_1)), np.asarray(powerp_1), label=f'power {agents[0]}')
    # axp.plot(range(0, len(powerp_2)), np.asarray(powerp_2), label=f'power {agents[1]}')
    # axp.plot(range(0, len(powerp_3)), np.asarray(powerp_3), label=f'power {agents[2]}')
    #
    # axp.set_ylabel(f"Power {agents[0]}")
    # axp.tick_params(axis='y', labelcolor='red')
    #
    # # # Segundo eje Y (derecho) con powerp_3
    # # axp2 = axp.twinx()
    # # axp2.plot(range(0, len(powerp_2)), np.asarray(powerp_2), label=f'power {agents[1]}', color='green')
    # # axp2.plot(range(0, len(powerp_3)), np.asarray(powerp_3), label=f'power {agents[2]}', color='blue')
    # # axp2.set_ylabel(f"Power {agents[1]} y {agents[2]}")
    # # axp2.tick_params(axis='y', labelcolor='black')
    #
    # # Leyenda combinada
    # lines1, labels1 = axp.get_legend_handles_labels()
    # # lines2, labels2 = axp2.get_legend_handles_labels()
    # # axp.legend(lines1 + lines2, labels1 + labels2, loc='best')
    # figp.tight_layout()
    # figp.show()
    #
    # # axp1.set_title(f'{algoritmo} - p_consumed')
    # # axp1.plot(range(0, len(p_consumed_pv)), np.asarray(p_consumed_pv), label=f'pv')
    # # axp1.plot(range(0, len(p_consumed_ev)), np.asarray(p_consumed_ev), label=f'ev')
    # # axp1.plot(range(0, len(p_consumed_building_building)), np.asarray(p_consumed_building_building), label=f'building_building')
    # # axp1.plot(range(0, len(p_consumed_building_pv)), np.asarray(p_consumed_building_pv), label=f'building_pv')
    # # axp1.plot(range(0, len(p_consumed_building_storage)), np.asarray(p_consumed_building_storage), label=f'building_storage')
    # # axp1.legend()
    # # figp1.show()
    # print(p_consumed_ev)

    ep = df.groupby("episode").mean()
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
    axv[0].legend(loc='best')
    axv[1].legend(loc='best')
    axv[2].legend(loc='upper left')
    axv2.legend(loc='upper right')
    axv[0].set_ylim(0.9,1.1)
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
    figp2.show()

    percentage_of_v_out_of_control = calcular_promedio_movil(ep["percentage_of_v_out_of_control"], ventana, mode)
    axp.set_title('percentage_of_v_out_of_control')
    axp.plot(range(0, len(percentage_of_v_out_of_control)), percentage_of_v_out_of_control, label='v_out_con')
    axp.legend(loc='best')
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
    axex.plot(range(0, len(np.asarray(ep["percentage_of_v_out_of_control"]))), np.asarray(ep["percentage_of_v_out_of_control"]), label='v_out_con')
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