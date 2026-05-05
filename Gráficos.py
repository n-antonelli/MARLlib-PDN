import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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
ventana = 20
algoritmo = "mappo"
dirección = {
    "mappo" : ['shared_policy','total_loss'],
    "ippo" : ['shared_policy','total_loss'],
    "vdppo" : ['shared_policy','total_loss'],
    "vda2c" : ['shared_policy','grad_gnorm'],
}

device = 'oficina'
mode = 'train'

train_path = 'MAPPOTrainer_voltage_case33_3min_final_f8af1_00000_0_2026-05-04_18-26-54'

eval_path = 'MAPPOTrainer_PGW_PGW_4029c_00000_0_2025-08-23_18-43-48'
cantidad_agentes = 4

if device == 'oficina':
    url = "C:\\Users\\Usuario\\Documents\\Programas\\MARLlib-PDN"
else:
    url = 'C:\\Users\\Nicolas\\Documents\\UNSL\\Programas\\MARLlib-PDN'


def calcular_promedio_movil(datos, tamano_ventana):
    """
    Calcula el promedio móvil de una lista de datos.

    Args:
        datos (list o np.array): La lista de valores de potencia a promediar.
        tamano_ventana (int): El número de pasos en la ventana de promediado.

    Returns:
        np.array: Un nuevo array con los datos suavizados.
    """
    # Convertir a array de numpy si no lo es, para facilitar las operaciones
    datos = np.array(datos)

    # Crear un array para almacenar los promedios suavizados
    datos_suavizados = []

    # Iterar sobre los datos y calcular el promedio para cada ventana
    # Esto asegura que no salgas de los límites del array
    for i in range(len(datos) - tamano_ventana + 1):
        # Selecciona la "ventana" de datos actual
        ventana = datos[i:i + tamano_ventana]

        # Calcula el promedio de esa ventana y lo añade a la lista
        promedio_ventana = np.mean(ventana)
        datos_suavizados.append(promedio_ventana)

    return np.array(datos_suavizados)

if mode == 'train':
    with open(f'{url}'
              f'\\examples\\exp_results\\mappo_mlp_case33_3min_final'
              f'\\{train_path}'
              '\\result.json',
              'r') as file:
        train_data = []
        for episode in file:
            train_data.append(json.loads(episode))

    agents = train_data[0]['config']['model']['custom_model_config']['policy_mapping_info']['case33_3min_final']['team_prefix']
    agents = [f'agent_zone_{i+1}' for i in range(cantidad_agentes)]
    pol_agents = {f'agent_zone_{i+1}': f'pol_agent_zone_{i+1}' for i in range(cantidad_agentes)}
    figl, axl = plt.subplots(1,len(agents)+1, figsize=(12, 6))
    loss_episode = []
    loss_episode_ag0 = []
    loss_episode_ag1 = []
    loss_episode_ag2 = []
    loss_episode_ag3 = []
    loss_episode_ag4 = []

    figr, axr = plt.subplots(1,len(agents)+1, figsize=(16, 6))
    reward_policy = []
    reward_episode = []
    reward_episode_ag0 = []
    reward_episode_ag1 = []
    reward_episode_ag2 = []
    reward_episode_ag3 = []
    reward_episode_ag4 = []

    figv, axv = plt.subplots(figsize=(12, 6))
    vvio = []

    figp, axp = plt.subplots(figsize=(12, 6))
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

    # axl.plot(range(0, len(loss_episode)), np.asarray(loss_episode), label=f'loss')
    for i in range(len(agents) + 1):
        if i == len(agents):
            axr[i].plot(range(0, len(loss_episode)), np.asarray(loss_episode), label='total_rew')
        else:
            axl[i].set_title(f'{algoritmo} loss - {agents[i]}')
            axl[i].plot(range(0, len(loss_episode_ag[agents[i]])), np.asarray(loss_episode_ag[agents[i]]), label=f'loss_{agents[i]}')

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
    # print(p_consumed_ev)  # TODO: Hay que promediar en 'on_episode_step'?

    # ------ evaluación ------
elif mode == 'eval':
    with open(f'{url}'
              f'\\examples\\exp_results\\{algoritmo}_mlp_PGW'
              f'\\{eval_path}'
              '\\eval_data1.json',
              'r') as file:
        eval_data = []
        for episode in file:
            eval_data.append(json.loads(episode))
    agents = eval_data[0]["rewards"].keys()
    figre, axre = plt.subplots(figsize=(12, 6))
    figve, axve = plt.subplots(figsize=(12, 6))
    figppe, axppe = plt.subplots(figsize=(12, 6))
    figpce, axpce = plt.subplots(figsize=(12, 6))

    axre.set_title(f'{algoritmo} episode reward')
    for agent in agents:
        axre.plot(range(0, len(eval_data[0]["rewards"][agent])), np.asarray(eval_data[0]["rewards"][agent]), label=f'reward {agent}')
    axre.legend()
    figre.show()

    axve.set_title(f'{algoritmo} episode vvio')
    axve.plot(range(0, len(eval_data[0]["vvio"])), np.asarray(eval_data[0]["vvio"]), label=f'vvio')
    figve.show()

    power_p = {}
    for agent in agents:
        power_p[agent] = calcular_promedio_movil(eval_data[0]["power_p"][agent], ventana)
    axppe.set_title(f'{algoritmo} episode power p')
    for agent in agents:
        axppe.plot(range(0, len(power_p[agent])), np.asarray(power_p[agent]), label=f'power p {agent}')
    axppe.legend()
    figppe.show()

    components = eval_data[0]["p_consumed"].keys()
    power_p_cons = {}
    axpce.set_title(f'{algoritmo} episode p consumed')
    for comp in components:
        power_p_cons[comp] = calcular_promedio_movil(eval_data[0]["p_consumed"][comp], ventana)
        if 'building' in comp: # Si solo quiero graficar el consumo del edificio
            if 'ff' not in comp:
                axpce.plot(range(0, len(power_p_cons[comp])), np.asarray(power_p_cons[comp]), label=comp)
    axpce.legend()
    figpce.show()