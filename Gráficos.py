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

train_path = 'MAPPOTrainer_voltage_case33_3min_final_732d2_00000_0_2026-07-01_10-40-40'
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

eval_path = 'MAPPOTrainer_voltage_case33_3min_final_00211_00000_0_2026-05-12_11-36-45'
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
    datos = np.asarray(datos, dtype=float)
    n = len(datos)

    if n < ventana:
        return datos.copy()

    offset = ventana // 2
    resultado = np.zeros(n)

    # Inicio: ventana creciente (clipping en el borde izquierdo)
    for i in range(offset):
        resultado[i] = np.mean(datos[0 : i + offset + 1])

    # Centro: ventana completa con convolución
    kernel = np.ones(ventana) / ventana
    centro = np.convolve(datos, kernel, mode='valid')
    resultado[offset : offset + len(centro)] = centro

    # Final: ventana decreciente (clipping en el borde derecho)
    inicio_final = offset + len(centro)
    for i in range(inicio_final, n):
        resultado[i] = np.mean(datos[i - offset : n])

    return resultado

if mode == 'train':
    with open(f'{url}'
              f'\\{train_path}'
              '\\result.json',
              'r') as file:
        train_data = []
        for episode in file:
            train_data.append(json.loads(episode))
    # df = pd.read_csv(f'{url}\\examples\\exp_results\\mappo_mlp_case33_3min_final\\{train_path}\\results\\physical_log.csv')
    df = pd.read_csv(f'C:\\PDN_runs\\Pruebas\\results\\physical_log_2026-07-01_10-40-59.csv', error_bad_lines=False, warn_bad_lines=False,)

    agents = train_data[0]['config']['model']['custom_model_config']['policy_mapping_info']['case33_3min_final']['team_prefix']
    agents = [f'agent_zone_{i+1}' for i in range(cantidad_agentes)]
    pol_agents = {f'agent_zone_{i+1}': f'pol_agent_zone_{i+1}' for i in range(cantidad_agentes)}
    figl, axl = plt.subplots(1,len(agents)+1, figsize=(12, 6))
    figr, axr = plt.subplots(1,len(agents)+1, figsize=(16, 6))
    figv, axv = plt.subplots(3, 1, figsize=(12, 10))
    figp, axp = plt.subplots(figsize=(12, 6))

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
    axv[0].set_title("Voltaje medio por episodio")
    axv[1].set_title("Potencia reactiva generada")
    axv[2].set_title("Porcentaje potencia perdida de línea")
    # axv1 = axv.twinx()
    # axv2 = axv.twinx()
    axv[0].plot(range(0, len(ep["v_mean"])), np.asarray(ep["v_mean"]), label='v_mean')
    axv[1].plot(range(0, len(ep["q_gen_total"])), np.asarray(ep["q_gen_total"]), label='q_gen_total', color='green')
    axv[2].plot(range(0, len(ep["line_loading"])), np.asarray(ep["line_loading"]), label='line_loading', color='red')
    axv[0].legend(loc='best')
    axv[1].legend(loc='best')
    axv[2].legend(loc='best')
    figv.show()

    axp.set_title('percentage_of_v_out_of_control')
    axp.plot(range(0, len(ep["percentage_of_v_out_of_control"])), np.asarray(ep["percentage_of_v_out_of_control"]), label='v_out_con')
    axp.legend(loc='best')
    figp.show()

    # ------ evaluación ------
elif mode == 'eval':
    # with open(f'{url}'
    #           f'\\examples\\exp_results\\{algoritmo}_mlp_PGW'
    #           f'\\{eval_path}'
    #           '\\eval_data1.json',
    #           'r') as file:
    #     eval_data = []
    #     for episode in file:
    #         eval_data.append(json.loads(episode))
    df = pd.read_csv(f'{url}\\examples\\exp_results\\mappo_mlp_case33_3min_final\\{eval_path}\\results\\physical_log.csv')
    # agents = eval_data[0]["rewards"].keys()
    agents = [f'agent_zone_{i + 1}' for i in range(cantidad_agentes)]
    # figre, axre = plt.subplots(figsize=(12, 6))
    figve, axve = plt.subplots(3, 1, figsize=(12, 6))
    figppe, axppe = plt.subplots(figsize=(12, 6))
    # figpce, axpce = plt.subplots(figsize=(12, 6))

    # axre.set_title(f'{algoritmo} episode reward')
    # for agent in agents:
    #     axre.plot(range(0, len(eval_data[0]["rewards"][agent])), np.asarray(eval_data[0]["rewards"][agent]), label=f'reward {agent}')
    # axre.legend()
    # figre.show()
    #
    # axve.set_title(f'{algoritmo} episode vvio')
    # axve.plot(range(0, len(eval_data[0]["vvio"])), np.asarray(eval_data[0]["vvio"]), label=f'vvio')
    # figve.show()
    #
    # power_p = {}
    # for agent in agents:
    #     power_p[agent] = calcular_promedio_movil(eval_data[0]["power_p"][agent], ventana)
    # axppe.set_title(f'{algoritmo} episode power p')
    # for agent in agents:
    #     axppe.plot(range(0, len(power_p[agent])), np.asarray(power_p[agent]), label=f'power p {agent}')
    # axppe.legend()
    # figppe.show()
    #
    # components = eval_data[0]["p_consumed"].keys()
    # power_p_cons = {}
    # axpce.set_title(f'{algoritmo} episode p consumed')
    # for comp in components:
    #     power_p_cons[comp] = calcular_promedio_movil(eval_data[0]["p_consumed"][comp], ventana)
    #     if 'building' in comp: # Si solo quiero graficar el consumo del edificio
    #         if 'ff' not in comp:
    #             axpce.plot(range(0, len(power_p_cons[comp])), np.asarray(power_p_cons[comp]), label=comp)
    # axpce.legend()
    # figpce.show()

    # Por step dentro de un episodio específico
    ep = df[df["episode"] == 1]  # 1 para summer, 0  para winter
    ep = ep.iloc[0:480]

    axve[0].set_title("Voltaje medio por episodio")
    axve[1].set_title("Potencia reactiva generada")
    axve[2].set_title("Porcentaje potencia perdida de línea")
    axve[0].plot(time_labels, np.asarray(ep["v_mean"]), label='v_mean')
    axve[1].plot(time_labels, np.asarray(ep["q_gen_total"]), label='q_gen_total', color='green')
    axve[2].plot(time_labels, np.asarray(ep["line_loading"]), label='line_loading', color='red')
    axve[0].set_xticks([])
    axve[1].set_xticks([])
    axve[2].set_xticks(tick_positions)
    axve[2].set_xticklabels(time_labels[tick_positions], rotation=45)
    axve[0].legend(loc='best')
    axve[1].legend(loc='best')
    axve[2].legend(loc='best')
    figve.show()

    axppe.set_title('percentage_of_v_out_of_control')
    axppe.plot(time_labels, np.asarray(ep["percentage_of_v_out_of_control"]), label='v_out_con')
    axppe.set_xticks(tick_positions)
    axppe.legend(loc='best')
    figppe.show()