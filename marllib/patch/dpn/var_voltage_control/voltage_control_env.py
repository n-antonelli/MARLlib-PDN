from ..multiagentenv import MultiAgentEnv
import numpy as np
import pandapower as pp
from pandapower import ppException
import pandas as pd
import copy
import os
from collections import namedtuple
from .pf_res_plot import pf_res_plotly
from .voltage_barrier.voltage_barrier_backend import VoltageBarrier
import functools


@functools.lru_cache(maxsize=1)
def load_shared_power_data(data_path, demand_scale):
    """Carga los datasets una sola vez en memoria y devuelve arreglos NumPy en float32."""
    pv_df = pd.read_csv(f"{data_path}/pv_active.csv", index_col=None)
    pv_df.index = pd.to_datetime(pv_df.iloc[:, 0])
    pv_df.index.name = 'time'
    pv_df = pv_df.iloc[::1, 1:] * demand_scale
    p_demand_df = pd.read_csv(f"{data_path}/load_active.csv", index_col=None)
    p_demand_df.index = pd.to_datetime(p_demand_df.iloc[:, 0])
    p_demand_df.index.name = 'time'
    p_demand_df = p_demand_df.iloc[::1, 1:] * demand_scale
    q_demand_df = pd.read_csv(f"{data_path}/load_reactive.csv", index_col=None)
    q_demand_df.index = pd.to_datetime(q_demand_df.iloc[:, 0])
    q_demand_df.index.name = 'time'
    q_demand_df = q_demand_df.iloc[::1, 1:] * demand_scale
    # demand_path = os.path.join(self.data_path, 'load_reactive.csv')
    # demand = pd.read_csv(demand_path, index_col=None)
    # demand.index = pd.to_datetime(demand.iloc[:, 0])
    # demand.index.name = 'time'
    # demand = demand.iloc[::1, 1:] * self.args.demand_scale

    # Precalcular desviaciones estándar escaladas
    return {
        "pv_data": pv_df,
        "p_demand_data": p_demand_df,
        "q_demand_data": q_demand_df,
        "pv_std": pv_df.values.std(axis=0) / 100.0,
        "p_demand_std": p_demand_df.values.std(axis=0) / 100.0,
        "q_demand_std": q_demand_df.values.std(axis=0) / 100.0,
    }


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


class ActionSpace(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high


class VoltageControl(MultiAgentEnv):
    """this class is for the environment of distributed active voltage control

        it is easy to interact with the environment, e.g.,

        state, global_state = env.reset()
        for t in range(240):
            actions = agents.get_actions(state) # a vector involving all agents' actions
            reward, done, info = env.step(actions)
            next_state = env.get_obs()
            state = next_state
    """

    def __init__(self, kwargs):
        """initialisation
        """
        # unpack args
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        # set the data path
        self.data_path = args.data_path

        # set the random seed
        np.random.seed(args.seed)
        
        # load the model of power network
        self.base_powergrid = self._load_network()

        ###############
        # load data
        # Original
        # self.pv_data = self._load_pv_data()
        # self.active_demand_data = self._load_active_demand_data()
        # self.reactive_demand_data = self._load_reactive_demand_data()
        # self.pv_std = self.pv_data.values.std(axis=0) / 100.0  # Porcentaje de desviación a escala decimal / per-unit (p.u.)
        # self.active_demand_std = self.active_demand_data.values.std(axis=0) / 100.0  # Porcentaje de desviación a escala decimal / per-unit (p.u.)
        # self.reactive_demand_std = self.reactive_demand_data.values.std(axis=0) / 100.0  # Porcentaje de desviación a escala decimal / per-unit (p.u.)

        # Modificación
        datos = load_shared_power_data(args.data_path, args.demand_scale)
        self.pv_data = datos["pv_data"]
        self.active_demand_data = datos["p_demand_data"]
        self.reactive_demand_data = datos["q_demand_data"]
        self.pv_std = datos["pv_std"]
        self.active_demand_std = datos["p_demand_std"]
        self.reactive_demand_std = datos["q_demand_std"]

        ###############

        # define episode and rewards
        self.episode_limit = args.episode_limit
        self.voltage_barrier_type = getattr(args, "voltage_barrier_type", "l1")
        self.voltage_weight = getattr(args, "voltage_weight", 1.0)
        self.q_weight = getattr(args, "q_weight", 0.1)
        self.line_weight = getattr(args, "line_weight", None)
        self.dv_dq_weight = getattr(args, "dq_dv_weight", None)
        self.main_weight = getattr(args, "main_weight", None)  # peso del componente global del reward

        # define constraints and uncertainty
        self.v_upper = getattr(args, "v_upper", 1.05)
        self.v_lower = getattr(args, "v_lower", 0.95)
        self._set_reactive_power_boundary()

        # define action space and observation space
        self.action_space = ActionSpace(low=-self.args.action_scale+self.args.action_bias, high=self.args.action_scale+self.args.action_bias)
        self.history = getattr(args, "history", 1)
        self.state_space = getattr(args, "state_space", ["pv", "demand", "reactive", "vm_pu", "va_degree", "q-1"])
        if self.args.mode == "distributed":
            self.n_actions = 1
            self.n_agents = len(self.base_powergrid.sgen)
        elif self.args.mode == "decentralised":
            self.n_actions = len(self.base_powergrid.sgen)
            self.n_agents = len(set(self.base_powergrid.bus["zone"].to_numpy(copy=True))) - 1  # exclude the main zone
        agents_obs, state = self.reset()
        ###############
        # original
        # self.obs_size = agents_obs[0].shape[0]
        # self.state_size = state.shape[0]
        # Modificar: utilizar el tamaño del que mayor observación tenga!
        # obs con distintos tamaños
        self.obs_size = [len(agents_obs[i]) for i in range(self.n_agents)]
        self.state_size = len(state) if isinstance(state, list) else state.shape[0]
        ###############
        self.last_v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        self.last_q = self.powergrid.sgen["q_mvar"].to_numpy(copy=True)

        # initialise voltage barrier function
        self.voltage_barrier = VoltageBarrier(self.voltage_barrier_type)
        self._rendering_initialized = False

        self.max_v_seen = 1e-6
        self.max_l_seen = 1e-6
        self.max_q_seen = 1e-6
        self.prev_actions = {}

        for i in range(self.n_agents):
            zone_res_buses = self.powergrid.res_bus.sort_index().loc[self.powergrid.bus["zone"] == f"zone{i + 1}"]

        num_sgens = len(self.powergrid.sgen)
        self.prev_q = {}

    def reset(self, reset_time=True):
        """reset the env
        """
        # reset the time step, cumulative rewards and obs history
        self.steps = 0
        self.sum_rewards = 0
        if self.history > 1:
            self.obs_history = {i: [] for i in range(self.n_agents)}

        # reset the power grid
        self.powergrid = copy.deepcopy(self.base_powergrid)
        solvable = False
        while not solvable:
            # reset the time stamp
            if reset_time:
                self._episode_start_hour = self._select_start_hour()
                ###########
                # # Seleccionar día propuesto (aleatorio o secuencial)
                # proposed_start_day = self._select_start_day()
                # # Validar e incrementar hasta encontrar un día con datos reales en la BD
                # self._episode_start_day, self.current_episode_data = self._get_valid_start_day(proposed_start_day)
                # ###########
                self._episode_start_day = self._select_start_day()
                self._episode_start_interval = self._select_start_interval()
            # get one episode of data
            self.pv_histories = self._get_episode_pv_history()
            self.active_demand_histories = self._get_episode_active_demand_history()
            self.reactive_demand_histories = self._get_episode_reactive_demand_history()
            self._set_demand_and_pv()
            # random initialise action
            if self.args.reset_action:
                self.powergrid.sgen["q_mvar"] = self.get_action()
                self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(self.powergrid.sgen["q_mvar"], self.powergrid.sgen["p_mw"])
            try:    
                pp.runpp(self.powergrid)
                solvable = True
            except ppException:
                print ("The power flow for the initialisation of demand and PV cannot be solved.")
                print (f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
                print (f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
                print (f"This is the active demand: \n{self.powergrid.load['p_mw']}")
                print (f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
                print (f"This is the res_bus: \n{self.powergrid.res_bus}")
                solvable = False
        self.prev_actions = {}
        self.prev_q = dict()
        for i in range(self.n_agents):
            zone_key = f"zone{i + 1}"
            sgens_in_zone = self.powergrid.sgen.loc[self.powergrid.sgen["name"] == zone_key]
            # Crea un arreglo de ceros del tamaño de sgens que pertenecen a esa zona
            self.prev_q[zone_key] = np.zeros(len(sgens_in_zone), dtype=np.float32)

        return self.get_obs(), self.get_state()
    
    def manual_reset(self, day, hour, interval):
        """manual reset the initial date
        """
        # reset the time step, cumulative rewards and obs history
        self.steps = 1
        self.sum_rewards = 0
        if self.history > 1:
            self.obs_history = {i: [] for i in range(self.n_agents)}

        # reset the power grid
        self.powergrid = copy.deepcopy(self.base_powergrid)

        # reset the time stamp
        self._episode_start_hour = hour
        self._episode_start_day = day
        self._episode_start_interval = interval
        solvable = False
        while not solvable:
            # get one episode of data
            self.pv_histories = self._get_episode_pv_history()
            self.active_demand_histories = self._get_episode_active_demand_history()
            self.reactive_demand_histories = self._get_episode_reactive_demand_history()
            self._set_demand_and_pv(add_noise=False)
            # random initialise action
            if self.args.reset_action:
                self.powergrid.sgen["q_mvar"] = self.get_action()
                self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(self.powergrid.sgen["q_mvar"], self.powergrid.sgen["p_mw"])
            try:    
                pp.runpp(self.powergrid)
                solvable = True
            except ppException:
                print ("The power flow for the initialisation of demand and PV cannot be solved.")
                print (f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
                print (f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
                print (f"This is the active demand: \n{self.powergrid.load['p_mw']}")
                print (f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
                print (f"This is the res_bus: \n{self.powergrid.res_bus}")
                solvable = False

        return self.get_obs(), self.get_state()

    def step(self, global_action, action_dict, add_noise=False):  # TODO: original = True
        """function for the interaction between agent and the env each time step
        """
        custom_metrics = {}
        last_powergrid = copy.deepcopy(self.powergrid)

        # check whether the power balance is unsolvable
        solvable = self._take_action(global_action)
        info = {'solvable': solvable}
        if not solvable:
            self.powergrid = last_powergrid  # restaurar ANTES de calcular reward
        # get the reward of current actions
        reward, info = self._calc_reward(action_dict, info)
        if not solvable:
            # print('Normal Reward: ', reward)
        # else:
            q_loss = np.mean( np.abs(self.powergrid.sgen["q_mvar"]) )
            print('Destroy Reward: ', reward)
            # for agent in reward.keys():
            #     reward[agent] -= 1  # 200.
            # keep q_loss
            info["destroy"] = 1.
            info["totally_controllable_ratio"] = 0.
            info["q_loss"] = q_loss
        # Agregar info de cada agente
        # for agent_id in action_dict.keys():
        #     # Ejemplo: guardar la tensión media de la zona del agente
        #     idx = int(agent_id.replace("zone", ""))
        #     info[agent_id].update({
        #         "mean_voltage": v_loss[idx],  # O el valor real vm_pu
        #         "power_loss": line_loss[idx],
        #         "q_injected": q_loss[idx]
        #     })

        custom_metrics['average_voltage'] = info["average_voltage"]
        custom_metrics['total_line_loss'] = info["total_line_loss"]
        custom_metrics['q_loss'] = info["q_loss"]

        self.powergrid['custom_metrics'] = custom_metrics

        #############
        # Original
        #
        # # set the pv and demand for the next time step
        # self._set_demand_and_pv(add_noise=add_noise)
        #
        # # terminate if episode_limit is reached
        # self.steps += 1
        # self.sum_rewards += sum(reward.values())
        # if self.steps >= self.episode_limit or not solvable:
        #     terminated = True
        # else:
        #     terminated = False

        # Modificación
        # Avanzar el contador de pasos al siguiente timestep
        self.steps += 1
        self.sum_rewards += sum(reward.values())

        # Verificar condición de término
        terminated = True if (self.steps >= self.episode_limit or not solvable) else False

        # Cargar la demanda y fotovoltaica para el SIGUIENTE paso (t + 1)
        if not terminated:
            self._set_demand_and_pv(add_noise=add_noise)
        # Guardar q previo
        for i in range(self.n_agents):
            self.prev_q[f"zone{i + 1}"] = self.powergrid.sgen["q_mvar"].loc[self.powergrid.sgen["name"] == f"zone{i + 1}"]
        # self.prev_q = list(self.powergrid.sgen["q_mvar"].sort_index().to_numpy(copy=True))
        # if terminated:
        #     print (f"Episode terminated at time: {self.steps} with return: {self.sum_rewards:2.4f}.")

        return reward, terminated, info

    def get_state(self):
        """return the global state for the power system
           the default state: voltage, active power of generators, bus state, load active power, load reactive power
        """
        state = []
        if "demand" in self.state_space:
            state += list(self.powergrid.res_bus["p_mw"].sort_index().to_numpy(copy=True))
            state += list(self.powergrid.res_bus["q_mvar"].sort_index().to_numpy(copy=True))
        if "pv" in self.state_space:
            state += list(self.powergrid.sgen["p_mw"].sort_index().to_numpy(copy=True))
        if "reactive" in self.state_space:
            state += list(self.powergrid.sgen["q_mvar"].sort_index().to_numpy(copy=True))
        if "vm_pu" in self.state_space:
            state += list(self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True))
        if "va_degree" in self.state_space:
            state += list(self.powergrid.res_bus["va_degree"].sort_index().to_numpy(copy=True))
        if "prev_q" in self.state_space:
            prev_q_flat = []

            if hasattr(self, "prev_q") and isinstance(self.prev_q, dict) and len(self.prev_q) > 0:
                # Recorrer ordenadamente las zonas para mantener la consistencia dimensional
                for i in range(self.n_agents):
                    zone_key = f"zone{i + 1}"
                    if zone_key in self.prev_q:
                        val = self.prev_q[zone_key]
                        # Extraer valores si es un Pandas Series, NumPy array o un escalar
                        if hasattr(val, "to_numpy"):
                            prev_q_flat.extend(val.to_numpy(copy=True).tolist())
                        elif isinstance(val, (list, np.ndarray)):
                            prev_q_flat.extend(list(val))
                        else:
                            prev_q_flat.append(float(val))
                state += prev_q_flat
            else:
                # Fallback de ceros con la cantidad total real de sgens en la red
                num_sgens = len(self.powergrid.sgen)
                state += [0.0] * num_sgens
        state = np.array(state)
        return state
    
    def get_obs(self):
        """return the obs for each agent in the power system
           the default obs: voltage, active power of generators, bus state, load active power, load reactive power
           each agent can only observe the state within the zone where it belongs
        """
        clusters = self._get_clusters_info()

        if self.args.mode == "distributed":
            obs_zone_dict = dict()
            zone_list = list()
            obs_len_list = list()
            for i in range(len(self.powergrid.sgen)):  # Para cada panel solar
                obs = list()
                zone_buses, zone, pv, q, sgen_bus = clusters[f"sgen{i}"]
                zone_list.append(zone)
                if not (zone in obs_zone_dict.keys()):
                    if "demand" in self.state_space:
                        copy_zone_buses = copy.deepcopy(zone_buses)
                        copy_zone_buses.loc[sgen_bus]["p_mw"] += pv
                        copy_zone_buses.loc[sgen_bus]["q_mvar"] += q
                        obs += list(copy_zone_buses.loc[:, "p_mw"].to_numpy(copy=True))
                        obs += list(copy_zone_buses.loc[:, "q_mvar"].to_numpy(copy=True))
                    if "pv" in self.state_space:
                        obs.append(pv)
                    if "reactive" in self.state_space:
                        obs.append(q)
                    if "vm_pu" in self.state_space:
                        obs += list(zone_buses.loc[:, "vm_pu"].to_numpy(copy=True))
                    if "va_degree" in self.state_space:
                        # transform the voltage phase to radian
                        obs += list(zone_buses.loc[:, "va_degree"].to_numpy(copy=True) * np.pi / 180)
                    obs_zone_dict[zone] = np.array(obs)
                obs_len_list.append(obs_zone_dict[zone].shape[0])
            agents_obs = list()
            obs_max_len = max(obs_len_list)
            for zone in zone_list:
                obs_zone = obs_zone_dict[zone]
                pad_obs_zone = np.concatenate([obs_zone, np.zeros(obs_max_len - obs_zone.shape[0])], axis=0)
                agents_obs.append(pad_obs_zone)
        ###########
        # observaciones diferentes tamaños
        elif self.args.mode == "decentralised":
            zone_obs_list = list()
            for i in range(self.n_agents):  # Para cada agente
                zone_buses, pv, q, sgen_buses, prev_q = clusters[f"zone{i + 1}"]
                obs = list()
                if "demand" in self.state_space:
                    copy_zone_buses = copy.deepcopy(zone_buses)
                    copy_zone_buses.loc[sgen_buses]["p_mw"] += pv
                    copy_zone_buses.loc[sgen_buses]["q_mvar"] += q
                    obs += list(copy_zone_buses.loc[:, "p_mw"].to_numpy(copy=True))
                    obs += list(copy_zone_buses.loc[:, "q_mvar"].to_numpy(copy=True))
                if "pv" in self.state_space:
                    obs += list(pv.to_numpy(copy=True))
                if "reactive" in self.state_space:
                    obs += list(q.to_numpy(copy=True))
                if "vm_pu" in self.state_space:
                    obs += list(zone_buses.loc[:, "vm_pu"].to_numpy(copy=True))
                if "va_degree" in self.state_space:
                    obs += list(zone_buses.loc[:, "va_degree"].to_numpy(copy=True) * np.pi / 180)
                if "prev_q" in self.state_space:
                    obs += list(np.copy(prev_q))

                # Guardamos la observación original
                zone_obs_list.append(np.array(obs))

            # agents_obs ahora es directamente la lista de observaciones reales
            agents_obs = zone_obs_list

        # Manejo de la historia (History)
        if self.history > 1:
            agents_obs_ = []
            for i, obs in enumerate(agents_obs):
                # crea ceros con el tamaño REAL de la observación de este agente específico, no del máximo global.
                if len(self.obs_history[i]) >= self.history - 1:
                    obs_ = np.concatenate(self.obs_history[i][-self.history + 1:] + [obs], axis=0)
                else:
                    # Rellenamos con ceros del tamaño local del agente i
                    zeros = [np.zeros_like(obs)] * (self.history - len(self.obs_history[i]) - 1)
                    obs_ = self.obs_history[i] + [obs]
                    obs_ = zeros + obs_
                    obs_ = np.concatenate(obs_, axis=0)

                agents_obs_.append(copy.deepcopy(obs_))
                self.obs_history[i].append(copy.deepcopy(obs))
            agents_obs = agents_obs_

        return agents_obs

        # original
        # elif self.args.mode == "decentralised":
        #     obs_len_list = list()
        #     zone_obs_list = list()
        #     for i in range(self.n_agents):  # Para cada agente
        #         zone_buses, pv, q, sgen_buses = clusters[f"zone{i+1}"]
        #         obs = list()
        #         if "demand" in self.state_space:
        #             copy_zone_buses = copy.deepcopy(zone_buses)
        #             copy_zone_buses.loc[sgen_buses]["p_mw"] += pv
        #             copy_zone_buses.loc[sgen_buses]["q_mvar"] += q
        #             obs += list(copy_zone_buses.loc[:, "p_mw"].to_numpy(copy=True))
        #             obs += list(copy_zone_buses.loc[:, "q_mvar"].to_numpy(copy=True))
        #         if "pv" in self.state_space:
        #             obs += list(pv.to_numpy(copy=True))
        #         if "reactive" in self.state_space:
        #             obs += list(q.to_numpy(copy=True))
        #         if "vm_pu" in self.state_space:
        #             obs += list(zone_buses.loc[:, "vm_pu"].to_numpy(copy=True))
        #         if "va_degree" in self.state_space:
        #             obs += list(zone_buses.loc[:, "va_degree"].to_numpy(copy=True) * np.pi / 180)
        #         obs = np.array(obs)
        #         zone_obs_list.append(obs)
        #         obs_len_list.append(obs.shape[0])
        #     agents_obs = []
        #     obs_max_len = max(obs_len_list)
        #     for obs_zone in zone_obs_list:
        #         pad_obs_zone = np.concatenate( [obs_zone, np.zeros(obs_max_len - obs_zone.shape[0])], axis=0 )
        #         agents_obs.append(pad_obs_zone)
        # if self.history > 1:
        #     agents_obs_ = []
        #     for i, obs in enumerate(agents_obs):
        #         if len(self.obs_history[i]) >= self.history - 1:
        #             obs_ = np.concatenate(self.obs_history[i][-self.history+1:]+[obs], axis=0)
        #         else:
        #             zeros = [np.zeros_like(obs)] * ( self.history - len(self.obs_history[i]) - 1 )
        #             obs_ = self.obs_history[i] + [obs]
        #             obs_ = zeros + obs_
        #             obs_ = np.concatenate(obs_, axis=0)
        #         agents_obs_.append(copy.deepcopy(obs_))
        #         self.obs_history[i].append(copy.deepcopy(obs))
        #     agents_obs = agents_obs_
        #
        # return agents_obs
        ###########
    def get_obs_agent(self, agent_id):
        """return observation for agent_id 
        """
        agents_obs = self.get_obs()
        return agents_obs[agent_id]
    
    def get_obs_size(self):
        """return the observation size
        """
        return self.obs_size

    def get_state_size(self):
        """return the state size
        """
        return self.state_size

    def get_action(self):
        """return the action according to a uniform distribution over [action_lower, action_upper)
        """
        rand_action = np.random.uniform(low=self.action_space.low, high=self.action_space.high, size=self.powergrid.sgen["q_mvar"].values.shape)
        return rand_action

    def get_total_actions(self):
        """return the total number of actions an agent could ever take 
        """
        return self.n_actions

    def get_avail_actions(self):
        """return available actions for all agents
        """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return np.expand_dims(np.array(avail_actions), axis=0)

    def get_avail_agent_actions(self, agent_id):
        """ return the available actions for agent_id 
        """
        if self.args.mode == "distributed":
            return [1]
        elif self.args.mode == "decentralised":
            avail_actions = np.zeros(self.n_actions)
            zone_sgens = self.base_powergrid.sgen.loc[self.base_powergrid.sgen["name"] == f"zone{agent_id+1}"]
            avail_actions[zone_sgens.index] = 1
            return avail_actions

    def get_num_of_agents(self):
        """return the number of agents
        """
        return self.n_agents

    def _get_voltage(self):
        return self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)

    def _create_basenet(self, base_net):
        """initilization of power grid
        set the pandapower net to use
        """
        if base_net is None:
            raise Exception("Please provide a base_net configured as pandapower format.")
        else:
            return base_net

    def _select_start_hour(self):
        """select start hour for an episode
        """
        if self.args.train_eval == 'train':
            return np.random.choice(24) # comenzar de forma aleatoria para el entrenamiento
        else:
            return 0
    
    def _select_start_interval(self):
        """select start interval for an episode
        """
        return np.random.choice( 60 // self.time_delta ) # Cantidad de datos por hora

    def _select_start_day(self):
        """select start day (date) for an episode
        """
        # Elige un número al azar entre los 1095 días que hay en el dataset (pv_data)
        pv_data = self.pv_data
        pv_days = (pv_data.index[-1] - pv_data.index[0]).days
        self.time_delta = (pv_data.index[1] - pv_data.index[0]).seconds // 60  # delta de minutos
        episode_days = ( self.episode_limit // (24 * (60 // self.time_delta) ) ) + 1  # margin
        max_valid_day = pv_days - episode_days
        eval_mode = getattr(self.args, 'eval_season', 'alternate')
        if getattr(self.args, 'train_eval', 'train') == 'train':
            return np.random.choice(max_valid_day)
        summer_months = [12, 1, 2]  # Dic, Ene, Feb
        winter_months = [6, 7, 8]  # Jun, Jul, Ago
        if eval_mode == 'alternate':
            # Alterna entre verano e invierno en cada llamada a reset()
            if not hasattr(self, '_eval_toggle'):
                self._eval_toggle = False
            self._eval_toggle = not self._eval_toggle
            target_months = summer_months if self._eval_toggle else winter_months
        elif eval_mode == 'winter':
            target_months = winter_months
        else:  # 'summer'
            target_months = summer_months
        # Generar el rango de fechas de inicio y filtrar índices
        start_date = pv_data.index[0]
        date_range = pd.date_range(start_date, periods=max_valid_day, freq='D')
        valid_days = np.where(date_range.month.isin(target_months))[0]

        return np.random.choice(valid_days)

    def _load_network(self):
        """load network
        """
        network_path = os.path.join(self.data_path, 'model.p')
        base_net = pp.from_pickle(network_path)
        return self._create_basenet(base_net)

    def _load_pv_data(self):
        """load pv data
        the sensor frequency is set to 3 or 15 mins as default
        """
        pv_path = os.path.join(self.data_path, 'pv_active.csv')
        pv = pd.read_csv(pv_path, index_col=None)
        pv.index = pd.to_datetime(pv.iloc[:, 0])
        pv.index.name = 'time'
        pv = pv.iloc[::1, 1:] * self.args.pv_scale
        return pv

    def _load_active_demand_data(self):
        """load active demand data
        the sensor frequency is set to 3 or 15 mins as default
        """
        demand_path = os.path.join(self.data_path, 'load_active.csv')
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[::1, 1:] * self.args.demand_scale
        return demand
    
    def _load_reactive_demand_data(self):
        """load reactive demand data
        the sensor frequency is set to 3 min as default
        """
        demand_path = os.path.join(self.data_path, 'load_reactive.csv')
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[::1, 1:] * self.args.demand_scale
        return demand

    def _get_episode_pv_history(self):
        """return the pv history in an episode
        """
        # Toma datos de 480 steps desde la hora de inicio aleatorio
        episode_length = self.episode_limit
        history = self.history
        start = self._episode_start_interval + self._episode_start_hour * (60 // self.time_delta) + self._episode_start_day * 24 * (60 // self.time_delta)
        nr_intervals = episode_length + history + 1  # margin of 1
        # episode_pv_history = self.pv_data[start:start + nr_intervals].values #Original
        episode_pv_history = self.pv_data.iloc[start:start + nr_intervals].values
        return episode_pv_history
    
    def _get_episode_active_demand_history(self):
        """return the active power histories for all loads in an episode
        """
        episode_length = self.episode_limit
        history = self.history
        start = self._episode_start_interval + self._episode_start_hour * (60 // self.time_delta) + self._episode_start_day * 24 * (60 // self.time_delta)
        nr_intervals = episode_length + history + 1  # margin of 1
        # episode_demand_history = self.active_demand_data[start:start + nr_intervals].values #Original
        episode_demand_history = self.active_demand_data.iloc[start: start + nr_intervals].values
        return episode_demand_history
    
    def _get_episode_reactive_demand_history(self):
        """return the reactive power histories for all loads in an episode
        """
        episode_length = self.episode_limit
        history = self.history
        start = self._episode_start_interval + self._episode_start_hour * (60 // self.time_delta) + self._episode_start_day * 24 * (60 // self.time_delta)
        nr_intervals = episode_length + history + 1  # margin of 1
        # episode_demand_history = self.reactive_demand_data[start:start + nr_intervals].values #Original
        episode_demand_history = self.reactive_demand_data.iloc[start : start + nr_intervals].values
        return episode_demand_history

    def _get_pv_history(self):
        """returns pv history
        """
        t = self.steps
        history = self.history
        return self.pv_histories[t:t+history, :]

    def _get_active_demand_history(self):
        """return the history demand
        """
        t = self.steps
        history = self.history
        return self.active_demand_histories[t:t+history, :]
    
    def _get_reactive_demand_history(self):
        """return the history demand
        """
        t = self.steps
        history = self.history
        return self.reactive_demand_histories[t:t+history, :]

    def _set_demand_and_pv(self, add_noise=True):
        """optionally update the demand and pv production according to the histories with some i.i.d. gaussian noise
        """
        # Agrega ruido a la demanda de potencia activa, reactiva y generación de potencia activa del pv
        pv = copy.copy(self._get_pv_history()[0, :])

        # add uncertainty to pv data with unit truncated gaussian (only positive accepted)
        if add_noise:
            pv += self.pv_std * np.abs(np.random.randn(*pv.shape))
        active_demand = copy.copy(self._get_active_demand_history()[0, :])

        # add uncertainty to active power of demand data with unit truncated gaussian (only positive accepted)
        if add_noise:
            active_demand += self.active_demand_std * np.abs(np.random.randn(*active_demand.shape))
        reactive_demand = copy.copy(self._get_reactive_demand_history()[0, :])

        # add uncertainty to reactive power of demand data with unit truncated gaussian (only positive accepted)
        if add_noise:
            reactive_demand += self.reactive_demand_std * np.abs(np.random.randn(*reactive_demand.shape))

        # update the record in the pandapower
        self.powergrid.sgen["p_mw"] = pv
        self.powergrid.load["p_mw"] = active_demand
        self.powergrid.load["q_mvar"] = reactive_demand

    def _set_reactive_power_boundary(self):
        """set the boundary of reactive power
        """
        self.factor = 1.2
        self.p_max = self.pv_data.to_numpy(copy=True).max(axis=0)
        self.s_max = self.factor * self.p_max
        # print (f"This is the s_max: \n{self.s_max}")

    def _get_clusters_info(self):
        """return the clusters of info
        the clusters info is divided by predefined zone
        distributed: each zone is equipped with several PV generators and each PV generator is an agent
        decentralised: each zone is controlled by an agent and each agent may have variant number of actions
        """
        clusters = dict()  # zone_res_buses, pv, q, sgen_res_buses
        if self.args.mode == "distributed":
            for i in range(len(self.powergrid.sgen)):
                zone = self.powergrid.sgen["name"].iloc[i]
                sgen_bus = self.powergrid.sgen["bus"].iloc[i]
                pv = self.powergrid.sgen["p_mw"].iloc[i]
                q = self.powergrid.sgen["q_mvar"].iloc[i]
                zone_res_buses = self.powergrid.res_bus.sort_index().loc[self.powergrid.bus["zone"] == zone]
                clusters[f"sgen{i}"] = (zone_res_buses, zone, pv, q, sgen_bus)
        elif self.args.mode == "decentralised":
            for i in range(self.n_agents):
                zone_res_buses = self.powergrid.res_bus.sort_index().loc[self.powergrid.bus["zone"] == f"zone{i + 1}"]
                sgen_res_buses = self.powergrid.sgen["bus"].loc[self.powergrid.sgen["name"] == f"zone{i + 1}"]
                pv = self.powergrid.sgen["p_mw"].loc[self.powergrid.sgen["name"] == f"zone{i + 1}"]
                q = self.powergrid.sgen["q_mvar"].loc[self.powergrid.sgen["name"] == f"zone{i + 1}"]
                prev_q = self.prev_q[f"zone{i + 1}"]
                clusters[f"zone{i + 1}"] = (zone_res_buses, pv, q, sgen_res_buses, prev_q)

        return clusters
    
    def _take_action(self, actions):
        """take the control variables
        the control variables we consider are the exact reactive power
        of each distributed generator
        """
        # Calcula la potencia reactiva en base a la potencia aparente máxima y la potencia activa actual.
        # La multiplica por las acciones
        self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(actions, self.powergrid.sgen["p_mw"])

        # solve power flow to get the latest voltage with new reactive power and old deamnd and PV active power
        try:
            pp.runpp(self.powergrid)
            return True
        except ppException:
            print ("The power flow for the reactive power penetration cannot be solved.")
            print (f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
            print (f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
            print (f"This is the active demand: \n{self.powergrid.load['p_mw']}")
            print (f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
            print (f"This is the res_bus: \n{self.powergrid.res_bus}")
            return False
    
    def _clip_reactive_power(self, reactive_actions, active_power):
        """clip the reactive power to the hard safety range
        """
        reactive_power_constraint = np.sqrt(self.s_max**2 - active_power**2)
        return reactive_power_constraint * reactive_actions
    
    def _calc_reward(self, action_dict, info={}):
        """reward function
        consider 5 possible choices on voltage barrier functions:
            l1
            l2
            courant_beltrami
            bowl
            bump
        """
        # percentage of voltage out of control
        v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        percent_of_v_out_of_control = ( np.sum(v < self.v_lower) + np.sum(v > self.v_upper) ) / v.shape[0]
        info["percentage_of_v_out_of_control"] = percent_of_v_out_of_control  # porcentaje de tensión fuera de control
        info["percentage_of_lower_than_lower_v"] = np.sum(v < self.v_lower) / v.shape[0]  # porcentaje de tensión baja
        info["percentage_of_higher_than_upper_v"] = np.sum(v > self.v_upper) / v.shape[0]  # porcentaje de tensión alta
        info["totally_controllable_ratio"] = 0. if percent_of_v_out_of_control > 1e-3 else 1.  # porcentaje de tensión controlable

        # voltage violation
        v_ref = 0.5 * (self.v_lower + self.v_upper)
        info["average_voltage_deviation"] = np.mean( np.abs( v - v_ref ) )  # promedio de desviación de tensión
        info["average_voltage"] = np.mean(v)  # promedio de tensión
        info["max_voltage_drop_deviation"] = np.max( (v < self.v_lower) * (self.v_lower - v) )  # máxima desviación de tensión por encima
        info["max_voltage_rise_deviation"] = np.max( (v > self.v_upper) * (v - self.v_upper) )  # máxima desviación de tensión por debajo

        # line loss
        line_loss = np.sum(self.powergrid.res_line["pl_mw"])
        avg_line_loss = np.mean(self.powergrid.res_line["pl_mw"])
        info["total_line_loss"] = line_loss  # Total de pérdida potencia activa en la línea

        # reactive power (q) loss
        q = self.powergrid.res_sgen["q_mvar"].sort_index().to_numpy(copy=True)
        q_loss = np.mean(np.abs(q))
        info["q_loss"] = q_loss  # Promedio de potencia reactiva
        #############
        # # reward function original
        # ## voltage barrier function
        # v_loss = np.mean(self.voltage_barrier.step(v)) * self.voltage_weight
        # ## add soft constraint for line or q
        # # Se puede agregar loss de la línea o el loss de potencia reactiva
        # if self.line_weight != None:
        #     loss = avg_line_loss * self.line_weight + v_loss
        # elif self.q_weight != None:
        #     loss = q_loss * self.q_weight + v_loss
        # else:
        #     raise NotImplementedError("Please at least give one weight, either q_weight or line_weight.")
        # reward = -loss
        # print(self.steps, ' ---- v_loss: ', v_loss,' - ', 'q_loss: ', q_loss)

        # rewards separados
        # v_loss = np.zeros(self.n_agents + 1)
        # line_loss = np.zeros(self.n_agents + 1)
        # q_loss = np.zeros(self.n_agents + 1)
        # names = [name for name in action_dict.keys()]
        # names.insert(0, 'main')
        # for i in range(self.n_agents+1):
        #     zone_name = "main" if i == 0 else f"zone{i}"
        #
        #     v_zone = self.powergrid.res_bus["vm_pu"].loc[self.powergrid.bus["zone"] == zone_name]
        #     v_loss[i] = np.mean(self.voltage_barrier.step(v_zone))
        #
        #     bus_in_zone = self.powergrid.bus.index[self.powergrid.bus["zone"] == zone_name]
        #     p_zone = self.powergrid.res_line["pl_mw"].loc[self.powergrid.line["to_bus"].isin(bus_in_zone)]
        #     #p_zone = self.powergrid.res_line["pl_mw"].loc[self.powergrid.bus["zone"] == zone_name]
        #     line_loss[i] = np.mean(np.abs(p_zone))
        #
        #     q_zone = self.powergrid.res_sgen["q_mvar"].loc[self.powergrid.sgen["name"] == zone_name]
        #     q_loss[i] = np.mean(np.abs(q_zone))  # abs iguala q inductivo y capacitivo
        #
        # if np.isnan(q_loss[0]):
        #     q_loss[0] = 0.0
        # for i, name in enumerate(names):
        #     if name != 'main':
        #         # loss de cada agente más el loss del main
        #         v_loss_total = v_loss[i] + v_loss[0] * self.main_weight
        #         line_loss_total = line_loss[i] + line_loss[0] * self.main_weight
        #         q_loss_total = q_loss[i] + q_loss[0] * self.main_weight  # si no hay panel --> q_loss[0] = Nan
        #         if self.line_weight != None:
        #             loss = line_loss_total * self.line_weight + v_loss_total * self.voltage_weight
        #         elif self.q_weight != None:
        #             loss = q_loss_total * q_weight_solv + v_loss_total * self.voltage_weight
        #         else:
        #             raise NotImplementedError("Please at least give one weight, either q_weight or line_weight.")
        #         reward[name] = -loss
        #
        #
        # # record destroy
        # info["destroy"] = 0.0
        #
        # return reward, info

        q_weight_solv = self.q_weight * 2 if not info['solvable'] else self.q_weight
        reward = dict()


        v_losses = {}
        line_losses = {}
        q_losses = {}
        q_values = {}

        zones = [f"zone{i}" for i in range(1, self.n_agents + 1)] + ["main"]

        for zone in zones:
            # --- Cálculo de V-Loss ---
            v_zone = self.powergrid.res_bus["vm_pu"].loc[self.powergrid.bus["zone"] == zone]
            # v_losses[zone] = np.mean(self.voltage_barrier.step(v_zone))
            v_barrier_values = self.voltage_barrier.step(v_zone)
            # effective_losses = np.maximum(0.0, v_barrier_values - 0.05) #
            v_losses[zone] = np.max(v_barrier_values)

            # --- Cálculo de Line-Loss (con mapeo to_bus para evitar el error 33 vs 32) ---
            buses_in_zone = self.powergrid.bus.index[self.powergrid.bus["zone"] == zone]
            p_zone_lines = self.powergrid.res_line["pl_mw"].loc[self.powergrid.line["to_bus"].isin(buses_in_zone)]
            line_losses[zone] = np.mean(np.abs(p_zone_lines)) if not p_zone_lines.empty else 0.0

            # --- Cálculo de Q-Loss (Sgen) ---
            q_vals = self.powergrid.res_sgen["q_mvar"].loc[self.powergrid.sgen["name"] == zone]
            # Manejo de NaNs: si no hay sgen o es NaN, ponemos 0.0
            q_values[zone] = self.powergrid.res_sgen["q_mvar"].loc[self.powergrid.sgen["name"] == zone]
            q_losses[zone] = np.nan_to_num(np.mean(np.abs(q_vals))) if not q_vals.empty else 0.0  # TODO: verificar si conviene sacar el abs (capacitivo=inductivo)

        if np.isnan(q_losses["main"]):
            q_losses["main"] = 0.0
        # 2. Asignamos los rewards a los agentes usando el mapeo de nombres
        for agent_name in action_dict.keys():
            # Si el agente se llama 'agent_zone_1', la zona es 'zone_1'
            zone_id = agent_name.replace("agent_", "").replace("_", "")

            # # Mezclamos pérdida local + pérdida de la cabecera (main)
            # # Reward_i = -(Loss_local + weight * Loss_main)
            # v_total = v_losses[zone_id] + v_losses["main"] * self.main_weight
            #
            # # Actualizar y escalar tensión
            # self.max_v_seen = max(self.max_v_seen, v_total)
            # v_scaled = v_total / self.max_v_seen
            #
            # if self.line_weight is not None:
            #     # 2a. Calcular pérdidas brutas de línea
            #     l_total = line_losses[zone_id] + line_losses["main"] * self.main_weight
            #
            #     # Actualizar y escalar línea
            #     self.max_l_seen = max(self.max_l_seen, l_total)
            #     l_scaled = l_total / self.max_l_seen
            #
            #     # Calcular pérdida final ponderada (máximo teórico de cada término es 1)
            #     loss = (l_scaled * self.line_weight) + (v_scaled * self.voltage_weight)
            # else:
            #     # 2b. Calcular pérdidas brutas de reactiva
            #     q_total = q_losses[zone_id] + q_losses["main"] * self.main_weight
            #
            #     # Actualizar y escalar reactiva
            #     self.max_q_seen = max(self.max_q_seen, q_total)
            #     q_scaled = q_total / self.max_q_seen
            #
            #     # Calcular pérdida final ponderada
            #     loss = (q_scaled * q_weight_solv) + (v_scaled * self.voltage_weight)
            #
            # reward[agent_name] = -float(loss)
            # 1. Definir bases fijas de normalización (evitar max_seen dinámicos)
            V_BASE = 0.05
            Q_BASE_MVAR = 8.0  # Capacidad máxima esperada por zona en MVAR
            DQ_BASE_MVAR = 0.60 # Máxima variación esperada por zona en MVAR
            Line_BASE = 0.5  # Máxima pérdida esperada promedio

            # Prueba de losses escalados antes de la suma
            # --- 1. Pérdida de Tensión (Normalizar por separado) ---
            v_zone_scaled = v_losses[zone_id] / V_BASE
            v_main_scaled = v_losses["main"] / V_BASE
            v_scaled = v_zone_scaled + self.main_weight * v_main_scaled

            # --- 2. Pérdida de Línea (Elevar al cuadrado individualmente) ---
            line_zone_scaled = (line_losses[zone_id] / Line_BASE) ** 2
            line_main_scaled = (line_losses["main"] / Line_BASE) ** 2
            line_scaled = line_zone_scaled + self.main_weight * line_main_scaled

            # --- 3. Pérdida de Reactiva (Elevar al cuadrado individualmente) ---
            q_zone_scaled = (q_losses[zone_id] / Q_BASE_MVAR) ** 2
            q_main_scaled = (q_losses["main"] / Q_BASE_MVAR) ** 2
            q_scaled = q_zone_scaled + self.main_weight * q_main_scaled


            # # Pérdida de Tensión (usar barrier directa sin dividir por max_v)
            # v_total = v_losses[zone_id] + v_losses["main"] * self.main_weight
            # v_scaled = v_total / V_BASE
            # # Pérdida de Línea
            # line_total = line_losses[zone_id] + line_losses["main"] * self.main_weight
            # line_scaled = (line_total / Line_BASE) ** 2  # Penaliza cuadráticamente picos de Q
            # # Pérdida de Reactiva (normalizada por base fija, penalización cuadrática)
            # q_total = q_losses[zone_id] + q_losses["main"] * self.main_weight
            # q_scaled = (q_total / Q_BASE_MVAR) ** 2  # Penaliza cuadráticamente picos de Q
            # # Penalización por suavizado / variación de acción (evita oscilaciones bang-bang)
            q_action_curr = np.max(q_values[zone_id])
            q_action_prev = self.prev_actions.get(zone_id, q_action_curr)
            delta_q_loss = (q_action_curr - q_action_prev)
            delta_q_scaled = (delta_q_loss / DQ_BASE_MVAR) ** 2
            self.prev_actions[zone_id] = q_action_curr
            if self.line_weight != None:
                line_loss = line_scaled * self.line_weight
            else:
                line_loss = 0
            if self.q_weight != None:
                q_loss = q_scaled * self.q_weight
            else:
                q_loss = 0
            v_loss = v_scaled * self.voltage_weight
            delta_q_loss = delta_q_scaled * self.dv_dq_weight

            loss = (v_loss) + \
                   line_loss + \
                   q_loss + \
                   (delta_q_loss)

            reward[agent_name] = -float(loss)

        #############

        info["destroy"] = 0.0
        return reward, info


    def _get_res_bus_v(self):
        v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        return v
    
    def _get_res_bus_active(self):
        active = self.powergrid.res_bus["p_mw"].sort_index().to_numpy(copy=True)
        return active

    def _get_res_bus_reactive(self):
        reactive = self.powergrid.res_bus["q_mvar"].sort_index().to_numpy(copy=True)
        return reactive

    def _get_res_line_loss(self):
        line_loss = self.powergrid.res_line["pl_mw"].sort_index().to_numpy(copy=True)
        return line_loss

    def _get_sgen_active(self):
        active = self.powergrid.sgen["p_mw"].to_numpy(copy=True)
        return active
    
    def _get_sgen_reactive(self):
        reactive = self.powergrid.sgen["q_mvar"].to_numpy(copy=True)
        return reactive
    
    def _init_render(self):
        from .rendering_voltage_control_env import Viewer
        self.viewer = Viewer()
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, return_rgb_array=(mode == "rgb_array"))

    def res_pf_plot(self):
        if not os.path.exists("marllib/patch/dpn/var_voltage_control/plot_save"):
            os.mkdir("marllib/patch/dpn/var_voltage_control/plot_save")

        fig = pf_res_plotly(self.powergrid, 
                            aspectratio=(1.0, 1.0), 
                            filename="marllib/patch/dpn/var_voltage_control/plot_save/pf_res_plot.html",
                            auto_open=False,
                            climits_volt=(0.9, 1.1),
                            line_width=5, 
                            bus_size=12,
                            climits_load=(0, 100),
                            cpos_load=1.1,
                            cpos_volt=1.0
                        )
        fig.write_image("marllib/patch/dpn/var_voltage_control/plot_save/pf_res_plot.jpeg")
