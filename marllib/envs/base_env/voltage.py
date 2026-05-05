# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marllib.patch.dpn.var_voltage_control.voltage_control_env import VoltageControl
import numpy as np
from gym.spaces import Dict as GymDict, Box, Space
import os
import gymnasium

mode = 'decentralised'  # distributed
if mode == 'decentralised':
    team_prefix = ('agent_zone_1', 'agent_zone_2', 'agent_zone_3', 'agent_zone_4')
else:
    team_prefix = ('agent_pv_1', 'agent_pv_2', 'agent_pv_3', 'agent_pv_4', 'agent_pv_5', 'agent_pv_6')

policy_mapping_dict = {
    # "all_scenario": {
    #     "description": "voltage control all scenarios",
    #     "team_prefix": ("agent_",),
    #     "all_agents_one_policy": True,
    #     "one_agent_one_policy": True,
    # },
    "case33_3min_final": {
        "description": "voltage control custom",
        "team_prefix": team_prefix,
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    }
}

# global_data_source_path = os.getcwd()
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
max_steps = 1000

def flat_dim(space: Space) -> int:
    """
    Retorna la dimensión total si serializás el espacio en un vector.
    - Para Box, devuelve el producto de sus shape.
    - Para GymDict (o OrderedDict de espacios), suma flat_dim de cada sub-espacio.
    """
    # # Convierto los espacios de Gym a Gymnasium para poder sacar el shape total.
    # if isinstance(space, Box):
    #     space = GymnBox(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    # elif isinstance(space, GymDict):
    #     space = GymnDict({k: flat_dim(v) for k, v in space.spaces.items()})

    if isinstance(space, gymnasium.spaces.box.Box) or isinstance(space, Box):
        # producto de todas las dimensiones, p.ej. shape=(15,) -> 15, shape=(2,3)->6
        return int(np.prod(space.shape))
    elif isinstance(space, gymnasium.spaces.Dict) or isinstance(space, GymDict):
        return sum(flat_dim(sub) for sub in space.spaces.values())
    else:
        raise NotImplementedError(f"No sé calcular dimensión de {type(space)}")

class RLlibVoltageControl(MultiAgentEnv):

    def __init__(self, env_config):
        net_topology = env_config.pop("map_name", None)
        # net_topology = env_config.pop("net_topology", None)
        data_path = env_config["data_path"].split("/")
        # net_topology = "case322_3min_final"  # case33_3min_final / case141_3min_final / case322_3min_final
        data_path[-1] = net_topology
        env_config["data_path"] = "/".join(data_path)

        # set the action range
        assert net_topology in ['case33_3min_final', 'case141_3min_final',
                                'case322_3min_final'], f'{net_topology} is not a valid scenario.'
        if net_topology == 'case33_3min_final':
            env_config["action_bias"] = 0.0
            env_config["action_scale"] = 0.8
        elif net_topology == 'case141_3min_final':
            env_config["action_bias"] = 0.0
            env_config["action_scale"] = 0.6
        elif net_topology == 'case322_3min_final':
            env_config["action_bias"] = 0.0
            env_config["action_scale"] = 0.8

        # define control mode and voltage barrier function
        env_config["mode"] = mode  # distributed-->cada panel tiene un agente / decentralised-->cada zona tiene un agente (puede tener varios paneles/acciones)
        env_config["voltage_barrier_type"] = 'l1'
        env_config["data_path"] = os.path.join(project_root, "marllib\\patch\\dpn\\var_voltage_control\\data", #"marllib/patch/dpn/var_voltage_control/data",
                                               net_topology)
        self.env = VoltageControl(env_config)
        self.num_agents = self.env.get_num_of_agents()
        ###############
        # # espacio de acción iguales
        # self.agent_pv_map = {}
        # # Para saber cuántos PVs tiene la zona con más paneles para que el Action Space sea lo suficientemente grande.
        # zone_ids = self.env.base_powergrid.bus.zone.unique()
        # # Excluimos la zona 0 si es la main zone
        # zone_ids = [z for z in zone_ids if z != 'main']
        # for i, zone in enumerate(sorted(zone_ids)):
        #     agent_id = f"agent_{i}"
        #     # Buscamos qué sgen (paneles) están en buses de esta zona
        #     buses_in_zone = self.env.base_powergrid.bus[self.env.base_powergrid.bus.zone == zone].index
        #     pvs_indices = self.env.base_powergrid.sgen[self.env.base_powergrid.sgen.bus.isin(buses_in_zone)].index.tolist()
        #     self.agent_pv_map[agent_id] = pvs_indices
        #
        #     # 2. Definimos el espacio de acción según el que más tenga (ej. 2)
        # self.max_pvs = max(len(indices) for indices in self.agent_pv_map.values())
        # self.action_space = Box(
        #     self.env.action_space.low,
        #     self.env.action_space.high,
        #     shape=(self.max_pvs,),
        #     dtype=np.float32
        # )


        # Espacio de observación y de acción dferentes
        #### observaciones
        self.new_agents = []
        agents = policy_mapping_dict[net_topology]["team_prefix"]
        # self.agents = env_config["agents"]
        # for a in self.agents:
        #     # Top-level name argument overrides a name in the config.  The
        #     # constructor will error out otherwise because it gets two values
        #     # of the argument, one from config dict and one from the agent name.
        #     _config = a["config"]
        #     if "name" in a["config"]:
        #         _config = {k: v for k, v in _config.items() if k != "name"}
        #     # Call the constructor and append to the agent list.
        #     new_agent = a["cls"](name=a["name"], **_config, **env_config["common_config"])
        #     self.new_agents.append(new_agent)
        obs_size = 50.0
        agentes_externos = 6
        observaciones_agentes = 4
        local_dims = self.env.obs_size
        state_dim = sum(local_dims) + (agentes_externos * observaciones_agentes)
        self.observation_space = GymDict({})
        #for i, agent in enumerate(self.new_agents):
        for i in range(self.num_agents):
            loc_dim = local_dims[i]
            # Box para la obs local
            obs_box = Box(low=-obs_size, high=obs_size, shape=(loc_dim,), dtype=np.float64)
            # Box para el state global (idéntico para todos los agentes)
            state_box = Box(low=-obs_size, high=obs_size, shape=(state_dim,), dtype=np.float64)
            self.observation_space[agents[i]] = GymDict({
                "obs": obs_box,
                "state": state_box,
            })
        ###### acciones
        self.action_space = GymDict({})
        self.agent_pv_map = {}
        zone_ids = [z for z in self.env.base_powergrid.bus.zone.unique() if z != 0 and z != 'main']
        zone_ids = sorted(zone_ids)

        for i, zone in enumerate(zone_ids):
            agent_id = f"agent_zone_{i+1}"

            # 3. Buscamos qué buses pertenecen a esta zona
            buses_in_zone = self.env.base_powergrid.bus[self.env.base_powergrid.bus.zone == zone].index

            # 4. Filtramos los sgen (PVs) que están conectados a esos buses
            pvs_indices = self.env.base_powergrid.sgen[self.env.base_powergrid.sgen.bus.isin(buses_in_zone)].index.tolist()

            # 5. Guardamos el mapeo para usarlo luego en el método step()
            self.agent_pv_map[agent_id] = pvs_indices

            # 6. CREACIÓN DEL ESPACIO INDIVIDUAL
            # Calculamos cuántos PVs tiene esta zona específica
            num_pvs_in_zone = len(pvs_indices)

            # Creamos un Box cuyo shape es exactamente el número de PVs de esta zona
            # Usamos .low[0] y .high[0] asumiendo que los límites son uniformes
            self.action_space[agent_id] = Box(
                low=self.env.action_space.low,
                high=self.env.action_space.high,
                shape=(num_pvs_in_zone,),
                dtype=np.float32
            )


        # # Originales
        # self.action_space = Box(self.env.action_space.low, self.env.action_space.high, shape=(1,))
        # self.observation_space = GymDict({
        #     "obs": Box(-100.0, 100.0, shape=(self.env.get_obs_size(),), ),
        #     "state": Box(-100.0, 100.0, shape=(self.env.get_state_size(),), ),
        # })
        ###############
        self.agents = ["agent_zone_{}".format(i+1) for i in range(self.num_agents)]
        env_config["map_name"] = net_topology
        self.env_config = env_config

    def reset(self):
        o, s = self.env.reset()
        obs = {}
        for index, agent in enumerate(self.agents):
            obs[agent] = {
                "obs": np.float32(o[index]),
                "state": np.float32(s),
            }
        return obs

    def step(self, action_dict):
        ###############
        # Creamos un vector de ceros del tamaño total de paneles en la red
        num_total_pvs = len(self.env.base_powergrid.sgen)
        global_action = np.zeros(num_total_pvs)

        for agent_id, actions in action_dict.items():
            # 'actions' es un array de tamaño (max_pvs,), ej: [0.5, -0.2]
            pvs_del_agente = self.agent_pv_map[agent_id]  # ej: [5] (solo un panel)

            for i, pv_index in enumerate(pvs_del_agente):
                # Solo tomamos las acciones necesarias, las demás se ignoran
                global_action[pv_index] = actions[i]

        # Enviamos el vector completo al entorno de PDN
        r, d, info = self.env.step(global_action, action_dict)
        # action = [value[0] for value in action_dict.values()]
        # r, d, info = self.env.step(action)
        ###############
        o = self.env.get_obs()
        s = self.env.get_state()
        rewards = {}
        obs = {}
        for index, agent in enumerate(self.agents):
            obs[agent] = {
                "obs": np.float32(o[index]),
                "state": np.float32(s),
            }
            rewards[agent] = r[agent]
        dones = {"__all__": d}
        return obs, rewards, dones, {}

    def close(self):
        self.env.close()

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env_config["episode_limit"],
            "policy_mapping_info": policy_mapping_dict,
            "space_obs_per_agent": {  # solo 'obs' para el crítico centralizado
                agent_id: self.observation_space.spaces[agent_id].spaces["obs"]
                for agent_id in self.agents
            },
            "agent_name_ls": self.agents,
            # "seed": 111
        }
        return env_info
