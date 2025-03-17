from .base import register_artifact, register_log

BACKEND_MODULES = [
    "paibox.backend.mapper",
    "paibox.backend.graph",
    "paibox.backend.routing",
    "paibox.backend.placement",
    "paibox.backend.conf_exporting",
]

COMPONENTS = [
    "paibox.components.neuron.base",
    "paibox.components.neuron.neurons",
    "paibox.components.synapses.base",
    "paibox.components.synapses.synapses",
]

register_log("paibox", "paibox")
register_log("backend", BACKEND_MODULES)
register_log(
    "components",
    ["paibox.components.functional", "paibox.components.projection", *COMPONENTS],
)
register_log("sim", "simulator")

# mapper
register_artifact("build_core_blocks")
register_artifact("lcn_ex_adjustment")
register_artifact("cb_axon_grouping")
register_artifact("coord_assign")
register_artifact("get_dest")

# routing
register_artifact("routing_group_info")

# placement
register_artifact("core_block_info")
