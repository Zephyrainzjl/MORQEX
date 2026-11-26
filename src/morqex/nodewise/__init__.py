"""Node-specific microstructure selection and coupled model composition."""
from .binding import BindingSpec,NodeGeometry,bind_microstructures
__all__=['BindingSpec','NodeGeometry','bind_microstructures','Physics','PreparedCase','prepare_nodes','mechanical_snapshot','HalfCellStack','lithium_halfcell_parameters']
def __getattr__(name):
    if name in {'HalfCellStack','lithium_halfcell_parameters'}:
        from importlib import import_module
        return getattr(import_module('.halfcell',__name__),name)
    if name in {'Physics','PreparedCase','prepare_nodes','mechanical_snapshot'}:
        from importlib import import_module
        return getattr(import_module('.case',__name__),name)
    raise AttributeError(name)
