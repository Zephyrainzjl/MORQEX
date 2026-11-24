"""Current data API plus migration of trusted historical Python artifacts."""
from pathlib import Path
import pickle
from .volume import Volume

class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        for old,new in [('core','morqex.microstructure.core'),('mesh','morqex.microstructure.mesh'),('helper','morqex.microstructure.helper'),('EMR','morqex.microstructure.generate.v0')]:
            if module == old or module.startswith(old+'.'):
                module=new+module[len(old):]
                break
        modules={'morqex.microstructure.core.DataVol':'morqex.microstructure.core.voxel_fields','morqex.microstructure.core.PCGCore':'morqex.microstructure.core.particle_graph',
                 'morqex.microstructure.core.Stats':'morqex.microstructure.core.distributions','morqex.microstructure.core.Voxelizer':'morqex.microstructure.core.rasterization',
                 'morqex.microstructure.mesh.Mesh':'morqex.microstructure.mesh.elements','morqex.microstructure.helper.PCGIO':'morqex.microstructure.helper.persistence',
                 'morqex.microstructure.helper.PCGView':'morqex.microstructure.helper.plotting'}
        modules.update({'morqex.microstructure.core.GraphParse':'morqex.microstructure.core.graph_codec',
            'morqex.microstructure.generate.v0.DataVol':'morqex.microstructure.generate.v0.fields',
            'morqex.microstructure.generate.v0.TensorDataset':'morqex.microstructure.generate.v0.patches',
            'morqex.microstructure.generate.v0.model_pix':'morqex.microstructure.generate.v0.synthesis',
            'morqex.microstructure.generate.v0.Trainer_pix':'morqex.microstructure.generate.v0.adversarial',
            'morqex.microstructure.generate.v0.load_config':'morqex.microstructure.generate.v0.settings',
            'morqex.microstructure.generate.v0.helper':'morqex.microstructure.generate.v0.operations',
            'DataVol':'morqex.microstructure.core.voxel_fields','PCGCore':'morqex.microstructure.core.particle_graph',
            'GraphParse':'morqex.microstructure.core.graph_codec','Stats':'morqex.microstructure.core.distributions',
            'Voxelizer':'morqex.microstructure.core.rasterization','model_pix':'morqex.microstructure.generate.v0.synthesis'})
        if module == 'gem' or module.startswith('gem.'):
            module='morqex.microstructure'+module[3:]
        module=modules.get(module,module)
        return super().find_class(module,name)

def load_legacy(path):
    """Load the user's trusted PCG pickle, without installing global import aliases."""
    with Path(path).open('rb') as f:return _LegacyUnpickler(f).load()

def convert_legacy_volume(path,output,spacing=(1.,1.,1.)):
    obj=load_legacy(path)
    data=getattr(obj,'vol',obj)
    return Volume(data,spacing=spacing,metadata={'converted_from':str(Path(path).resolve())}).save(output)
