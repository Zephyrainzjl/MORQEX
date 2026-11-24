"""Python-native v0 configuration; no YAML file required."""
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class GANConfig:
    images: tuple
    output: str
    phases: int = 3
    epochs: int = 41
    batch_size: int = 8
    workers: int = 0
    overrides: dict = field(default_factory=dict)

    def legacy_dict(self):
        p = str(Path(self.output).resolve())
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        cfg = dict(model_version='pix', project=dict(name=Path(p).name,dir=str(Path(p).parent)),
            data=dict(image_type='nphase',img_channels=self.phases,data_type='tif2D',
                data_path=[str(Path(i).resolve()) for i in self.images],image_size=64,scale=1),
            model_params=dict(d_kernel=[4]*5,d_stride=[2]*5,
                d_feature_channels=[self.phases,64,128,256,512,1],d_padding=[1,1,1,1,0],
                z_channels=32,g_scale=[2]*5,g_feature_channels=[32,1024,512,128,32,self.phases]),
            generation=dict(lf=16,periodic=[0,1,1]),
            training_params=dict(num_gpu=1,num_epochs=self.epochs,batch_size=self.batch_size,
                Discriminator_batch_size=self.batch_size,lr_Generator=.0001,lr_Discriminator=.0001,
                beta1=.9,beta2=.99,Lambda=10,critic_iters=5,cudnn_benchmark=True,workers=self.workers,lz=4),
            output_path=dict(out_pth=p))
        for section, values in self.overrides.items():
            if isinstance(values,dict) and isinstance(cfg.get(section),dict): cfg[section].update(values)
            else: cfg[section] = values
        return cfg

def train_gan(config):
    from morqex.microstructure.generate.v0.adversarial import Trainer
    return Trainer(config.legacy_dict())

def sample_gan(config, checkpoint=True, return_raw=False,spacing=(1.,1.,1.)):
    from morqex.microstructure.generate.v0.inference import generator_checkpoint, generator
    result=(generator_checkpoint if checkpoint else generator)(config.legacy_dict())
    if return_raw:return result
    import numpy as np
    import tifffile
    from morqex.microstructure.volume import Volume
    labels=np.unique(tifffile.imread(config.images[0]))
    if len(labels)!=config.phases:raise ValueError('Input label count does not match phases')
    legacy=np.asarray([int(255*i/(config.phases-1)) for i in range(config.phases)])
    data=result[0]
    if not np.isin(data,legacy).all():raise ValueError('Unexpected original-model phase labels')
    return Volume(labels[np.searchsorted(legacy,data)],spacing=spacing,metadata={
        'version':0,'checkpoint_prefix':str(Path(config.output).resolve()),
        'phase_label_mapping':dict(zip(map(int,legacy),map(int,labels)))})
