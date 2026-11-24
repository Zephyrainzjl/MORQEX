"""Optional diffusion-guided conditional SliceGAN student, without 3D training data.

Teacher DDPM score distillation + 2D hinge adversarial training + composition
constraints. Experimental research stage; speed and quality need benchmarking.
"""
from pathlib import Path
from dataclasses import dataclass,asdict
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .v1 import _model,_images,_crop,_input,_atomic_save,DiffusionConfig
from diffusers import DDPMScheduler
from morqex.microstructure import Volume

class Generator(nn.Module):
    def __init__(self,phases,noise_channels=16,width=64,residual=False,tri_reference=False,normalize_reference=False,zero_reference=False):
        super().__init__()
        self.phases=phases;self.noise_channels=noise_channels
        self.reference = nn.Sequential(nn.Conv2d(phases,32,3,padding=1),nn.SiLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
        self.tri_reference=tri_reference
        self.normalize_reference=normalize_reference
        self.zero_reference=zero_reference
        self.condition = nn.Sequential(nn.Linear(phases+(192 if tri_reference else 64),128),nn.SiLU(),nn.Linear(128,128))
        self.stem = nn.Conv3d(noise_channels,width,3,padding=1,padding_mode='circular')
        channels=[width,width,width//2,width//4,width//4]
        self.convs=nn.ModuleList();self.norms=nn.ModuleList();self.films=nn.ModuleList()
        for a,b in zip(channels[:-1],channels[1:]):
            self.convs.append(nn.Conv3d(a,b,3,padding=1,padding_mode='circular'))
            self.norms.append(nn.GroupNorm(8,b,affine=False))
            self.films.append(nn.Linear(128,2*b))
        self.out=nn.Conv3d(channels[-1],phases,1)
        self.refinement=nn.ModuleList([nn.Conv3d(c,c,3,padding=1,padding_mode='circular') for c in channels[1:]]) if residual else None

    def forward(self,z,reference,fractions):
        if self.tri_reference:
            b,d,k,h,w=reference.shape
            reference=reference.reshape(b*d,k,h,w)
        if self.normalize_reference:
            reference=(reference-reference.mean((2,3),keepdim=True))/reference.std((2,3),keepdim=True).clamp_min(.1)
        features=self.reference(reference).flatten(1)
        if self.tri_reference:features=features.reshape(b,-1)
        if self.zero_reference:features=torch.zeros_like(features)
        style=self.condition(torch.cat((features,fractions),1))
        x=self.stem(z)
        for index,(conv,norm,film) in enumerate(zip(self.convs,self.norms,self.films)):
            x=F.interpolate(x,scale_factor=2,mode='trilinear',align_corners=False)
            x=norm(conv(x));scale,bias=film(style).chunk(2,1)
            x=F.silu(x*(1+scale[:,:,None,None,None])+bias[:,:,None,None,None])
            if self.refinement is not None:x=x+.1*F.silu(self.refinement[index](x))
        return self.out(x)

class Discriminator(nn.Module):
    def __init__(self,phases):
        super().__init__()
        spectral=nn.utils.parametrizations.spectral_norm
        layers=[]
        for a,b in zip([phases+phases+3,32,64],[32,64,128]):
            layers += [spectral(nn.Conv2d(a,b,4,2,1)),nn.LeakyReLU(.2)]
        layers += [spectral(nn.Conv2d(128,1,3,1,1))]
        self.net=nn.Sequential(*layers)
    def forward(self,x,fractions,axis):
        b,_,h,w=x.shape
        axis=F.one_hot(torch.full((b,),axis,device=x.device),3).float()
        return self.net(torch.cat((x,fractions[:,:,None,None].expand(-1,-1,h,w),
            axis[:,:,None,None].expand(-1,-1,h,w)),1))

@dataclass(frozen=True)
class HybridConfig:
    steps:int=10000
    seconds:float=1200.
    batch_size:int=2
    slices_per_volume:int=8
    lr:float=2e-4
    sds_weight:float=1.
    adversarial_weight:float=.1
    fraction_weight:float=10.
    correlation_weight:float=2.
    width:int=64
    residual:bool=False
    tri_reference:bool=False
    normalize_reference:bool=False
    zero_reference:bool=False
    categorical_straight_through:bool=False
    noise_channels:int=16
    checkpoint_every:int=100
    seed:int=42
    device:str='cuda'

def _slices(x,axis,count):
    b,k,d,h,w=x.shape
    ids=torch.randint(x.shape[axis+2],(count,),device=x.device)
    return x.index_select(axis+2,ids).movedim(axis+2,1).reshape(b*count,k,*[x.shape[j] for j in range(2,5) if j != axis+2])

def _correlation_loss(fake,real):
    a=(fake+1)/2;b=(real+1)/2
    loss=fake.new_zeros(())
    for lag in (1,2,4,8):
        if lag >= min(a.shape[-2:]):continue
        for dim in (-1,-2):
            pa=a.narrow(dim,0,a.shape[dim]-lag);qa=a.narrow(dim,lag,a.shape[dim]-lag)
            pb=b.narrow(dim,0,b.shape[dim]-lag);qb=b.narrow(dim,lag,b.shape[dim]-lag)
            # Normalized by phase fraction to avoid dominant-phase masking.
            observed=(pb*qb).mean((0,2,3)).detach()
            prediction=(pa*qa).mean((0,2,3))
            loss=loss+((prediction-observed)**2/(b.mean((0,2,3)).detach()+.05)).mean()
    return loss

def distill(teacher_checkpoint,paths,run_dir,config=None):
    cfg=config or HybridConfig()
    torch.manual_seed(cfg.seed)
    state=torch.load(teacher_checkpoint,map_location='cpu',weights_only=False)
    tc=DiffusionConfig(**{**state['config'],'device':cfg.device})
    teacher=_model(tc).to(cfg.device).eval().requires_grad_(False)
    teacher.load_state_dict(state['ema'])
    imgs,labels=_images(paths,tc,state['labels'])
    g=Generator(tc.phases,cfg.noise_channels,cfg.width,cfg.residual,cfg.tri_reference,cfg.normalize_reference,cfg.zero_reference).to(cfg.device)
    ema=copy.deepcopy(g).eval().requires_grad_(False)
    d=Discriminator(tc.phases).to(cfg.device)
    og=torch.optim.AdamW(g.parameters(),lr=cfg.lr,betas=(.5,.99))
    od=torch.optim.AdamW(d.parameters(),lr=cfg.lr,betas=(.5,.99))
    sched=DDPMScheduler(num_train_timesteps=tc.diffusion_steps,beta_schedule='squaredcos_cap_v2')
    run=Path(run_dir);run.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    step=0
    def save():
        _atomic_save(dict(generator=ema.state_dict(),config=asdict(cfg),phases=tc.phases,
            size=tc.image_size,labels=labels,paths=[str(Path(p).resolve()) for p in paths],
            teacher=str(teacher_checkpoint),step=step),run/'student.pt')
    try:
        with (run/'metrics.jsonl').open('a') as log:
            for step in range(1,cfg.steps+1):
                axis=step%3
                real=_crop(imgs[axis],cfg.batch_size*cfg.slices_per_volume,tc.image_size,tc.phases)
                if cfg.tri_reference:
                    all_refs=torch.stack([_crop(img,cfg.batch_size,tc.image_size,tc.phases) for img in imgs],1)
                    ref=all_refs[:,axis]
                    frac=((all_refs+1)/2).mean((1,3,4))
                    generator_reference=all_refs
                else:
                    ref=_crop(imgs[axis],cfg.batch_size,tc.image_size,tc.phases)
                    frac=((ref+1)/2).mean((2,3))
                    generator_reference=ref
                z=torch.randn(cfg.batch_size,cfg.noise_channels,*([tc.image_size//16]*3),device=cfg.device)
                logits=g(z,generator_reference,frac)
                probabilities=logits.softmax(1)
                if cfg.categorical_straight_through:
                    hard=F.one_hot(probabilities.argmax(1),tc.phases).movedim(-1,1).float()
                    probabilities=hard.detach()-probabilities.detach()+probabilities
                volume=probabilities*2-1
                fake=_slices(volume,axis,cfg.slices_per_volume)
                c=frac.repeat_interleave(cfg.slices_per_volume,0)
                # Real images use their observed composition, not synthetic labels.
                cr=((real+1)/2).mean((2,3))
                od.zero_grad(set_to_none=True)
                dl=F.relu(1-d(real,cr,axis)).mean()+F.relu(1+d(fake.detach(),c,axis)).mean()
                dl.backward();od.step()
                for p in d.parameters():p.requires_grad_(False)
                og.zero_grad(set_to_none=True)
                t=torch.randint(max(1,tc.diffusion_steps//20),tc.diffusion_steps*9//10,(len(fake),),device=cfg.device)
                noise=torch.randn_like(fake)
                noisy=sched.add_noise(fake.detach(),noise,t)
                references=ref.repeat_interleave(cfg.slices_per_volume,0)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg.device.startswith('cuda')):
                    predicted=teacher(_input(noisy,references,c,axis),t).sample.float()
                alpha=sched.alphas_cumprod.to(cfg.device)[t,None,None,None]
                gradient=(1-alpha)*(predicted-noise)
                target=(fake-gradient).detach()
                sds=.5*F.mse_loss(fake,target)
                adv=-d(fake,c,axis).mean()
                fraction=((volume.add(1).div(2).mean((2,3,4))-frac)**2).mean()
                correlation=_correlation_loss(fake,real)
                loss=cfg.sds_weight*sds+cfg.adversarial_weight*adv+cfg.fraction_weight*fraction+cfg.correlation_weight*correlation
                loss.backward();torch.nn.utils.clip_grad_norm_(g.parameters(),1.);og.step()
                for p in d.parameters():p.requires_grad_(True)
                with torch.no_grad():
                    for a,b in zip(ema.parameters(),g.parameters()):a.lerp_(b,.01)
                elapsed=time.monotonic()-start
                log.write(json.dumps(dict(step=step,sds=float(sds.detach()),gan=float(adv.detach()),
                    fraction=float(fraction.detach()),correlation=float(correlation.detach()),elapsed_seconds=elapsed))+'\n');log.flush()
                if step%cfg.checkpoint_every == 0:save()
                if elapsed>=cfg.seconds:break
    finally:save()
    return run/'student.pt'

@torch.inference_mode()
def sample_student(checkpoint,fractions=None,seed=42,device='cuda',size=None,exact_fractions=False,paths=None,
                   composition_guidance=False,composition_temperature=.05):
    from .v1 import _fractions,_quota,_label_array
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=HybridConfig(**state['config'])
    size=size or state['size']
    if size%16 or size<16:raise ValueError('size must be a positive multiple of 16')
    torch.manual_seed(seed)
    tc=DiffusionConfig(phases=state['phases'],image_size=state['size'],device=device)
    imgs,labels=_images(paths or state['paths'],tc,state['labels'])
    reference=torch.stack([_crop(img,1,state['size'],state['phases']) for img in imgs],1) if cfg.tri_reference else _crop(imgs[0],1,state['size'],state['phases'])
    if fractions is None:fractions=[float(np.mean([(img==i).float().mean().item() for img in imgs])) for i in range(state['phases'])]
    frac=_fractions(fractions,state['phases'],device)[None]
    g=Generator(state['phases'],cfg.noise_channels,cfg.width,cfg.residual,cfg.tri_reference,cfg.normalize_reference,cfg.zero_reference).to(device).eval()
    g.load_state_dict(state['generator'])
    z=torch.randn(1,cfg.noise_channels,*([size//16]*3),device=device)
    output=g(z,reference,frac)
    diagnostic=None
    if composition_guidance:
        from .conditioning import composition_potential
        output,diagnostic=composition_potential(output,frac[0],temperature=composition_temperature)
    logits=output[0].cpu().numpy()
    phase=_quota(logits,np.asarray(fractions)) if exact_fractions else logits.argmax(0)
    return Volume(_label_array(labels)[phase],metadata={'method':'diffusion-guided adversarial student',
        'seed':seed,'requested_fractions':list(fractions),'quota_postprocessing':exact_fractions,
        'composition_guidance':composition_guidance,'composition_diagnostic':diagnostic})
