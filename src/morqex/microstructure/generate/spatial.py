"""Few-step volumetric diffusion learned from 2D cross sections only.

The network predicts a clean 3D categorical field. Exact Gaussian posterior
coefficients advance a short diffusion chain; a shared, time/plane-aware 2D
discriminator supervises pairs of corresponding slices. There is no real 3D
training target and no pretrained teacher. This implementation follows the
published adversarial diffusion dimensional-expansion idea, with independent
GEM network/training code and explicit multiphase statistical regularization.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import copy,json,time,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from morqex.microstructure import Volume
from .v1 import DiffusionConfig,_images,_crop,_atomic_save,_label_array
from .hybrid import _correlation_loss


@dataclass(frozen=True)
class VoxelDiffusionConfig:
    size:int=64
    phases:int=3
    width:int=32
    batch_size:int=8
    slices_per_axis:int=3
    diffusion_steps:int=2
    steps:int=100000
    seconds:float=3300.
    learning_rate:float=2e-4
    fraction_weight:float=5.
    correlation_weight:float=1.
    interface_weight:float=.1
    entropy_weight:float=.02
    axial_attention:bool=True
    periodic_padding:bool=True
    checkpoint_every:int=100
    seed:int=42
    device:str='cuda'

    def __post_init__(self):
        if self.size<16 or self.size%4 or self.width%8 or self.width<8:
            raise ValueError('size >=16 divisible by4, width >=8 divisible by8 required')
        if min(self.batch_size,self.slices_per_axis,self.steps,self.diffusion_steps,self.checkpoint_every)<1 or self.seconds<=0:
            raise ValueError('Positive training settings required')


class Residual(nn.Module):
    def __init__(self,cin,cout,padding_mode='circular'):
        super().__init__()
        self.norm1=nn.GroupNorm(8,cin);self.norm2=nn.GroupNorm(8,cout,affine=False)
        self.conv1=nn.Conv3d(cin,cout,3,padding=1,padding_mode=padding_mode);self.conv2=nn.Conv3d(cout,cout,3,padding=1,padding_mode=padding_mode)
        self.affine=nn.Linear(128,2*cout)
        self.skip=nn.Identity() if cin==cout else nn.Conv3d(cin,cout,1)
    def forward(self,x,embedding):
        h=self.conv1(F.silu(self.norm1(x)))
        scale,bias=self.affine(embedding).chunk(2,1)
        h=self.norm2(h)*(1+scale[:,:,None,None,None])+bias[:,:,None,None,None]
        return (self.skip(x)+self.conv2(F.silu(h)))/2**.5


class Denoiser(nn.Module):
    def __init__(self,cfg):
        super().__init__();w=cfg.width;pad='circular' if cfg.periodic_padding else 'zeros'
        self.time=nn.Embedding(cfg.diffusion_steps,128)
        self.latent=nn.Linear(16,128)
        self.stem=nn.Conv3d(cfg.phases,w,3,padding=1,padding_mode=pad)
        self.low=Residual(w,w,pad);self.down1=nn.Conv3d(w,2*w,4,2,1,padding_mode=pad)
        self.middle=Residual(2*w,2*w,pad);self.down2=nn.Conv3d(2*w,3*w,4,2,1,padding_mode=pad)
        self.deep=Residual(3*w,3*w,pad);self.deep2=Residual(3*w,3*w,pad)
        self.attention=AxialAttention(3*w) if cfg.axial_attention else nn.Identity()
        self.up1=Residual(5*w,2*w,pad);self.up2=Residual(3*w,w,pad)
        self.out=nn.Conv3d(w,cfg.phases,1)
    def forward(self,x,t,z,temperature=1.):
        e=F.silu(self.time(t)+self.latent(z))
        a=self.low(self.stem(x),e);b=self.middle(self.down1(a),e)
        h=self.attention(self.deep2(self.deep(self.down2(b),e),e))
        h=self.up1(torch.cat((F.interpolate(h,size=b.shape[2:],mode='trilinear',align_corners=False),b),1),e)
        h=self.up2(torch.cat((F.interpolate(h,size=a.shape[2:],mode='trilinear',align_corners=False),a),1),e)
        return self.out(F.silu(h)).float().div(temperature).softmax(1)*2-1


class AxialAttention(nn.Module):
    """Global information exchange along three axes at the coarse grid only."""
    def __init__(self,channels):
        super().__init__();self.norm=nn.GroupNorm(8,channels)
        self.attention=nn.MultiheadAttention(channels,8,batch_first=True)
    def forward(self,x):
        for axis in (2,3,4):
            order=[0]+[d for d in (2,3,4) if d!=axis]+[axis,1]
            h=self.norm(x).permute(order);shape=h.shape
            sequence=h.reshape(-1,shape[-2],shape[-1])
            attended=self.attention(sequence,sequence,sequence,need_weights=False)[0]
            inverse=[order.index(i) for i in range(5)]
            x=x+.1*attended.reshape(shape).permute(inverse)
        return x


class SliceCritic(nn.Module):
    def __init__(self,cfg):
        super().__init__();sn=nn.utils.parametrizations.spectral_norm
        layers=[]
        for a,b in zip((cfg.phases*2+4,32,64,128),(32,64,128,128)):
            layers.extend((sn(nn.Conv2d(a,b,3,2,1)),nn.LeakyReLU(.2)))
        layers.append(sn(nn.Conv2d(128,1,3,padding=1)));self.net=nn.Sequential(*layers)
        self.steps=cfg.diffusion_steps
    def forward(self,current,previous,t,axis):
        b,_,h,w=current.shape
        condition=current.new_zeros((b,4,h,w));condition[:,axis]=1
        condition[:,3]=float(t)/self.steps
        return self.net(torch.cat((current,previous,condition),1))


class Chain:
    def __init__(self,steps,device):
        t=torch.linspace(0,1,steps+1,device=device)
        bar=torch.cos((t+.008)/1.008*torch.pi/2).square()
        beta=(1-bar[1:]/bar[:-1]).clamp(1e-5,.999)
        alpha=1-beta;self.bar=alpha.cumprod(0)
        previous=torch.cat((self.bar.new_ones(1),self.bar[:-1]))
        self.beta=beta;self.alpha=alpha
        self.coef1=beta*previous.sqrt()/(1-self.bar)
        self.coef2=(1-previous)*alpha.sqrt()/(1-self.bar)
        self.variance=beta*(1-previous)/(1-self.bar)
    def real_pair(self,clean,t):
        if t==0:current=clean
        else:current=self.bar[t-1].sqrt()*clean+(1-self.bar[t-1]).sqrt()*torch.randn_like(clean)
        previous=self.alpha[t].sqrt()*current+self.beta[t].sqrt()*torch.randn_like(current)
        return current,previous
    def reverse(self,clean,previous,t):
        x=self.coef1[t]*clean+self.coef2[t]*previous
        if t:x=x+self.variance[t].sqrt()*torch.randn_like(x)
        return x


def _planes(x,axis,indices):
    return x.index_select(axis+2,indices).movedim(axis+2,1).flatten(0,1)


def _interface(x):
    return torch.stack([(x.diff(dim=i).abs()/2).mean() for i in range(2,x.ndim)]).mean()


def fit_volume(images,run_dir,config=None,resume=None):
    """Train a new 3D diffusion model from one/three 2D phase-label images."""
    cfg=config or VoxelDiffusionConfig();torch.manual_seed(cfg.seed)
    run=Path(run_dir).resolve();package=Path(__file__).resolve().parents[4]
    if run==package or package in run.parents:raise ValueError('Use an external run directory')
    run.mkdir(parents=True,exist_ok=True)
    tensors,labels=_images(images,DiffusionConfig(image_size=cfg.size,phases=cfg.phases,device=cfg.device))
    net=Denoiser(cfg).to(cfg.device);ema=copy.deepcopy(net).eval().requires_grad_(False)
    critic=SliceCritic(cfg).to(cfg.device)
    optim=torch.optim.AdamW(net.parameters(),lr=cfg.learning_rate,betas=(.5,.99))
    disc_optim=torch.optim.AdamW(critic.parameters(),lr=cfg.learning_rate,betas=(.5,.99))
    chain=Chain(cfg.diffusion_steps,cfg.device)
    amp=str(cfg.device).startswith('cuda');step=0;begin=0;prior_seconds=0.
    if resume:
        state=torch.load(resume,map_location=cfg.device,weights_only=False)
        for key in ('size','phases','width','diffusion_steps','axial_attention'):
            if state['config'].get(key,False)!=asdict(cfg)[key]:raise ValueError(f'Resume mismatch: {key}')
        if state['labels']!=labels:raise ValueError('Resume phase labels differ')
        net.load_state_dict(state['network']);ema.load_state_dict(state['ema']);critic.load_state_dict(state['critic'])
        optim.load_state_dict(state['optimizer']);disc_optim.load_state_dict(state['critic_optimizer'])
        torch.set_rng_state(state['rng'].cpu())
        if amp:torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
        begin=step=state['step'];prior_seconds=state.get('cumulative_seconds',state['elapsed_seconds'])
    started=time.monotonic()
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fractions=torch.stack([(x==i).float().mean() for i in range(cfg.phases) for x in tensors]).reshape(cfg.phases,3).mean(1)
    (run/'config.json').write_text(json.dumps(asdict(cfg),indent=2))
    def save():
        _atomic_save(dict(network=net.state_dict(),ema=ema.state_dict(),critic=critic.state_dict(),
            optimizer=optim.state_dict(),critic_optimizer=disc_optim.state_dict(),config=asdict(cfg),
            step=step,labels=labels,paths=[str(Path(p).resolve()) for p in images],
            elapsed_seconds=time.monotonic()-started,cumulative_seconds=prior_seconds+time.monotonic()-started,
            resumed_from=str(resume) if resume else None,rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if amp else None,source_sha256=source_sha,
            method='volumetric adversarial diffusion',real_3d_training=False),run/'model.pt')
    if amp:torch.cuda.reset_peak_memory_stats()
    try:
        with (run/'metrics.jsonl').open('a') as log:
            for step in range(begin+1,cfg.steps+1):
                x=torch.randn(cfg.batch_size,cfg.phases,*([cfg.size]*3),device=cfg.device)
                z=torch.randn(cfg.batch_size,16,device=cfg.device)
                statistics={k:0. for k in ('gan','fraction','correlation','interface','entropy','discriminator')}
                for t in reversed(range(cfg.diffusion_steps)):
                    timestep=torch.full((cfg.batch_size,),t,device=cfg.device,dtype=torch.long)
                    temperature=max(.35,1.-.65*min(1,step/2500))
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                        clean=net(x,timestep,z,temperature)
                        next_x=chain.reverse(clean,x,t)
                    real=[];pairs=[];fake=[];old=[];clean_slices=[]
                    indices=torch.randint(cfg.size,(cfg.slices_per_axis,),device=cfg.device)
                    for axis in range(3):
                        r=_crop(tensors[axis],cfg.batch_size*cfg.slices_per_axis,cfg.size,cfg.phases)
                        a,b=chain.real_pair(r,t);real.append(r);pairs.append((a,b))
                        fake.append(_planes(next_x,axis,indices));old.append(_planes(x,axis,indices))
                        clean_slices.append(_planes(clean,axis,indices))
                    disc_optim.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                        dl=sum(F.softplus(-critic(a,b,t,axis)).mean()+F.softplus(critic(fake[axis].detach(),old[axis],t,axis)).mean()
                               for axis,(a,b) in enumerate(pairs))/3
                    dl.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),5.);disc_optim.step()
                    critic.requires_grad_(False);optim.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                        adversarial=sum(F.softplus(-critic(fake[axis],old[axis],t,axis)).mean() for axis in range(3))/3
                        f=((clean+1)/2).mean((2,3,4))
                        fraction=(f-fractions).square().mean()
                        correlation=sum(_correlation_loss(clean_slices[i],real[i]) for i in range(3))/3
                        interface=(_interface(clean)-torch.stack([_interface(r) for r in real]).mean()).square()
                        p=((clean+1)/2).clamp_min(1e-7);entropy=-(p*p.log()).sum(1).mean()
                        loss=adversarial+cfg.fraction_weight*fraction+cfg.correlation_weight*correlation+cfg.interface_weight*interface+cfg.entropy_weight*entropy
                    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optim.step()
                    critic.requires_grad_(True)
                    x=next_x.detach()
                    with torch.no_grad():
                        for a,b in zip(ema.parameters(),net.parameters()):a.lerp_(b,.005)
                    for key,value in zip(statistics,(adversarial,fraction,correlation,interface,entropy,dl)):
                        statistics[key]+=float(value.detach())/cfg.diffusion_steps
                elapsed=time.monotonic()-started
                record=dict(step=step,elapsed_seconds=elapsed,**statistics,
                    peak_memory_bytes=torch.cuda.max_memory_allocated() if amp else 0,
                    voxels_per_second=(step-begin)*cfg.batch_size*cfg.size**3/elapsed)
                log.write(json.dumps(record)+'\n');log.flush()
                if step%cfg.checkpoint_every==0:save()
                if elapsed>=cfg.seconds:break
    finally:save()
    return run/'model.pt'


@torch.inference_mode()
def draw_volume(checkpoint,size=256,seed=42,device='cuda',spacing=(1.,1.,1.)):
    """Native-size 3D diffusion chain; no numerical property input/postprojection."""
    if size<16 or size%4:raise ValueError('size >=16 and divisible by4 required')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=VoxelDiffusionConfig(**{**state['config'],'axial_attention':state['config'].get('axial_attention',False),
        'periodic_padding':state['config'].get('periodic_padding',False),'device':device})
    torch.manual_seed(seed);net=Denoiser(cfg).to(device).eval();net.load_state_dict(state['ema'])
    chain=Chain(cfg.diffusion_steps,device)
    x=torch.randn(1,cfg.phases,size,size,size,device=device);z=torch.randn(1,16,device=device)
    temperature=max(.35,1.-.65*min(1,state['step']/2500))
    for t in reversed(range(cfg.diffusion_steps)):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
            clean=net(x,torch.tensor([t],device=device),z,temperature)
            x=chain.reverse(clean,x,t)
    labels=_label_array(state['labels'])[x[0].argmax(0).cpu().numpy()]
    return Volume(labels,spacing=spacing,metadata=dict(version=1,method='volumetric adversarial diffusion',
        seed=seed,steps=cfg.diffusion_steps,training_step=state['step'],real_3d_training=False,
        numerical_property_condition=False,quota_postprocessing=False,
        training_fractions=state.get('training_fractions')))
