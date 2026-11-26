"""Declared finite-volume random media; statistics are measured, not assumed.

Binary directional S2 is estimated with the stated periodic convention. Connected
domain size is not called particle size when touching particles are unsegmented.
The spectral random-field generator matches phase counts exactly and reports its
remaining descriptor error. A single finite draw is not automatically an RVE.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy import fft,ndimage
from ..geometry import MicrostructureData

@dataclass(frozen=True)
class Distribution:
    kind:str='lognormal'
    parameters:dict=None
    def sample(self,rng,size):
        p=self.parameters or {}
        if self.kind=='constant':value=np.full(size,p['value'])
        elif self.kind=='normal':value=rng.normal(p['mean'],p['std'],size)
        elif self.kind=='lognormal':value=rng.lognormal(p['log_mean'],p['log_std'],size)
        elif self.kind=='uniform':value=rng.uniform(p['low'],p['high'],size)
        elif self.kind=='gamma':value=rng.gamma(p['shape'],p['scale'],size)
        elif self.kind=='empirical':value=rng.choice(p['values'],size=size,p=p.get('probabilities'))
        else:raise ValueError('Unknown probability distribution')
        if not np.isfinite(value).all():raise ValueError('Distribution produced nonfinite values')
        return value

@dataclass(frozen=True)
class StatisticsCard:
    fractions:dict
    correlation_lengths_m:dict
    phase_kinds:dict
    source:str
    cbd_surface_preference:float=.8
    periodic:bool=False
    def __post_init__(self):
        if not self.source.strip() or set(self.fractions)!=set(self.phase_kinds) or set(self.fractions)!=set(self.correlation_lengths_m):raise ValueError('Explicit matching phase/statistics definitions and source required')
        f=np.array(list(self.fractions.values()),float)
        if not np.isfinite(f).all() or np.min(f)<=0 or abs(f.sum()-1)>1e-12:raise ValueError('Positive phase probabilities must sum to one')
        if not 0<=self.cbd_surface_preference<=1:raise ValueError('CBD preference in [0,1] required')
        if any(np.asarray(v).shape!=(3,) or not np.isfinite(v).all() or min(v)<=0 for v in self.correlation_lengths_m.values()):raise ValueError('Positive physical ZYX correlation lengths required')
    def metadata(self):return asdict(self)
    @classmethod
    def from_dict(cls,data):
        return cls({int(k):v for k,v in data['fractions'].items()},{int(k):v for k,v in data['correlation_lengths_m'].items()},
            {int(k):v for k,v in data['phase_kinds'].items()},data['source'],data.get('cbd_surface_preference',.8),data.get('periodic',False))

def characterize(structure,*,lags=32):
    a=structure.labels;N=a.size;fractions={};curves={};lengths={};interfaces={}
    for phase in sorted(structure.phase_kinds):
        indicator=(a==phase).astype('float32');f=float(np.count_nonzero(a==phase)/N);fractions[phase]=f;curves[phase]=[];lengths[phase]=[];interfaces[phase]=0.
        for axis,n in enumerate(a.shape):
            spectrum=fft.rfft(indicator,axis=axis,workers=1);power=(spectrum.real**2+spectrum.imag**2).mean(axis=tuple(i for i in range(3) if i!=axis));del spectrum
            s2=fft.irfft(power,n=n)/n;count=min(lags,n//2)+1;curves[phase].append(s2[:count].astype(float).tolist())
            normalized=(s2-f*f)/max(f*(1-f),1e-12);cross=np.flatnonzero(normalized[:n//2+1]<=np.exp(-1))
            ell=float(cross[0]) if len(cross) else max(1.,n/2);lengths[phase].append(max(.5,ell)*structure.spacing_zyx_m[axis])
            interfaces[phase]+=float(np.mean(indicator!=np.roll(indicator,1,axis)))/structure.spacing_zyx_m[axis]
    card=StatisticsCard(fractions,lengths,structure.phase_kinds,'Measured complete supplied labels '+structure.fingerprint)
    return card,{'phase_fractions':fractions,'directional_S2':curves,'specific_interface_area_per_m':interfaces,
        'boundary':'periodic estimator; not a declaration of physical sample periodicity','scope':'phase statistics, not a unique reconstruction or resolved-particle size distribution'}

def connectivity(structure,channel_phases=None):
    labels=structure.labels;result={}
    for channel,kinds in [('ionic',{'pore','cbd'}),('electronic',{'active','cbd','inert'})]:
        phases=[p for p,k in structure.phase_kinds.items() if k in kinds] if channel_phases is None else channel_phases[channel]
        if not set(phases)<=set(structure.phase_kinds):raise ValueError('Transport channel contains an undefined material phase')
        component,count=ndimage.label(np.isin(labels,phases))
        left=set(np.unique(component[:,:,0]))-{0};right=set(np.unique(component[:,:,-1]))-{0};result[channel]=bool(left&right)
    return result

def generate(card,shape=(16,16,16),spacing_m=(1e-6,)*3,*,seed=104,maximum_attempts=24,require_percolation=True,channel_phases=None):
    shape=tuple(map(int,shape));spacing=np.asarray(spacing_m,float)
    if len(shape)!=3 or min(shape)<2 or np.prod(shape)>16_777_216 or spacing.shape!=(3,) or min(spacing)<=0:raise ValueError('Bounded 3D grid and positive SI spacing required')
    phases=sorted(card.fractions);active=[p for p in phases if card.phase_kinds[p]=='active'];cbd=[p for p in phases if card.phase_kinds[p]=='cbd']
    counts=np.floor(np.array([card.fractions[p] for p in phases])*np.prod(shape)).astype(int)
    remainder=int(np.prod(shape)-counts.sum());residual=np.array([card.fractions[p] for p in phases])*np.prod(shape)-counts
    counts[np.argsort(-residual,kind='stable')[:remainder]]+=1;number=dict(zip(phases,counts))
    if np.min(counts)<=0:raise ValueError('Requested grid cannot represent every nonzero phase probability')
    for attempt in range(maximum_attempts):
        rng=np.random.default_rng(np.random.SeedSequence([seed,attempt]));out=np.full(shape,-1,dtype='int32');fields={}
        for phase in phases:
            sigma=np.asarray(card.correlation_lengths_m[phase])/spacing/2
            field=ndimage.gaussian_filter(rng.standard_normal(shape),sigma,mode='wrap' if card.periodic else 'reflect');field=(field-field.mean())/max(field.std(),1e-12);fields[phase]=field
        order=[*active,*cbd,*[p for p in phases if p not in active+cbd]]
        for phase in order[:-1]:
            score=fields[phase].copy()
            if phase in cbd and active:
                distance=ndimage.distance_transform_edt(~np.isin(out,active),sampling=spacing)
                score+=4*card.cbd_surface_preference*np.exp(-distance/max(min(card.correlation_lengths_m[phase]),min(spacing)))
            available=np.flatnonzero(out.ravel()<0);selected=available[np.argsort(score.ravel()[available],kind='stable')[-number[phase]:]];out.ravel()[selected]=phase
        out[out<0]=order[-1]
        data=MicrostructureData(out,tuple(spacing),(0.,)*3,card.phase_kinds,{'provider':'MORQEX statistical Gaussian media','statistics':card.metadata(),'seed':seed,'attempt':attempt,
            'source_discretization':'new stochastic draw, not subdivision of a small prior geometry','scope':'exact phase counts; empirical descriptor and connectivity checks, not automatic statistical equivalence or fabrication validity'})
        channels=connectivity(data,channel_phases)
        if not require_percolation or all(channels.values()):
            _,measured=characterize(data,lags=min(16,min(shape)//2));data.provenance.update(measured_statistics=measured,connectivity=channels)
            return data
    raise ValueError('No ionically/electronically percolating draw in the specified seed/attempt budget; change the statistical model or domain size')
