"""Component-conservative positive partition-of-unity concentration nodes."""
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class ComponentBasis:
    orders:np.ndarray
    offsets:np.ndarray
    feature_kind:str='depth-local_surface-direction'
    @property
    def dimension(self):return int(self.offsets[-1])
    @classmethod
    def allocate(cls,components,features,*,target):
        count=int(np.max(components));counts=np.bincount(components.ravel(),minlength=count+1)
        if target<count:raise ValueError(f'{count} independent connected inventories require at least {count} states; eight universal states cannot preserve arbitrary particle inventories')
        orders=np.ones((count+1,3),np.int32);sums=np.array([np.bincount(components.ravel(),weights=f.ravel(),minlength=count+1) for f in features])
        squares=np.array([np.bincount(components.ravel(),weights=(f*f).ravel(),minlength=count+1) for f in features])
        variance=np.maximum(0,squares/np.maximum(counts,1)-(sums/np.maximum(counts,1))**2)
        dimension=count
        while dimension<target:
            candidates=[]
            for component in range(1,count+1):
                size=int(np.prod(orders[component]))
                for axis in range(3):
                    extra=size//int(orders[component,axis])
                    if dimension+extra<=target and size+extra<=counts[component] and variance[axis,component]>1e-10 and orders[component,axis]<5:
                        score=counts[component]*variance[axis,component]/(orders[component,axis]**2*extra)
                        candidates.append((score,component,axis,extra))
            if not candidates:break
            _,component,axis,extra=max(candidates);orders[component,axis]+=1;dimension+=extra
        sizes=np.prod(orders[1:],axis=1);offsets=np.r_[0,np.cumsum(sizes)].astype(np.int32)
        return cls(orders,offsets)
    def evaluate(self,components,features,*,xp=np):
        components=xp.asarray(components);features=xp.asarray(features);out=xp.zeros((components.size,self.dimension),dtype='float64')
        for component in range(1,len(self.orders)):
            selected=components.ravel()==component
            if not bool(xp.any(selected)):continue
            axes=[]
            for axis,order in enumerate(self.orders[component]):
                degree=int(order)-1;f=features[axis].ravel()[selected].astype('float64')
                axes.append([math.comb(degree,j)*f**j*(1-f)**(degree-j) for j in range(degree+1)])
            local=xp.stack([a*b*c for a in axes[0] for b in axes[1] for c in axes[2]],axis=1)
            out[selected,self.offsets[component-1]:self.offsets[component]]=local
        return out
    def metadata(self):return {'orders':self.orders.tolist(),'offsets':self.offsets.tolist(),'dimension':self.dimension,'feature_kind':self.feature_kind,
        'inventory':'one partition of unity per connected active component','positivity':'nonnegative basis; node bounds imply fine-field concentration bounds'}
    @classmethod
    def from_metadata(cls,data):return cls(np.array(data['orders'],np.int32),np.array(data['offsets'],np.int32),data['feature_kind'])

def full_features(arrays):
    features=arrays['features'].copy();features[1]=arrays['faces']/6.;return features
