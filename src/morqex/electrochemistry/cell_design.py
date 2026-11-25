"""Explicit geometry-aware capacity and C-rate contracts.

Changing active volume fractions while retaining a nameplate current changes
the physical C-rate. This helper reports both and only rebalances a cell when
the caller explicitly supplies a design policy and usable stoichiometry windows.
"""
from dataclasses import dataclass
import numpy as np

FARADAY=96485.33212

@dataclass(frozen=True)
class CellDesign:
    negative_window:tuple
    positive_window:tuple
    policy:str='preserve'
    negative_to_positive_ratio:float=1.1
    def __post_init__(self):
        if self.policy not in {'preserve','balance-negative-thickness'}:raise ValueError('Explicit preserve or balance-negative-thickness policy required')
        for window in [self.negative_window,self.positive_window]:
            if len(window)!=2 or not np.isfinite(window).all() or not 0<window[0]<window[1]<1:raise ValueError('Usable stoichiometry windows must be ordered within (0,1)')
        if not np.isfinite(self.negative_to_positive_ratio) or self.negative_to_positive_ratio<=0:raise ValueError('Positive N/P capacity ratio required')
    def apply(self,parameters,*,active_fractions=None):
        p=parameters.copy();before={};changes={}
        for d,fraction in (active_fractions or {}).items():
            if d not in {'negative','positive'} or not 0<float(fraction)<=1:raise ValueError('Positive intrinsic active volume fraction required')
            name=f'{d.capitalize()} electrode active material volume fraction';before[name]=float(p[name]);changes[name]=float(fraction)
        p.update(changes)
        area=float(p['Electrode width [m]'])*float(p['Electrode height [m]'])*float(p['Number of electrodes connected in parallel to make a cell'])
        def capacity(d):
            lo,hi=getattr(self,d+'_window');D=d.capitalize()
            return area*float(p[f'{D} electrode thickness [m]'])*float(p[f'{D} electrode active material volume fraction'])*float(p[f'Maximum concentration in {d} electrode [mol.m-3]'])*(hi-lo)*FARADAY/3600
        original_nominal=float(p['Nominal cell capacity [A.h]']);old_ratio=capacity('negative')/capacity('positive')
        if self.policy=='balance-negative-thickness':
            name='Negative electrode thickness [m]';before[name]=float(p[name]);changes[name]=float(p[name])*self.negative_to_positive_ratio/old_ratio;p.update({name:changes[name]})
            # Nominal capacity here has an explicit design meaning. Preserve
            # leaves the user nameplate unchanged and reports its discrepancy.
            before['Nominal cell capacity [A.h]']=original_nominal;changes['Nominal cell capacity [A.h]']=min(capacity('negative'),capacity('positive'));p.update({'Nominal cell capacity [A.h]':changes['Nominal cell capacity [A.h]']})
        capacities={d:capacity(d) for d in ['negative','positive']};limiting=min(capacities.values())
        return p,{'policy':self.policy,'usable_windows':{'negative':list(self.negative_window),'positive':list(self.positive_window)},'electrode_usable_capacity_Ah':capacities,
            'N_to_P_usable_capacity_ratio':capacities['negative']/capacities['positive'],'nameplate_capacity_Ah':float(p['Nominal cell capacity [A.h]']),
            'current_for_0_1C_nameplate_A':.1*float(p['Nominal cell capacity [A.h]']),
            'physical_C_rate_at_0_1C_nameplate':.1*float(p['Nominal cell capacity [A.h]'])/limiting,
            'before':before,'changes':changes,'scope':'Inventory/window capacity audit, not a prediction of rate-limited delivered capacity or automatic SOC initialization'}
