"""Independent Ai--Wu--Martinez-Paneda (JPS 2022, 231805) equation adapter.

Small-strain hybrid elasticity, stress-assisted diffusion, history-field AT2,
and positive-variation fatigue, assembled as one nonlinear system. This is NOT
a statement of paper reproduction: geometry, discretization, time convergence,
and published observables must also pass. No author COMSOL code is copied.

Dimensionless coordinates x/R, displacement u/R, concentration c/cmax,
time t D/R^2, energy density psi/E. Scalar FE and quadrature are scikit-fem;
precomputed sparse evaluation operators avoid reinterpreting forms per Newton
iteration. All history trials originate from the accepted state.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, bmat
from scipy.sparse.linalg import spsolve, splu
from .backend import require_local_budget


@dataclass(frozen=True)
class ParticleMaterial:
    young_Pa: float = 93e9
    poisson: float = .3
    diffusivity_m2_s: float = 7.08e-15
    molar_volume_m3_mol: float = 3.497e-6
    cmax_mol_m3: float = 22900.
    toughness_J_m2: float = 10.
    temperature_K: float = 298.
    radius_m: float = 5e-6
    length_m: float = 1e-8
    residual_stiffness: float = 1e-5
    reference_soc: float = 0.

    def __post_init__(self):
        values=list(self.__dict__.values())
        if not np.isfinite(values).all(): raise ValueError('Finite material parameters required')
        if any(v<=0 for k,v in self.__dict__.items() if k not in ('poisson','reference_soc')):
            raise ValueError('Positive SI material parameters required')
        if not -1<self.poisson<.5 or not 0<=self.reference_soc<=1 or not 0<self.residual_stiffness<1:
            raise ValueError('Invalid material bounds')


@dataclass(frozen=True)
class ParticleState:
    u: np.ndarray
    c: np.ndarray
    d: np.ndarray
    history: np.ndarray
    fatigue: np.ndarray
    previous_alpha: np.ndarray
    time_s: float

    def __post_init__(self):
        for name in ('u','c','d','history','fatigue','previous_alpha'):
            a=np.array(getattr(self,name),dtype=float,copy=True)
            if a.ndim!=1 or not np.isfinite(a).all(): raise ValueError('Finite vector state required')
            a.flags.writeable=False;object.__setattr__(self,name,a)
        if not np.isfinite(self.time_s) or self.time_s<0:raise ValueError('Invalid accepted time')


def _left(v,A):
    return A.multiply(np.asarray(v)[:,None]).tocsr()


class PaperParticle:
    def __init__(self, mesh, material=ParticleMaterial(), *, geometry='sphere', order=1,
                 fatigue=True, stress_feedback=True, quadrature_order=4, damage_constraint='irreversible_vi', linearization='newton', boundary_policy='fixed_flux', concentration_domain='bounded', phase_form='paper_divergence', toughness_floor=0., seed_width_over_length=1.):
        require_local_budget()
        from skfem import Basis, ElementTriP1, ElementTriP2, FacetBasis
        if geometry not in ('sphere','cylinder') or order not in (1,2):raise ValueError('Quarter sphere/cylinder, P1/P2 required')
        self.material=material;self.mesh=mesh;self.geometry=geometry
        if damage_constraint not in ('history','irreversible_vi'):raise ValueError('Unknown damage constraint')
        self.damage_constraint=damage_constraint
        if linearization not in ('newton','chord'):raise ValueError('Unknown nonlinear linearization')
        self.linearization=linearization;self._linear_cache=None
        if boundary_policy not in ('fixed_flux','surface_limits'):raise ValueError('Unknown concentration boundary policy')
        self.boundary_policy=boundary_policy
        if concentration_domain not in ('bounded','dilute_diagnostic'):raise ValueError('Unknown concentration domain')
        self.concentration_domain=concentration_domain
        if phase_form not in ('paper_divergence','author_helmholtz'):raise ValueError('Unknown phase equation')
        if not 0<=toughness_floor<1 or not np.isfinite(seed_width_over_length) or seed_width_over_length<=0:raise ValueError('Invalid source-model controls')
        self.phase_form=phase_form;self.toughness_floor=float(toughness_floor);self.seed_width_over_length=float(seed_width_over_length)
        self.fatigue_enabled=bool(fatigue);self.stress_feedback=bool(stress_feedback)
        element=ElementTriP1() if order==1 else ElementTriP2()
        basis=Basis(mesh,element,intorder=quadrature_order);self.basis=basis
        self.n=int(basis.N);self.nq=int(basis.dx.size)
        shape=basis.dx.shape
        rows=np.arange(self.nq).reshape(shape)
        def evaluation(values):
            rr=[];cc=[];vv=[]
            for i,a in enumerate(values):
                rr.append(rows.ravel());cc.append(np.broadcast_to(basis.element_dofs[i,:,None],shape).ravel());vv.append(np.asarray(a).ravel())
            return coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(self.nq,self.n)).tocsr()
        self.Q=evaluation([v[0].value for v in basis.basis])
        self.G=[evaluation([v[0].grad[i] for v in basis.basis]) for i in range(2)]
        x=basis.global_coordinates().reshape(2,-1)
        weight=2*np.pi*x[0] if geometry=='sphere' else np.ones(self.nq)
        self.w=basis.dx.ravel()*weight;self.volume=float(self.w.sum())
        self.lump=np.asarray(self.Q.T@self.w)
        # P2 vertex basis has zero integral in 2D. Use the consistent L2
        # projection for stress, avoiding an invalid lumped P2 projector.
        if order!=1:raise ValueError('P2 stress projection not yet qualified; use P1 explicitly')
        self.projection=_left(1/self.lump,self.Q.T@diags(self.w))
        self.M=(self.Q.T@_left(self.w,self.Q)).tocsr()
        self.K=sum(G.T@_left(self.w,G) for G in self.G).tocsr()
        selector=[coo_matrix((np.ones(self.n),(np.arange(self.n),2*np.arange(self.n)+j)),shape=(self.n,2*self.n)).tocsr() for j in range(2)]
        self.B=[self.G[0]@selector[0],self.G[1]@selector[1],
                (_left(1/x[0],self.Q)@selector[0] if geometry=='sphere' else csr_matrix((self.nq,2*self.n))),
                (self.G[1]@selector[0]+self.G[0]@selector[1])/np.sqrt(2)]
        self.L=sum(self.B[:3]).tocsr()
        fixed=np.r_[2*np.flatnonzero(np.abs(basis.doflocs[0])<1e-12),2*np.flatnonzero(np.abs(basis.doflocs[1])<1e-12)+1]
        self.free_u=np.setdiff1d(np.arange(2*self.n),fixed)
        self.free=np.r_[self.free_u,2*self.n+np.arange(self.n),3*self.n+np.arange(self.n)]
        # Only the curved outer surface exchanges lithium. Symmetry faces do not.
        facets=mesh.boundary_facets();mid=mesh.p[:,mesh.facets[:,facets]].mean(axis=1)
        facets=facets[(mid[0]>1e-12)&(mid[1]>1e-12)]
        fb=FacetBasis(mesh,element,facets=facets,intorder=quadrature_order)
        fw=fb.dx*(2*np.pi*fb.global_coordinates()[0] if geometry=='sphere' else 1)
        self.surface_load=np.zeros(self.n)
        for i,b in enumerate(fb.basis):
            np.add.at(self.surface_load,fb.element_dofs[i],np.sum(fw*b[0].value,axis=1))
        self.surface_area=float(self.surface_load.sum())
        self.surface_nodes=self.surface_load>0
        m=material;self.mu=1/(2*(1+m.poisson));self.lam=m.poisson/((1+m.poisson)*(1-2*m.poisson));self.bulk=1/(3*(1-2*m.poisson))
        self.beta=m.molar_volume_m3_mol*m.cmax_mol_m3/3
        self.chi=m.molar_volume_m3_mol*m.young_Pa/(8.314462618*m.temperature_K) if stress_feedback else 0.
        self.ell=m.length_m/m.radius_m;self.gc=m.toughness_J_m2/(m.young_Pa*m.radius_m)
        self.threshold=self.gc/(12*self.ell);self.time_scale=m.radius_m**2/m.diffusivity_m2_s
        self.quadrature_coordinates=x

    def fields(self,u,c,d,old):
        cq=self.Q@c;dq=self.Q@d
        e=np.array([B@u for B in self.B]);e[:3]-=self.beta*(cq-self.material.reference_soc)
        tr=e[:3].sum(axis=0);dev=e.copy();dev[:3]-=tr/3
        sigma=2*self.mu*e;sigma[:3]+=self.lam*tr
        plus=.5*self.bulk*np.maximum(tr,0)**2+self.mu*np.sum(dev**2,axis=0)
        sp=2*self.mu*dev;sp[:3]+=self.bulk*np.maximum(tr,0)
        g=(1-dq)**2+self.material.residual_stiffness;gp=-2*(1-dq)
        alpha=g*plus
        history=np.maximum(old.history,plus)
        fatigue=old.fatigue+np.maximum(alpha-old.previous_alpha,0) if self.fatigue_enabled else old.fatigue
        ratio=fatigue/self.threshold
        factor=np.where(ratio<=1,1.,4/(1+ratio)**2) if self.fatigue_enabled else np.ones_like(ratio)
        derivative=np.where(ratio>1,-8/self.threshold/(1+ratio)**3,0.) if self.fatigue_enabled else np.zeros_like(ratio)
        derivative=np.where(factor>self.toughness_floor,derivative,0.);factor=np.maximum(factor,self.toughness_floor)
        sh=self.projection@(g*self.bulk*tr)
        return dict(cq=cq,dq=dq,tr=tr,sigma=sigma,plus=plus,sp=sp,g=g,gp=gp,alpha=alpha,
                    history=history,fatigue=fatigue,factor=factor,df=derivative,sh=sh)

    def residual(self,x,old,dt_s,outward_flux_mol_m2_s,*,jacobian=True):
        n=self.n;u=x[:2*n];c=x[2*n:3*n];d=x[3*n:];f=self.fields(u,c,d,old)
        w=self.w;Q=self.Q;G=self.G;B=self.B;g=f['g'];gp=f['gp'];H=f['history'];fac=f['factor'];dq=f['dq']
        dt=dt_s/self.time_scale
        flux=outward_flux_mol_m2_s*self.material.radius_m/(self.material.diffusivity_m2_s*self.material.cmax_mol_m3)
        ru=sum(Bi.T@(w*g*si) for Bi,si in zip(B,f['sigma']))
        rc=self.M@(c-old.c)/dt+self.K@c+flux*self.surface_load
        rc-=self.chi*sum(Gi.T@(w*f['cq']*(Gi@f['sh'])) for Gi in G)
        rd=self.gc*(sum(Gi.T@(w*fac*self.ell*(Gi@d)) for Gi in G)+Q.T@(w*fac*dq/self.ell))-2*Q.T@(w*(1-dq)*H)
        if self.phase_form=='author_helmholtz':
            rd=self.ell**2*(self.K@d)+self.M@d-2*Q.T@(w*(1-dq)*H*self.ell/(self.gc*fac))
        r=np.r_[ru,rc,rd]
        if not jacobian:return r,f
        Juu=sum(Bi.T@_left(w*g*2*self.mu,Bi) for Bi in B)+self.L.T@_left(w*g*self.lam,self.L)
        Juc=-self.L.T@_left(w*g*3*self.bulk*self.beta,Q)
        Jud=sum(Bi.T@_left(w*gp*si,Q) for Bi,si in zip(B,f['sigma']))
        Lc=sum(Gi.T@_left(w*f['cq'],Gi) for Gi in G)
        drift=sum(Gi.T@_left(w*(Gi@f['sh']),Q) for Gi in G)
        Su=self.projection@_left(g*self.bulk,self.L)
        Sc=self.projection@_left(-g*3*self.bulk*self.beta,Q)
        Sd=self.projection@_left(gp*self.bulk*f['tr'],Q)
        Jcu=-self.chi*(Lc@Su);Jcc=self.M/dt+self.K-self.chi*(drift+Lc@Sc);Jcd=-self.chi*(Lc@Sd)
        psiu=sum(_left(si,Bi) for Bi,si in zip(B,f['sp']))
        psic=_left(-3*self.beta*self.bulk*np.maximum(f['tr'],0),Q)
        activeH=(f['plus']>old.history).astype(float)
        activeA=(f['alpha']>old.previous_alpha).astype(float)
        Fu=_left(f['df']*activeA*g,psiu);Fc=_left(f['df']*activeA*g,psic)
        Fd=_left(f['df']*activeA*gp*f['plus'],Q)
        Af=self.gc*(sum(Gi.T@diags(w*self.ell*(Gi@d)) for Gi in G)+Q.T@diags(w*dq/self.ell))
        AH=-2*Q.T@diags(w*(1-dq)*activeH)
        Jdu=Af@Fu+AH@psiu;Jdc=Af@Fc+AH@psic
        Jdd=self.gc*(sum(Gi.T@_left(w*fac*self.ell,Gi) for Gi in G)+Q.T@_left(w*fac/self.ell,Q))+2*Q.T@_left(w*H,Q)+Af@Fd
        if self.phase_form=='author_helmholtz':
            Af=2*Q.T@diags(w*(1-dq)*H*self.ell/(self.gc*fac**2))
            AH=-2*Q.T@diags(w*(1-dq)*self.ell/(self.gc*fac)*activeH)
            Jdu=Af@Fu+AH@psiu;Jdc=Af@Fc+AH@psic
            Jdd=self.ell**2*self.K+self.M+2*Q.T@_left(w*H*self.ell/(self.gc*fac),Q)+Af@Fd
        return r,bmat([[Juu,Juc,Jud],[Jcu,Jcc,Jcd],[Jdu,Jdc,Jdd]],format='csr'),f

    def initial_state(self,soc=.9,crack_fraction=.02,seed_energy_J_m3=1e12):
        if not 0<soc<1 or not 0<=crack_fraction<1:raise ValueError('Invalid initial SOC/crack')
        n=self.n;x=self.quadrature_coordinates
        H=seed_energy_J_m3/self.material.young_Pa*np.exp(-100*(x[1]/(self.ell*self.seed_width_over_length))**2)*(x[0]>=1-crack_fraction)
        multiplier=1 if self.geometry=='sphere' else 1+self.material.poisson
        u=(multiplier*self.beta*(soc-self.material.reference_soc)*self.basis.doflocs).T.ravel()
        c=np.full(n,soc);zero=np.zeros(self.nq)
        probe=ParticleState(u,c,np.zeros(n),H,zero,zero,0.)
        H=self.fields(u,c,np.zeros(n),probe)['history']
        A=self.gc*(self.ell*self.K+self.M/self.ell)+self.Q.T@_left(2*self.w*H,self.Q)
        rhs=self.Q.T@(2*self.w*H)
        if self.damage_constraint=='irreversible_vi':
            from .phase_field import solve_box_quadratic
            d=solve_box_quadratic(A.tocsr(),rhs,np.zeros(n))[0]
        else:d=spsolve(A.tocsc(),rhs)
        # Axisymmetry admits a stress-free uniform expansion. Plane strain
        # requires a transverse strain (1+nu)*beta*c to satisfy free traction.
        old=ParticleState(u,c,d,H,zero,zero,0.)
        f=self.fields(u,c,d,old)
        # Initial equilibrated state does not count as an artificial load cycle.
        return ParticleState(u,c,d,np.maximum(H,f['plus']),zero,f['alpha'],0.)

    def cracked_fraction(self,d,threshold=.95):
        """Exact P1 superlevel measure, including the axisymmetric 2*pi*r.

        Counting quadrature points is NOT a converged crack-area measurement:
        a moving .95 contour can jump between quadrature points. Clip each
        affine triangle analytically and integrate its first radial moment.
        """
        if not 0<threshold<1:raise ValueError('Subunit crack threshold required')
        v=np.asarray(d)[self.mesh.t].T
        p=self.mesh.p[:,self.mesh.t].transpose(2,1,0)
        cross=(p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,1,1]-p[:,0,1])*(p[:,2,0]-p[:,0,0])
        area=np.abs(cross)/2
        volume=area*(2*np.pi*p[:,:,0].mean(axis=1) if self.geometry=='sphere' else 1)
        count=(v>threshold).sum(axis=1);measured=np.where(count==3,volume,0.)
        for number in (1,2):
            selected=np.flatnonzero(count==number)
            if not len(selected):continue
            # one high vertex: include its corner; one low: subtract its corner.
            idx=np.argmax(v[selected],axis=1) if number==1 else np.argmin(v[selected],axis=1)
            j=(idx+1)%3;k=(idx+2)%3;V=v[selected];P=p[selected];ii=np.arange(len(selected))
            f=(threshold-V[ii,idx])/(V[ii,j]-V[ii,idx]);g=(threshold-V[ii,idx])/(V[ii,k]-V[ii,idx])
            ri=P[ii,idx,0];rj=ri+f*(P[ii,j,0]-ri);rk=ri+g*(P[ii,k,0]-ri)
            part=area[selected]*f*g*(2*np.pi*(ri+rj+rk)/3 if self.geometry=='sphere' else 1)
            measured[selected]=part if number==1 else volume[selected]-part
        return float(measured.sum()/volume.sum())

    def advance(self,old,dt_s,outward_flux_mol_m2_s,*,tolerance=1e-8,max_iterations=30):
        if not np.isfinite([dt_s,outward_flux_mol_m2_s,tolerance]).all() or dt_s<=0 or not 0<tolerance<1:
            raise ValueError('Finite positive step/tolerance required')
        x=np.r_[old.u,old.c,old.d].copy();free=self.free;n=self.n
        scaling=None;factorizations=0
        def constrained(r,J,x,diagonal=None):
            diagonal=J.diagonal()[2*n:] if diagonal is None else diagonal
            diagonal=np.maximum(np.abs(diagonal),1e-30)
            values=x[2*n:];predictor=values-r[2*n:]/diagonal
            low=np.zeros(2*n,dtype=bool);high=low.copy();lower_bound=np.r_[np.zeros(n),old.d]
            if self.damage_constraint=='irreversible_vi':
                low[n:]=predictor[n:]<=old.d;high[n:]=(predictor[n:]>=1)&~low[n:]
            if self.boundary_policy=='surface_limits':
                if outward_flux_mol_m2_s>0:low[:n]=self.surface_nodes&(predictor[:n]<=0)
                if outward_flux_mol_m2_s<0 and self.concentration_domain=='bounded':high[:n]=self.surface_nodes&(predictor[:n]>=1)
            active=np.flatnonzero(low|high);bounds=np.where(low,lower_bound,1.)
            r=r.copy();r[2*n+active]=diagonal[active]*(values[active]-bounds[active])
            if J is not None:
                keep=np.ones(4*n);keep[2*n+active]=0.
                added=np.zeros(4*n);added[2*n+active]=diagonal[active]
                J=_left(keep,J)+diags(added)
            return r,J,low|high
        for iteration in range(max_iterations+1):
            cache=self._linear_cache
            fresh=self.linearization=='newton' or cache is None or cache['dt']!=dt_s or iteration%6==5
            if not fresh:
                r,f=self.residual(x,old,dt_s,outward_flux_mol_m2_s,jacobian=False)
                constraint_diagonal=cache['constraint_diagonal']
                raw_r=r.copy();r,_,active=constrained(r,None,x,constraint_diagonal)
                # Chord iteration reuses a linearization, never the physical
                # residual/history. Refresh after stagnation/line-search failure
                # or six iterations. Infinitesimal active-set flips otherwise
                # cause a full factorization at every iteration.
            if fresh:
                r,J,f=self.residual(x,old,dt_s,outward_flux_mol_m2_s)
                constraint_diagonal=J.diagonal()[2*n:];raw_r=r.copy()
                r,J,active=constrained(r,J,x)
                scale=np.maximum(np.abs(J.diagonal()[free]),1e-30)
                cache=dict(dt=dt_s,constraint_diagonal=constraint_diagonal,active=active,scale=scale,lu=None,J=J)
            if scaling is None:scaling=cache['scale']
            norm=float(np.max(np.abs(r[free])/scaling))
            inventory=float(self.lump@(x[2*n:3*n]-old.c))
            nominal=outward_flux_mol_m2_s*self.material.radius_m/(self.material.diffusivity_m2_s*self.material.cmax_mol_m3)*self.surface_load
            actual=nominal.copy();actual[active[:n]]-=raw_r[2*n:3*n][active[:n]]
            expected=-float(actual.sum())*dt_s/self.time_scale
            mass_error=abs(inventory-expected)/max(abs(expected),self.volume*1e-12)
            if norm<tolerance and mass_error<1e-7:
                c=x[2*n:3*n];d=x[3*n:]
                if c.min()<-1e-10 or (self.concentration_domain=='bounded' and c.max()>1+1e-10):
                    raise RuntimeError('Concentration bounds violated; reduce step, no clipping')
                sign=np.sign(outward_flux_mol_m2_s)
                wrong_flux=np.minimum(sign*actual,0)
                if -wrong_flux.sum()>1e-7*max(abs(nominal.sum()),1e-30):
                    raise RuntimeError('Surface cutoff requires unphysical reverse flux; refine transport discretization')
                if d.min()<-1e-6 or d.max()>1+1e-6:raise RuntimeError('Damage bounds violated; refine FE discretization')
                healing=float(np.max(old.d-d))
                if healing>1e-6:raise RuntimeError(f'History-field model healed by {healing}; no accepted state')
                state=ParticleState(x[:2*n],c,d,f['history'],f['fatigue'],f['alpha'],old.time_s+dt_s)
                return state,dict(newton_iterations=iteration,residual=norm,mass_error=mass_error,
                    factorizations=factorizations,
                    surface_limited_nodes=int(active[:n].sum()),
                    delivered_flux_ratio=float(actual.sum()/nominal.sum()) if nominal.sum()!=0 else 1.,
                    lithium_change_dimensionless=inventory,
                    min_soc=float(c.min()),max_soc=float(c.max()),
                    saturation_excess_volume_fraction=float(self.w@(f['cq']>1+1e-10)/self.volume),
                    intact_saturation_excess_volume_fraction=float(self.w@((f['cq']>1+1e-10)&(f['dq']<.95))/self.volume),
                    physical_concentration_passed=bool(c.min()>=-1e-10 and c.max()<=1+1e-10),
                    damage_healing=healing,mean_soc=float(self.lump@c/self.volume),
                    mean_damage=float(self.lump@d/self.volume),cracked_fraction=self.cracked_fraction(d),
                    cracked_fraction_quadrature=float(self.w@(f['dq']>.95)/self.volume),
                    mean_toughness_factor=float(self.w@f['factor']/self.volume))
            if cache['lu'] is None:
                cache['lu']=splu(cache['J'][free][:,free].tocsc());cache.pop('J');factorizations+=1
            self._linear_cache=cache
            delta=cache['lu'].solve(-r[free])
            if not np.isfinite(delta).all():raise RuntimeError('Nonfinite monolithic Newton direction')
            merit=np.linalg.norm(r[free]/scaling);step=1.
            for _ in range(22):
                trial=x.copy();trial[free]+=step*delta
                rt,_=self.residual(trial,old,dt_s,outward_flux_mol_m2_s,jacobian=False)
                rt,_,_=constrained(rt,None,trial,constraint_diagonal)
                if np.linalg.norm(rt[free]/scaling)<=(1-1e-4*step)*merit:
                    x=trial;break
                step*=.5
            else:
                self._linear_cache=None
                if not fresh:continue
                raise RuntimeError('Monolithic line search failed; accepted history unchanged')
        raise RuntimeError('Monolithic iteration budget exhausted; reduce time step')
