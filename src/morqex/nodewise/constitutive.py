"""Spatial constitutive fields: a different conservative operator at each x.

The same equations serve homogeneous and heterogeneous electrode bindings.
Zero padding adds no lithium, reaction area, stiffness or evolving state. Local
matrices are not averaged before coupling to the DFN conservation equations.
"""
import hashlib
import numpy as np
import pybamm as pb
from ..electrochemistry.microstructure_dfn import MicrostructureStack


class _AtNodes:
    __name__='morqex_material_at_nodes'
    def __init__(self,values):self.values=np.asarray(values,float)
    def __call__(self,x):
        import casadi as ca
        if isinstance(x,(ca.MX,ca.SX,ca.DM)):
            if x.numel()!=len(self.values):raise ValueError('Coefficient grid and discretized DFN grid differ')
            return ca.DM(self.values)
        x=np.asarray(x)
        if np.isnan(x).any():return np.full_like(x,np.nan,dtype=float)
        if x.size!=len(self.values):raise ValueError('Coefficient grid and discretized DFN grid differ')
        return self.values.reshape(x.shape)


class NodalField(pb.Function):
    def __init__(self,values,child):
        self.values=np.array(values,float,copy=True);self.values.flags.writeable=False
        identity=hashlib.sha256(self.values.tobytes()).hexdigest()
        super().__init__(_AtNodes(self.values),child,name='function (MORQEX nodal '+identity+')')
    def _function_diff(self,children,idx):return pb.Scalar(0.)
    def _function_new_copy(self,children):return NodalField(self.values,children[0])
    def create_copy(self,new_children=None,perform_simplifications=True):return self._function_new_copy(self._children_for_copying(new_children))


def field(values,domain):
    values=np.asarray(values,float)
    if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('Finite spatial coefficient vector required')
    if np.all(values==values[0]):return float(values[0])
    coordinate=pb.SpatialVariable('x_n' if domain=='negative' else 'x_p',domain=f'{domain} electrode',auxiliary_domains={'secondary':'current collector'},coord_sys='cartesian')
    return NodalField(values,coordinate)


def fields(values,domain):
    a=np.asarray(values,float);result=np.empty(a.shape[1:],dtype=object)
    for index in np.ndindex(result.shape):result[index]=field(a[(slice(None),*index)],domain)
    return result


class SpatialMatrix:
    def __init__(self,values):self.values=values
    def toarray(self):return self.values.copy()
    def __getitem__(self,key):return self.values[key]


class NodewiseOperator:
    def __init__(self,domain,kernels):
        if not kernels:raise ValueError('At least one node kernel required')
        self.domain=domain;self.kernels=tuple(kernels);self.nodes=len(kernels)
        self.local=tuple(k.operator for k in kernels);self.dimension=max(op.dimension for op in self.local)
        m=self.dimension;N=self.nodes
        self.reference_concentration=self.local[0].reference_concentration
        if any(op.reference_concentration!=self.reference_concentration for op in self.local):raise ValueError('Different local initial reference concentrations require an explicit initialization field')
        if any(hasattr(op,'regional_chemistry') for op in self.local):raise ValueError('Nodewise doping needs explicitly projected spatial chemistry fields; no silent scalar fallback')
        weights=np.zeros((N,m));areas=weights.copy();K=np.zeros((N,m,m));H=np.zeros((N,m+2,m+2))
        for n,op in enumerate(self.local):
            k=op.dimension;weights[n,:k]=op.weights;areas[n,:k]=op.reaction_area;K[n,:k,:k]=op.stiffness.toarray()
            indices=[0,*range(1,k+1),m+1];H[n][np.ix_(indices,indices)]=op.H
        self.numeric_weights=weights;self.numeric_areas=areas
        self.weights=fields(weights,domain);self.division_weights=fields(np.where(weights>0,weights,1.),domain)
        self.reaction_area=fields(areas,domain);self.reaction_masks=fields((areas>0).astype(float),domain)
        self.active_fraction=field(weights.sum(axis=1),domain);self.area_density=field(areas.sum(axis=1),domain)
        self.porosity=field([op.porosity for op in self.local],domain)
        self.stiffness=SpatialMatrix(fields(K,domain));self.H=fields(H,domain)
        self.diffusion_edges=list(zip(*np.nonzero(np.any(np.triu(K,1)!=0,axis=0))))
        self.transport={component:{'efficiency':field([op.transport[component]['efficiency'] for op in self.local],domain)} for component in ['electrode','electrolyte']}
        if any(hasattr(op,'spreading') for op in self.local):
            if not all(hasattr(op,'spreading') for op in self.local):raise ValueError('Choose the same electrochemical port closure for all nodes in an electrode')
            self.spreading={}
            for channel in ['electronic','ionic']:
                matrix=np.zeros((N,m,m))
                for n,op in enumerate(self.local):matrix[n,:op.dimension,:op.dimension]=op.spreading[channel]
                self.spreading[channel]=fields(matrix,domain)
        self.fingerprint=hashlib.sha256(''.join(op.metadata()['fingerprint'] for op in self.local).encode()).hexdigest()
    def metadata(self):
        metadata=self.local[0].metadata().copy()
        metadata.update(fingerprint=self.fingerprint,nodewise=True,nodes=self.nodes,
            local_fingerprints=[op.metadata()['fingerprint'] for op in self.local],
            local_dimensions=[op.dimension for op in self.local],padded_dimension=self.dimension,
            scope='independent local concentration/damage/reservoir states; geometry may share immutable operators')
        laws=[op.metadata().get('transport_law') for op in self.local]
        if any(law!=laws[0] for law in laws):raise ValueError('Different node transport laws require a spatial law adapter')
        return metadata


class NodewiseCohesive:
    def __init__(self,operator):
        self.operator=operator;N=operator.nodes;m=operator.dimension
        parts=[k.parts() for k in operator.kernels];rank=max(len(p[1]) for p in parts)
        residuals=[p[3] for p in parts]
        if len(set(residuals))!=1:raise ValueError('Spatial residual interface stiffness not yet qualified')
        self.residual=residuals[0];D=np.zeros((N,m+2,m+2));F=np.zeros((N,rank,m+2));rho=np.zeros((N,rank))
        for n,(k,(direct,factors,eigenvalues,_)) in enumerate(zip(operator.kernels,parts)):
            q=k.operator.dimension;indices=[0,*range(1,q+1),m+1]
            D[n][np.ix_(indices,indices)]=direct
            F[n,:len(factors),indices]=factors.T;rho[n,:len(factors)]=eigenvalues
        self.numeric_direct=D;self.numeric_factors=F;self.numeric_eigenvalues=rho
        self.direct=fields(D,operator.domain);self.factors=fields(F,operator.domain);self.eigenvalues=fields(rho,operator.domain)
    def hessian(self,damage):
        H=[]
        for k in self.operator.kernels:H.append(k.hessian(damage))
        m=self.operator.dimension;full=np.zeros((len(H),m+2,m+2))
        for i,h in enumerate(H):
            n=len(h)-2;idx=[0,*range(1,n+1),m+1];full[i][np.ix_(idx,idx)]=h
        return fields(full,self.operator.domain)


class NodewiseStack(MicrostructureStack):
    def __init__(self,kernels,parameters,**kwargs):
        if set(kernels)!={'negative','positive'}:raise ValueError('Both electrode node lists required')
        super().__init__(kernels['negative'][0].operator,kernels['positive'][0].operator,parameters,**kwargs)
        self.electrodes={d:NodewiseOperator(d,k) for d,k in kernels.items()}
        from ..electrochemistry.aging_dfn import EnergyKernel
        self.kernels={d:EnergyKernel(op,NodewiseCohesive(op)) for d,op in self.electrodes.items()}
    def update_parameters(self,parameters):
        for d,op in self.electrodes.items():
            # Validate every material/reference contract against the scalar path.
            for local in op.local:
                check=MicrostructureStack(local if d=='negative' else self.electrodes['negative'].local[0],
                    local if d=='positive' else self.electrodes['positive'].local[0],parameters,
                    allow_linear_ocp_entropy=self.linear_ocp_entropy)
                check.update_parameters(parameters.copy())
            def parameter_field(value):
                return (lambda *coordinates:value) if isinstance(value,pb.Symbol) else value
            parameters.update({f'{d.capitalize()} electrode porosity':parameter_field(op.porosity),
                f'{d.capitalize()} electrode active material volume fraction':parameter_field(op.active_fraction)})
        return parameters
    def var_pts(self,model,separator_nodes=5):
        return {**model.default_var_pts,'x_n':self.electrodes['negative'].nodes,
            'x_s':separator_nodes,'x_p':self.electrodes['positive'].nodes,'r_n':2,'r_p':2}
