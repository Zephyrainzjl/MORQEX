"""One material state -> two transport networks -> conservative region ports.

The caller supplies constitutive conductivity and physical face transmission.
Damage is not interpreted as open electrolyte volume. Resolved pore plus an
explicit porous-CBD law is the only ionic domain in this fixed-volume adapter.
"""
from dataclasses import dataclass,replace
import numpy as np
from scipy import sparse
from .graph_transport import voxel_network,certify_update
from .network_dfn import diffusion_edges


class NetworkCache:
    """Event-triggered Schur reuse; bound is transport-only, not DFN accuracy."""
    def __init__(self,tolerance=0.):
        if not 0<=tolerance<1:raise ValueError('Subunit cache tolerance required')
        self.tolerance=tolerance;self.reference=None;self.result=None;self.rebuilds=0;self.reuses=0

    def evaluate(self,current):
        certificate=None if self.reference is None else certify_update(self.reference.network,current.network,tolerance=self.tolerance)
        rebuild=certificate is None or certificate.rebuild
        if rebuild:
            self.result=current.homogenize();self.reference=current;self.rebuilds+=1
        else:self.reuses+=1
        return self.result,{'rebuilt':rebuild,'rebuilds':self.rebuilds,'reuses':self.reuses,
            'relative_quadratic_bound':0. if rebuild else certificate.relative_edge_change,
            'reason':'initial operator' if certificate is None else certificate.reason}

    def restore(self,layout,conductance,rebuilds,reuses):
        from .contracts import ConductingNetwork
        old=layout.network
        network=ConductingNetwork(old.vertex_count,old.edges,conductance,old.anchors,old.channel,old.provenance)
        self.reference=replace(layout,network=network)
        self.result=self.reference.homogenize()
        self.rebuilds=int(rebuilds);self.reuses=int(reuses)


@dataclass(frozen=True)
class DualNetworkState:
    electronic_conductivity_S_m: float
    electrolyte_efficiency: float
    reaction_fractions: np.ndarray
    diffusion_edge_factors: np.ndarray
    electron_accessible: np.ndarray
    ion_accessible: np.ndarray
    reaction_face_accessible: np.ndarray
    electronic_potential: np.ndarray
    ionic_potential: np.ndarray
    diagnostics: dict


class DualNetworkClosure:
    def __init__(self,diffusion,port,*,axis=2,update_tolerance=0.,ionic_reference_conductivity_S_m=1.):
        self.diffusion=diffusion;self.port=port;self.axis=axis
        reference=float(ionic_reference_conductivity_S_m)
        if not np.isfinite(reference) or reference<=0:raise ValueError('Positive ionic normalization conductivity required')
        self.ionic_reference_conductivity_S_m=reference
        if not np.array_equal(diffusion.voxel_ids,port.voxel_ids):raise ValueError('Same material cells required')
        self.electronic_cache=NetworkCache(update_tolerance);self.ionic_cache=NetworkCache(update_tolerance)
        self.region_edges,self.reference_edge_weights=diffusion_edges(port)
        K=sparse.triu(diffusion.stiffness,k=1).tocoo()
        self.cell_edges=np.column_stack([K.row,K.col]);self.cell_weights=-K.data
        coordinates=diffusion.coordinates_zyx[diffusion.face_cells]
        delta=diffusion.face_normals[:,::-1].astype(int)
        adjacent=coordinates+delta
        inside=np.all((adjacent>=0)&(adjacent<np.array(port.voxel_shape)),axis=1)
        self.face_inside=inside;self.face_neighbor=np.zeros(len(inside),int)
        self.face_neighbor[inside]=np.ravel_multi_index(adjacent[inside].T,port.voxel_shape)
        self.adjacent=adjacent

    def evaluate(self,electronic_S_m,ionic_efficiency,*,solid_face_factors=None,
                 electronic_face_factors=None,ionic_face_factors=None,interface_fractions=None):
        shape=self.port.voxel_shape;h=self.diffusion.volume.spacing
        e=voxel_network(electronic_S_m,h,axis=self.axis,channel='electronic',reservoir_side=1,face_factors=electronic_face_factors)
        # eta is dimensionless. Graph edges are in S: use an explicit reference
        # conductivity, then divide the effective S/m by the same reference.
        # Actual electrolyte kappa(c_e,T) remains in the macroscopic equations.
        i=voxel_network(np.asarray(ionic_efficiency)*self.ionic_reference_conductivity_S_m,h,axis=self.axis,channel='ionic',reservoir_side=0,face_factors=ionic_face_factors)
        if e.shape!=shape or i.shape!=shape:raise ValueError('Transport networks must share the material reference lattice')
        er,ec=self.electronic_cache.evaluate(e);ir,ic=self.ionic_cache.evaluate(i)
        ea=e.reservoir_accessible();ia=i.reservoir_accessible()
        fv=self.diffusion;port=self.port
        face= self.face_inside & fv.reactive & ea.ravel()[port.voxel_ids[fv.face_cells]]
        face &= ia.ravel()[self.face_neighbor]
        fraction=np.ones(len(face)) if interface_fractions is None else np.asarray(interface_fractions,float)
        if fraction.shape!=face.shape or not np.isfinite(fraction).all() or np.any((fraction<0)|(fraction>1)):
            raise ValueError('Physical accessible interface fractions in [0,1] required')
        area=np.bincount(port.partition[fv.face_cells],weights=fv.face_areas*face*fraction,minlength=port.dimension)/fv.rve_volume
        # Newly created surfaces require explicit extra geometry, not >1 masks.
        if np.any(area>port.reaction_area*(1+1e-12)):raise ValueError('Reaction interface exceeds reference geometry; rebuild surface port')
        af=np.divide(area,port.reaction_area,out=np.zeros_like(area),where=port.reaction_area>0)
        # Only remove summation-order roundoff after the explicit geometric
        # upper-bound gate above; no state or physical area is projected.
        af=np.minimum(af,1.)
        factors=np.ones((3,)+shape) if solid_face_factors is None else np.asarray(solid_face_factors,float)
        if factors.shape!=(3,)+shape or not np.isfinite(factors).all() or np.any(factors<0):raise ValueError('Physical solid diffusion face factors required')
        a,b=self.cell_edges.T
        coords=fv.coordinates_zyx
        axes=np.argmax(abs(coords[b]-coords[a]),axis=1)
        lower=np.minimum(coords[a],coords[b])
        values=factors[(axes,*lower.T)]*self.cell_weights
        P=sparse.csr_matrix((np.ones(len(port.partition)),(np.arange(len(port.partition)),port.partition)),shape=(len(port.partition),port.dimension))
        n=len(port.partition)
        K=sparse.coo_matrix((np.r_[values,values,-values,-values],(np.r_[a,b,a,b],np.r_[a,b,b,a])),shape=(n,n)).tocsr()
        reduced=(P.T@K@P).tocsr()
        ef=np.array([-reduced[u,v]/g for (u,v),g in zip(self.region_edges,self.reference_edge_weights)])
        # Report, rather than hide, sub-region disconnections requiring enrichment.
        from scipy.sparse.csgraph import connected_components
        adjacency=K.copy();adjacency.eliminate_zeros()
        split=[]
        for region in range(port.dimension):
            ids=np.flatnonzero(port.partition==region)
            if connected_components(adjacency[ids][:,ids],directed=False)[0]>1:split.append(region)
        return DualNetworkState(er['effective_S_m'],ir['effective_S_m']/self.ionic_reference_conductivity_S_m,af,ef,ea,ia,face,
            er['potential'],ir['potential'],{'electronic':ec,'ionic':ic,
            'electronic_components':e.network.components()[0],'ionic_components':i.network.components()[0],
            'accessible_reaction_area_m2':float(area.sum()*fv.rve_volume),
            'unresolved_split_regions':split,'material_volume_m3':float(fv.volumes.sum()),
            'ionic_reference_conductivity_S_m':self.ionic_reference_conductivity_S_m,
            'scope':'Fixed material volumes; prescribed conductance laws; no inferred crack wetting or new pore volume'})
