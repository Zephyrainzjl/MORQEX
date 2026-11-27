"""MFEM conforming h-refinement and exact nested-P1 state transfer.

The original material geometry and phase/voxel IDs are preserved. This is not
crack insertion or a certified error estimator. No coarsening is performed.
"""
from dataclasses import dataclass
import ctypes as ct
import hashlib,json,subprocess,shutil,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from morqex import Workspace
from morqex.mechanics import TetMesh
from .backend import MFEMScalarBackend,NativeConfig,require_local_budget,_sha,_LOCK

_NATIVE=Path(__file__).with_name('native_refinement')


def _identity(base):
    inputs={'adapter':{p.name:_sha(p) for p in sorted(_NATIVE.iterdir()) if p.is_file()},
            'mfem':base.manifest['dependency_sha256'],'mfem_build':base.manifest['identity'],
            'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0]}
    return hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),inputs


def build_refinement(config,workspace):
    require_local_budget()
    if config.provider!='vendored':raise ValueError('Use vendored MFEM for refinement')
    ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    base=MFEMScalarBackend(config,ws);identity,inputs=_identity(base)
    out=ws.path('build/top_damage_refinement',identity[:20]);out.mkdir(parents=True,exist_ok=True)
    dependency=Path(base.manifest['dependency']);started=time.perf_counter()
    ninja=shutil.which('ninja') or str(Path(sys.executable).parent/'ninja')
    with (out/'build.log').open('a') as log:
        subprocess.run(['cmake','-S',str(_NATIVE),'-B',str(out),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_MAKE_PROGRAM='+ninja,'-DMFEM_SOURCE='+str(Path(__file__).parents[1]/'vendor/mfem'),
            '-DMFEM_BUILD='+str(dependency.parent),'-DMFEM_LIBRARY='+str(dependency)],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
        subprocess.run(['cmake','--build',str(out),'--parallel','1'],stdout=log,stderr=subprocess.STDOUT,
            check=True,timeout=config.slice_seconds)
    lib=out/'libmorqex_refinement.so'
    result={'state':'complete','identity':identity,'inputs':inputs,'library':str(lib),
        'library_sha256':_sha(lib),'dependency':str(dependency),'dependency_sha256':_sha(dependency),
        'seconds':time.perf_counter()-started}
    (out/'manifest.json').write_text(json.dumps(result,indent=2));return result


@dataclass(frozen=True)
class RefinementResult:
    mesh: TetMesh
    parent_cells: np.ndarray
    prolongation: csr_matrix

    def cell_field(self,field):
        return np.asarray(field)[self.parent_cells].copy()

    def nodal_field(self,field):
        return self.prolongation@np.asarray(field)


class RefinementBackend:
    def __init__(self,config,workspace):
        require_local_budget()
        ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
        self.base=MFEMScalarBackend(config,ws);identity,_=_identity(self.base)
        self.manifest=json.loads(ws.path('build/top_damage_refinement',identity[:20],'manifest.json').read_text())
        m=self.manifest
        if m['identity']!=identity or m['state']!='complete' or _sha(m['library'])!=m['library_sha256'] or _sha(m['dependency'])!=m['dependency_sha256']:
            raise ValueError('Missing or changed refinement backend')
        self.lib=ct.CDLL(m['library']);self.lib.mq_refinement_abi.restype=ct.c_int
        if self.lib.mq_refinement_abi()!=1:raise ValueError('Wrong refinement ABI')
        dp=ct.POINTER(ct.c_double);ip=ct.POINTER(ct.c_int)
        self.lib.mq_refine.argtypes=[ct.c_int,ct.c_int,dp,ip,ct.c_int,ip,ct.c_char_p,ct.c_int]
        self.lib.mq_refine.restype=ct.c_void_p
        for name in ('mq_refined_nv','mq_refined_ne'):
            getattr(self.lib,name).argtypes=[ct.c_void_p];getattr(self.lib,name).restype=ct.c_int
        self.lib.mq_refined_copy.argtypes=[ct.c_void_p,dp,ip,ip];self.lib.mq_refined_copy.restype=None
        self.lib.mq_refined_free.argtypes=[ct.c_void_p];self.lib.mq_refined_free.restype=None
        self.lib.mq_tet_rule.argtypes=[ct.c_int,dp,dp];self.lib.mq_tet_rule.restype=ct.c_int
        self.lib.mq_shape_hvp.argtypes=[ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_longlong)]+[dp]*5
        self.lib.mq_shape_hvp.restype=ct.c_int
        self.lib.mq_p2_hvp.argtypes=[ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_longlong)]+[dp]*6
        self.lib.mq_p2_hvp.restype=ct.c_int

    def tetrahedron_rule(self,order=5):
        if type(order) is not int or not 1<=order<=9:raise ValueError('MFEM quadrature order must be 1..9')
        n=self.lib.mq_tet_rule(order,None,None)
        bary=np.empty((n,4));weight=np.empty(n);dp=lambda x:x.ctypes.data_as(ct.POINTER(ct.c_double))
        self.lib.mq_tet_rule(order,dp(bary),dp(weight))
        if np.min(weight)<=0 or abs(weight.sum()-1)>1e-13:raise ValueError('Positive normalized quadrature required')
        return bary,weight

    def tangent_action(self,mesh,tangent):
        C=np.ascontiguousarray(tangent);dp=lambda x:x.ctypes.data_as(ct.POINTER(ct.c_double))
        if C.shape!=(len(mesh.cells),9,9) or not np.isfinite(C).all():raise ValueError('Compatible finite tangents required')
        def action(vector):
            v=np.ascontiguousarray(vector,dtype=float).reshape(-1);out=np.empty_like(v)
            if len(v)!=3*len(mesh.points) or not np.isfinite(v).all():raise ValueError('Compatible finite vector required')
            code=self.lib.mq_shape_hvp(len(mesh.cells),len(v),mesh.cells.shape[1],
                mesh.dofs.ctypes.data_as(ct.POINTER(ct.c_longlong)),dp(mesh.gradients),dp(C),dp(mesh.volumes),dp(v),dp(out))
            if code:raise RuntimeError('Quadrature tangent action rejected dimensions')
            return out
        return action

    def quadratic_tangent_action(self,geometry,tangent):
        g=geometry;C=np.ascontiguousarray(tangent);dp=lambda x:x.ctypes.data_as(ct.POINTER(ct.c_double))
        reference_gradients=np.ascontiguousarray(g.base.gradients)
        if C.shape!=(g.quadrature_count,9,9) or not np.isfinite(C).all():raise ValueError('Compatible finite P2 tangents required')
        def action(vector):
            v=np.ascontiguousarray(vector,dtype=float).reshape(-1);out=np.empty_like(v)
            if len(v)!=3*len(g.points) or not np.isfinite(v).all():raise ValueError('Compatible finite vector required')
            code=self.lib.mq_p2_hvp(len(g.base.cells),g.nq,len(v),g.dofs.ctypes.data_as(ct.POINTER(ct.c_longlong)),
                dp(reference_gradients),dp(g.shape_gradient),dp(C),dp(g.volumes),dp(v),dp(out))
            if code:raise RuntimeError('Compiled P2 tangent action rejected dimensions')
            return out
        return action

    def refine(self,mesh,marked):
        require_local_budget()
        mark=np.asarray(marked)
        if mark.dtype.kind not in 'iu' or mark.ndim!=1 or not len(mark) or mark.min()<0 or mark.max()>=len(mesh.cells):
            raise ValueError('Nonempty integer cell-index selection required')
        if max(len(mesh.points),len(mesh.cells))>=np.iinfo(np.int32).max:
            raise ValueError('MFEM refinement uses int32 indices')
        mark=np.ascontiguousarray(np.unique(mark),dtype=np.int32)
        xyz=np.ascontiguousarray(mesh.points);tet=np.ascontiguousarray(mesh.cells,dtype=np.int32)
        dp=lambda a:a.ctypes.data_as(ct.POINTER(ct.c_double))
        ip=lambda a:a.ctypes.data_as(ct.POINTER(ct.c_int))
        err=ct.create_string_buffer(4096)
        with _LOCK:
            handle=self.lib.mq_refine(len(xyz),len(tet),dp(xyz),ip(tet),len(mark),ip(mark),err,len(err))
            if not handle:raise RuntimeError(err.value.decode())
            try:
                nv=self.lib.mq_refined_nv(handle);ne=self.lib.mq_refined_ne(handle)
                points=np.empty((nv,3));cells=np.empty((ne,4),np.int32);parent=np.empty(ne,np.int32)
                self.lib.mq_refined_copy(handle,dp(points),ip(cells),ip(parent))
            finally:self.lib.mq_refined_free(handle)
        if parent.min()<0 or parent.max()>=len(mesh.cells):raise RuntimeError('Invalid refinement parent labels')
        refined=TetMesh(points,cells,mesh.phases[parent],mesh.box,
            None if mesh.voxel_ids is None else mesh.voxel_ids[parent])
        volumes=np.bincount(parent,weights=refined.volumes,minlength=len(mesh.cells))
        if not np.allclose(volumes,mesh.volumes,rtol=1e-11,atol=0):raise RuntimeError('Refinement changed parent material volume')
        # Every refined vertex gets barycentric coordinates in one actual parent.
        # Coincident but disconnected sheets retain separate node identifiers.
        _,first=np.unique(cells.ravel(),return_index=True)
        owners=parent[first//4];old=mesh.cells[owners]
        delta=points-mesh.points[old[:,0]]
        weights=np.einsum('nij,nj->ni',mesh.gradients[owners],delta);weights[:,0]+=1
        if weights.min()<-1e-10 or weights.max()>1+1e-10:raise RuntimeError('Refined node outside its parent')
        weights=np.clip(weights,0,1);weights/=weights.sum(axis=1)[:,None]
        P=csr_matrix((weights.ravel(),old.ravel(),np.arange(0,4*nv+1,4)),shape=(nv,len(mesh.points)))
        if not np.allclose(P@mesh.points,points,rtol=0,atol=np.max(mesh.box[1]-mesh.box[0])*1e-12):
            raise RuntimeError('Nested interpolation does not reproduce coordinates')
        return RefinementResult(refined,parent,P)
