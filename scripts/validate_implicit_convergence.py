"""Independent manufactured Poisson convergence on generated Tet4 meshes."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def solve(mesh):
    p=mesh.points;c=mesh.cells;v=mesh.signed_volumes();x=p[c]
    C=np.concatenate((np.ones((*x.shape[:2],1)),x),axis=2)
    gradient=np.linalg.inv(C)[:,1:,:].transpose(0,2,1)
    local=v[:,None,None]*np.einsum('nik,njk->nij',gradient,gradient)
    A=sparse.coo_matrix((local.ravel(),(np.repeat(c,4,axis=1).ravel(),np.tile(c,(1,4)).ravel())),shape=(len(p),)*2).tocsr()
    # Symmetric degree-two Tet4 integration, separate from the mesh generator.
    bary=np.full((4,4),(5-np.sqrt(5))/20);np.fill_diagonal(bary,(5+3*np.sqrt(5))/20)
    quadrature=np.einsum('qa,nai->nqi',bary,x);z=quadrature;w=z*(1-z)
    force=2*(w[:,:,0]*w[:,:,1]+w[:,:,0]*w[:,:,2]+w[:,:,1]*w[:,:,2])
    load=v[:,None]*np.einsum('nq,qa->na',force,bary)/4
    b=np.bincount(c.ravel(),weights=load.ravel(),minlength=len(p))
    from morqex.modeling.exchange import _facet_owners
    owners,_=_facet_owners(c,mesh.facets,len(p));boundary=np.unique(mesh.facets[owners[:,1]<0]);free=np.setdiff1d(np.arange(len(p)),boundary)
    exact=np.prod(p*(1-p),axis=1);u=exact.copy()
    u[free]=spsolve(A[free][:,free],b[free]-A[free][:,boundary]@u[boundary])
    uh=np.einsum('qa,na->nq',bary,u[c]);truth=np.prod(w,axis=2)
    gu=np.einsum('na,nai->ni',u[c],gradient)
    gt=np.stack(((1-2*z[:,:,0])*w[:,:,1]*w[:,:,2],(1-2*z[:,:,1])*w[:,:,0]*w[:,:,2],(1-2*z[:,:,2])*w[:,:,0]*w[:,:,1]),axis=2)
    l2=float(np.sqrt(np.sum(v[:,None]*(uh-truth)**2)/4));h1=float(np.sqrt(np.sum(v[:,None,None]*(gu[:,None,:]-gt)**2)/4))
    residual=float(np.linalg.norm((A@u-b)[free])/max(np.linalg.norm(b[free]),1e-30))
    return {'nodes':len(p),'cells':len(c),'L2_error':l2,'H1_seminorm_error':h1,'relative_free_residual':residual,'energy':float(u@A@u)/2,
        'quality':mesh.report['quality'],'box_volume_error':mesh.report['physical_box_volume_relative_error']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',type=Path,default=os.environ.get('MORQEX_WORKSPACE'));args=parser.parse_args()
    from morqex.modeling import from_regions,smooth_voxels,SmoothingOptions,implicit_mesh,ImplicitMeshOptions,ProgressBar
    from morqex.modeling.voxel import external_path
    root=external_path(args.workspace)/'implicit_geometry/convergence';root.mkdir(parents=True,exist_ok=True)
    p=smooth_voxels(from_regions(np.ones((32,32,32),np.uint8),spacing=(1/32,)*3),options=SmoothingOptions(method='none'))
    result=[];start=time.perf_counter()
    for name in ('coarse','balanced','fine','very_fine'):
        mesh=implicit_mesh(p,options=ImplicitMeshOptions(precision=name,isolated=True,optimization_seconds=10),workspace=args.workspace,progress=ProgressBar(json_lines=True))
        report=solve(mesh);report['precision']=name;report['geometry_score_sha256']=mesh.report['geometry_score_sha256'];result.append(report)
    assert len({r['geometry_score_sha256'] for r in result})==1
    assert all(r['relative_free_residual']<1e-9 for r in result)
    assert all(b['L2_error']<a['L2_error'] and b['H1_seminorm_error']<a['H1_seminorm_error'] for a,b in zip(result,result[1:]))
    data={'passed':True,'problem':'-Laplace u=f; u=x(1-x)y(1-y)z(1-z), exact Dirichlet on actual mesh boundary',
        'scope':'homogeneous manufactured Poisson convergence; not arbitrary electrode mechanics/electrochemistry validation',
        'quadrature':'symmetric four-point degree2, used consistently; independent of CGAL','results':result,'elapsed_seconds':time.perf_counter()-start}
    (root/'convergence.json').write_text(json.dumps(data,indent=2))
    rows=''.join('<tr>'+''.join('<td>'+str(r[k])+'</td>' for k in ('precision','nodes','cells','L2_error','H1_seminorm_error','relative_free_residual'))+'</tr>' for r in result)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MORQEX convergence</title><style>body{font:16px system-ui;margin:40px}td,th{padding:12px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style><h1>Manufactured Poisson convergence / 制造解收敛</h1><p>Same prepared32³ geometry; four FEM densities. / 同一准备几何，四种有限元密度。</p><table><tr><th>Precision</th><th>Nodes</th><th>Tet4</th><th>L2</th><th>H1 seminorm</th><th>Residual</th></tr>'+rows+'</table><p>Homogeneous benchmark only; not complete electrode physics certification. / 仅均匀基准，不是完整电极物理认证。</p><a href="convergence.json">Full measured data / 完整实测</a>')
    from morqex.visualization.report_font import embed_workspace_font
    view=root/'index.html';view.write_text(embed_workspace_font(view.read_text(),args.workspace))
    print(json.dumps(data),flush=True)

if __name__=='__main__':main()
