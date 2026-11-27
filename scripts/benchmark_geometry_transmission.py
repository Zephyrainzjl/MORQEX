"""Classical spherical conduction transmission: geometry + P1 convergence.

Prescribes the exact infinite-medium solution on a finite cube boundary. The
sphere is voxelized, so errors include geometry approximation and FE error.
This is a verification benchmark, not a calibrated battery simulation.
"""
import argparse,base64,json,os,time
from pathlib import Path
import h5py,numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import cg

RADIUS=.25
K_OUT=1.
K_IN=5.
BETA=(K_IN-K_OUT)/(K_IN+2*K_OUT)
GAMMA=3*K_OUT/(K_IN+2*K_OUT)


def exact(p):
    x=np.asarray(p)-.5;r=np.linalg.norm(x,axis=-1);inside=r<RADIUS
    safe=np.maximum(r,RADIUS)
    potential=-x[...,2]+BETA*RADIUS**3*x[...,2]/safe**3
    gradient=-np.broadcast_to(np.array([0.,0.,1.]),x.shape).copy()
    gradient+=BETA*RADIUS**3*(np.array([0.,0.,1.])/safe[...,None]**3-3*x[...,2,None]*x/safe[...,None]**5)
    potential=np.where(inside,-GAMMA*x[...,2],potential)
    gradient=np.where(inside[...,None],np.array([0.,0.,-GAMMA]),gradient)
    return potential,gradient


def reference_energy():
    # Green identity: 1/2 integral_boundary(phi k grad(phi).n).
    points,weights=np.polynomial.legendre.leggauss(24);points=(points+1)/2;weights=weights/2
    a,b=np.meshgrid(points,points,indexing='ij');weight=(weights[:,None]*weights).ravel();total=0.
    for axis in range(3):
        other=[j for j in range(3) if j!=axis]
        for side in (0.,1.):
            p=np.empty((len(weight),3));p[:,axis]=side;p[:,other]=np.column_stack((a.ravel(),b.ravel()))
            u,g=exact(p);total+=float((u*K_OUT*g[:,axis]*(2*side-1))@weight)
    return .5*total


def solve(size,folder):
    from morqex.modeling import from_regions,partitioned_tetgen
    start=time.perf_counter();z,y,x=np.indices((size,)*3,dtype=float);r2=sum(((q+.5)/size-.5)**2 for q in (z,y,x))
    phase=np.where(r2<RADIUS**2,7,0).astype(np.uint8)
    model=from_regions(phase,spacing=(1/size,)*3,region_names={0:'matrix',7:'sphere'});folder.mkdir(parents=True)
    model.save(folder/'geometry.h5')
    artifact=partitioned_tetgen(model,folder/'mesh',surface_method='voxel_aligned',slab_depth=max(2,size//2))
    with h5py.File(artifact.hdf5) as h:
        p=h['mesh/points_m'][:];t=h['mesh/tetrahedra'][:];region=h['mesh/phase'][:]
    basis=p[t[:,1:]]-p[t[:,:1]];volume=np.linalg.det(basis)/6
    gradient=np.linalg.inv(basis).transpose(0,2,1);gradient=np.concatenate((-gradient.sum(1,keepdims=True),gradient),axis=1)
    k=np.where(region==7,K_IN,K_OUT)
    local=np.einsum('nki,nli->nkl',gradient,gradient)*(volume*k)[:,None,None]
    matrix=coo_matrix((local.ravel(),(np.repeat(t,4,axis=1).ravel(),np.tile(t,(1,4)).ravel())),shape=(len(p),len(p))).tocsr()
    fixed=np.flatnonzero(np.any(np.isclose(p,0.,rtol=0,atol=1e-12)|np.isclose(p,1.,rtol=0,atol=1e-12),axis=1))
    free=np.setdiff1d(np.arange(len(p)),fixed);u=np.zeros(len(p));u[fixed]=exact(p[fixed])[0]
    rhs=-matrix[free][:,fixed]@u[fixed];interior=matrix[free][:,free];iterations=[0]
    def count(_):iterations[0]+=1
    u[free],info=cg(interior,rhs,M=diags(1/interior.diagonal()),rtol=1e-11,atol=0.,maxiter=5000,callback=count)
    if info:raise ArithmeticError('Jacobi-CG failed to converge: '+str(info))
    residual=matrix@u
    relative_residual=float(np.linalg.norm(residual[free])/max(np.linalg.norm(rhs),1e-30))
    a=.5854101966249685;b=.1381966011250105;bary=np.full((4,4),b);np.fill_diagonal(bary,a)
    l2=norm=h1=gnorm=0.
    for first in range(0,len(t),20000):
        cells=t[first:first+20000];xyz=p[cells];uq=u[cells]@bary.T;q=np.einsum('qk,nki->nqi',bary,xyz)
        ref,grad=exact(q);g=np.einsum('nk,nki->ni',u[cells],gradient[first:first+len(cells)])
        v=volume[first:first+len(cells)];kk=k[first:first+len(cells)]
        l2+=float(v@np.mean((uq-ref)**2,axis=1));norm+=float(v@np.mean(ref**2,axis=1))
        h1+=float((v*kk)@np.mean(np.sum((g[:,None,:]-grad)**2,axis=2),axis=1))
        gnorm+=float((v*kk)@np.mean(np.sum(grad**2,axis=2),axis=1))
    energy=float(.5*u@residual);reference=reference_energy()
    import meshio
    meshio.write(folder/'solution.vtu',meshio.Mesh(p,[('tetra',t)],point_data={'potential':u,'analytic_potential':exact(p)[0]},cell_data={'region':[region]}))
    report={'size':size,'source_voxels':int(size**3),'nodes':len(p),'tetrahedra':len(t),
        'relative_l2_error_quadrature':float(np.sqrt(l2/norm)),'relative_gradient_error_quadrature':float(np.sqrt(h1/gnorm)),
        'dirichlet_functional':energy,'analytic_boundary_functional':reference,'relative_functional_error':abs(energy/reference-1),
        'linear_solver':'Jacobi-preconditioned CG, rtol1e-11','linear_iterations':iterations[0],
        'relative_free_residual':relative_residual,'global_reaction_sum':float(residual.sum()),
        'sphere_voxel_volume_error':abs(float((phase==7).mean())/(4*np.pi*RADIUS**3/3)-1),
        'seconds':time.perf_counter()-start,'scope':'combined voxel-boundary and P1 error; four-point cell quadrature; Dirichlet functional is half dissipation, not stored energy; not a rigorous norm bound'}
    if not np.all(np.isfinite(u)) or relative_residual>1e-9:raise ArithmeticError('Conduction solve did not meet its residual requirement')
    (folder/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=os.environ.get('MORQEX_WORKSPACE'))
    p.add_argument('--sizes',nargs='+',type=int,default=[16,32,64]);a=p.parse_args()
    if a.workspace is None or any(n<8 or n>96 for n in a.sizes) or len(a.sizes)<3 or a.sizes!=sorted(set(a.sizes)):
        p.error('External workspace and at least three increasing distinct sizes in8..96 required')
    from morqex.modeling.voxel import external_path
    folder=external_path(a.workspace)/'geometry_transmission'
    if folder.exists():raise FileExistsError('Archive the previous benchmark explicitly before rerunning')
    folder.mkdir(parents=True)
    # Analytic transmission continuity and normal flux continuity, independent
    # of the FE mesh, on the exact spherical interface.
    beta=BETA;gamma=GAMMA
    if abs(1-beta-gamma)>1e-14 or abs(K_OUT*(1+2*beta)-K_IN*gamma)>1e-14:raise ArithmeticError('Analytic transmission identity failed')
    results=[solve(n,folder/f'n{n}') for n in a.sizes]
    passed=results[-1]['relative_l2_error_quadrature']<results[0]['relative_l2_error_quadrature'] and results[-1]['relative_functional_error']<results[0]['relative_functional_error']
    record={'passed_error_reduction':passed,'results':results,'radius_m':RADIUS,'domain_length_m':1.0,'conductivities':{'matrix':K_OUT,'sphere':K_IN},
        'analytic_solution':'outside: -z + beta R^3 z/r^3; inside: -gamma z; centered coordinates',
        'scope':'classical scalar-conduction verification only, with voxel geometry error; not a battery discharge or universal PDE certificate'}
    (folder/'validation.json').write_text(json.dumps(record,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');h=[1/r['size'] for r in results]
    for key,label in [('relative_l2_error_quadrature','Potential L2 (quadrature)'),('relative_gradient_error_quadrature','Gradient (quadrature)')]:axes[0].loglog(h,[r[key] for r in results],'o-',label=label)
    axes[0].set(xlabel='Voxel spacing / cube length',ylabel='Relative error');axes[0].legend();axes[0].grid(True,which='both',alpha=.3)
    axes[1].loglog(h,[r['relative_functional_error'] for r in results],'o-');axes[1].set(xlabel='Voxel spacing / cube length',ylabel='Relative Dirichlet functional error');axes[1].grid(True,which='both',alpha=.3)
    fig.savefig(folder/'convergence.png',dpi=180);fig.savefig(folder/'convergence.svg');plt.close(fig)
    picture=base64.b64encode((folder/'convergence.png').read_bytes()).decode()
    doc='<html><meta charset="utf-8"><title>MORQEX scalar transmission verification</title><body style="font:16px system-ui;max-width:1200px;margin:30px auto"><h1>Spherical conduction transmission / 球形夹杂导电验证</h1><p>Two conductivities, analytic boundary potential, exact potential and normal-flux continuity. 解析边界、电势连续和法向通量连续，独立检查有限元解。此处是数学验证，并非NMC电池放电。</p><p>Errors include voxel geometry and P1 approximation. Four-point quadrature estimates are not rigorous bounds; this does not qualify every full electrode mesh. 体素几何误差与P1误差同时存在，不是全电极PDE认证。</p><img style="width:100%" src="data:image/png;base64,'+picture+'"><p><a href="validation.json">Complete numerical data / 完整数值数据</a></p><pre>'+json.dumps(record,indent=2)+'</pre></body></html>'
    (folder/'index.html').write_text(doc)
    if not passed:raise ArithmeticError('Refinement did not reduce the selected errors; inspect source geometry and discretization separately')


if __name__=='__main__':main()
