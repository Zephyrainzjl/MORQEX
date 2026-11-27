// MORQEX C ABI; conforming tetrahedral bisection is supplied by vendored
// MFEM 4.10 (BSD-3-Clause). No replacement of the upstream algorithm.
#include <mfem.hpp>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
namespace {
struct Refined {
  std::vector<double> xyz;
  std::vector<int> tet, parent;
};
}
extern "C" {
int mq_refinement_abi() { return 1; }
void *mq_refine(int nv, int ne, const double *xyz, const int *tet,
                int nm, const int *marked, char *error, int error_size) {
  try {
    if (nv<4 || ne<1 || nm<1) throw std::invalid_argument("Nonempty mesh and marks required");
    mfem::Mesh mesh(3,nv,ne,0,3);
    for (int i=0;i<nv;++i) mesh.AddVertex(xyz+3*i);
    // Attribute is an exact parent label; no nearest-centre material mapping.
    for (int i=0;i<ne;++i) mesh.AddTet(tet+4*i,i+1);
    mesh.FinalizeTetMesh(1,1,true);
    mfem::Array<int> list(nm);
    for (int i=0;i<nm;++i) {
      if (marked[i]<0 || marked[i]>=ne) throw std::invalid_argument("Invalid marked cell");
      list[i]=marked[i];
    }
    mesh.GeneralRefinement(list,0); // conforming, including closure refinements
    auto r=std::make_unique<Refined>();
    for (int i=0;i<mesh.GetNV();++i) {
      const double *p=mesh.GetVertex(i);r->xyz.insert(r->xyz.end(),p,p+3);
    }
    for (int i=0;i<mesh.GetNE();++i) {
      if (mesh.GetElementBaseGeometry(i)!=mfem::Geometry::TETRAHEDRON)
        throw std::runtime_error("Refinement produced a non-tetrahedron");
      const int *v=mesh.GetElement(i)->GetVertices();
      r->tet.insert(r->tet.end(),v,v+4);r->parent.push_back(mesh.GetAttribute(i)-1);
    }
    return r.release();
  } catch (const std::exception &e) {
    if(error && error_size>0) std::snprintf(error,error_size,"%s",e.what());
    return nullptr;
  }
}
int mq_refined_nv(void *p) { return static_cast<Refined*>(p)->xyz.size()/3; }
int mq_refined_ne(void *p) { return static_cast<Refined*>(p)->parent.size(); }
void mq_refined_copy(void *p,double *xyz,int *tet,int *parent) {
  auto &r=*static_cast<Refined*>(p);
  std::copy(r.xyz.begin(),r.xyz.end(),xyz);std::copy(r.tet.begin(),r.tet.end(),tet);
  std::copy(r.parent.begin(),r.parent.end(),parent);
}
void mq_refined_free(void *p) { delete static_cast<Refined*>(p); }
int mq_tet_rule(int order, double *barycentric, double *weights) {
  if(order<1 || order>9) return -1;
  const auto &rule=mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON,order);
  if(barycentric && weights) for(int q=0;q<rule.GetNPoints();++q) {
    const auto &p=rule.IntPoint(q);
    barycentric[4*q]=1-p.x-p.y-p.z;barycentric[4*q+1]=p.x;
    barycentric[4*q+2]=p.y;barycentric[4*q+3]=p.z;weights[q]=6*p.weight;
  }
  return rule.GetNPoints();
}
// Generic quadrature-level matrix-free Hessian action. The constitutive
// tangent itself is evaluated by the independently verified MFEM kernel.
int mq_shape_hvp(int nq,int ndof,int nl,const long long *dofs,const double *grad,
                 const double *C,const double *volume,const double *v,double *out) {
  if(nq<1 || ndof<1 || nl<1 || nl>10) return 1;
  std::fill(out,out+ndof,0.);
  for(int q=0;q<nq;++q) {
    double df[9]={},dp[9]={};
    for(int a=0;a<nl;++a) for(int i=0;i<3;++i) {
      const auto row=dofs[q*3*nl+3*a+i];if(row<0 || row>=ndof) return 2;
      for(int j=0;j<3;++j) df[3*i+j]+=v[row]*grad[q*3*nl+3*a+j];
    }
    for(int i=0;i<9;++i) for(int j=0;j<9;++j) dp[i]+=C[81*q+9*i+j]*df[j];
    for(int a=0;a<nl;++a) for(int i=0;i<3;++i) for(int j=0;j<3;++j)
      out[dofs[q*3*nl+3*a+i]]+=volume[q]*dp[3*i+j]*grad[q*3*nl+3*a+j];
  }
  return 0;
}
int mq_p2_hvp(int ne,int nq,int ndof,const long long *dofs,const double *base_grad,
               const double *shape_gradient,const double *C,const double *volume,
               const double *v,double *out) {
  if(ne<1 || nq<1 || ndof<1) return 1;
  std::fill(out,out+ndof,0.);
  for(int e=0;e<ne;++e) {
    double u[30];for(int a=0;a<30;++a) {
      const auto row=dofs[30*e+a];if(row<0 || row>=ndof) return 2;u[a]=v[row];
    }
    for(int q=0;q<nq;++q) {
      const int point=e*nq+q;double grad[30]={},df[9]={},dp[9]={};
      for(int a=0;a<10;++a) for(int j=0;j<3;++j) for(int b=0;b<4;++b)
        grad[3*a+j]+=shape_gradient[40*q+4*a+b]*base_grad[12*e+3*b+j];
      for(int a=0;a<10;++a) for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        df[3*i+j]+=u[3*a+i]*grad[3*a+j];
      for(int i=0;i<9;++i) for(int j=0;j<9;++j) dp[i]+=C[81*point+9*i+j]*df[j];
      for(int a=0;a<10;++a) for(int i=0;i<3;++i) {
        double force=0.;for(int j=0;j<3;++j) force+=dp[3*i+j]*grad[3*a+j];
        out[dofs[30*e+3*a+i]]+=volume[point]*force;
      }
    }
  }
  return 0;
}
}
