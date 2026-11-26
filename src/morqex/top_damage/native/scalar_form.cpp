// MORQEX adapter; finite element integrators are supplied by MFEM (BSD-3).
// Explicit vertex permutation keeps the public API independent of MFEM DOFs.
#include <mfem.hpp>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
class ElementScalar : public mfem::Coefficient {
  const double *v;
public:
  explicit ElementScalar(const double *values) : v(values) {}
  double Eval(mfem::ElementTransformation &t, const mfem::IntegrationPoint &) override {
    return v[t.ElementNo];
  }
};
class ElementTensor : public mfem::MatrixCoefficient {
  const double *v;
public:
  explicit ElementTensor(const double *values) : MatrixCoefficient(3), v(values) {}
  void Eval(mfem::DenseMatrix &m, mfem::ElementTransformation &t,
            const mfem::IntegrationPoint &) override {
    m.SetSize(3);
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j)
      m(i,j)=v[9*t.ElementNo+3*i+j];
  }
};
struct Form {
  std::vector<int> rows, columns;
  std::vector<double> values, rhs;
};
}
extern "C" {
int mq_mfem_version() { return MFEM_VERSION; }
void *mq_scalar_form(int nv, int ne, const double *xyz, const int *tet,
                     const double *reaction, const double *diffusion,
                     const double *source, char *error, int error_size) {
  try {
    if (nv<4 || ne<1) throw std::invalid_argument("Empty tetrahedral mesh");
    mfem::Mesh mesh(3,nv,ne,0,3);
    for (int i=0;i<nv;++i) mesh.AddVertex(xyz+3*i);
    for (int i=0;i<ne;++i) mesh.AddTet(tet+4*i);
    mesh.FinalizeTetMesh(1,0,true);
    mfem::H1_FECollection fec(1,3);
    mfem::FiniteElementSpace space(&mesh,&fec);
    if (space.GetVSize()!=nv) throw std::runtime_error("Unexpected P1 space size");
    ElementScalar a(reaction), f(source);
    ElementTensor D(diffusion);
    mfem::BilinearForm form(&space);
    form.AddDomainIntegrator(new mfem::MassIntegrator(a));
    form.AddDomainIntegrator(new mfem::DiffusionIntegrator(D));
    form.Assemble(); form.Finalize();
    mfem::LinearForm rhs(&space);
    rhs.AddDomainIntegrator(new mfem::DomainLFIntegrator(f)); rhs.Assemble();
    std::vector<int> vertex_dof(nv), dof_vertex(nv,-1);
    mfem::Array<int> vd;
    for (int v=0;v<nv;++v) {
      space.GetVertexDofs(v,vd);
      if (vd.Size()!=1 || vd[0]<0 || vd[0]>=nv || dof_vertex[vd[0]]!=-1)
        throw std::runtime_error("Invalid scalar vertex DOF map");
      vertex_dof[v]=vd[0]; dof_vertex[vd[0]]=v;
    }
    auto result=std::make_unique<Form>();
    auto &A=form.SpMat(); const int *I=A.GetI(), *J=A.GetJ(); const double *V=A.GetData();
    result->rows.push_back(0);
    for (int v=0;v<nv;++v) {
      int row=vertex_dof[v]; result->rhs.push_back(rhs[row]);
      for (int k=I[row];k<I[row+1];++k) {
        result->columns.push_back(dof_vertex[J[k]]); result->values.push_back(V[k]);
      }
      result->rows.push_back(static_cast<int>(result->values.size()));
    }
    return result.release();
  } catch (const std::exception &e) {
    if (error && error_size>0) std::snprintf(error,error_size,"%s",e.what());
    return nullptr;
  } catch (...) {
    if (error && error_size>0) std::snprintf(error,error_size,"Unknown MFEM failure");
    return nullptr;
  }
}
int mq_form_nnz(void *p) { return static_cast<int>(static_cast<Form*>(p)->values.size()); }
void mq_form_copy(void *p,int *I,int *J,double *V,double *b) {
  auto &f=*static_cast<Form*>(p);
  std::copy(f.rows.begin(),f.rows.end(),I); std::copy(f.columns.begin(),f.columns.end(),J);
  std::copy(f.values.begin(),f.values.end(),V); std::copy(f.rhs.begin(),f.rhs.end(),b);
}
void mq_form_free(void *p) { delete static_cast<Form*>(p); }
}
