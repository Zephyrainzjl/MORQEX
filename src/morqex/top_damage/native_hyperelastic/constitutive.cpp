// MORQEX interface to MFEM 4.10's BSD-3 NeoHookeanModel.
// W0 = g*mu/2*(Je^(-2/3)*Fe:Fe-3)
//      + K/2*(g*<Je-1>_+^2 + <Je-1>_-^2), Fe=F*A, A=Fch^-1.
// Energy is per initial material volume (no hidden det(Fch) multiplier).
// This is an isotropic finite-strain reference, NOT a calibrated contact law.
#include <mfem.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

extern "C" int mq_hyperelastic_abi() { return 2; }
extern "C" int mq_neo_batch(int n, const double *F, const double *A,
                            const double *mu, const double *K, const double *g,
                            double *energy, double *plus, double *P,
                            double *tangent, char *error, int error_size) {
  try {
    if (n < 1) throw std::invalid_argument("Nonempty batch required");
    mfem::DenseMatrix f(3), a(3), fe(3), pe(3), h(9);
    for (int e=0;e<n;++e) {
      if (!(mu[e]>0 && K[e]>0 && g[e]>0 && g[e]<=1) ||
          !std::isfinite(mu[e]+K[e]+g[e]))
        throw std::invalid_argument("Invalid shear/bulk/degradation");
      for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        f(i,j)=F[9*e+3*i+j]; a(i,j)=A[9*e+3*i+j];
        if (!std::isfinite(f(i,j)+a(i,j)))
          throw std::invalid_argument("Nonfinite deformation");
      }
      mfem::Mult(f,a,fe);
      const double je=fe.Det();
      if (!(f.Det()>1e-12 && a.Det()>1e-12 && je>1e-12))
        throw std::domain_error("Nonpositive or near-singular deformation determinant");
      mfem::NeoHookeanModel model(g[e]*mu[e], K[e]*(je>=1 ? g[e] : 1.0));
      energy[e]=model.EvalW(fe);
      mfem::NeoHookeanModel iso(mu[e],0.0);
      plus[e]=iso.EvalW(fe)+0.5*K[e]*std::pow(std::max(je-1.0,0.0),2);
      model.EvalP(fe,pe);
      for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        double s=0; for(int k=0;k<3;++k) s+=pe(i,k)*a(j,k);
        P[9*e+3*i+j]=s;
      }
      if(tangent) {
        // Rows of DS=A represent derivatives w.r.t. each column of total F.
        h=0.0; model.AssembleH(fe,a,1.0,h);
        for(int i=0;i<9;++i) for(int j=0;j<9;++j)
          tangent[81*e+9*i+j]=h(i,j); // MFEM component-major == row-major F.
      }
    }
    return 0;
  } catch(const std::exception &e) {
    if(error && error_size>0) std::snprintf(error,error_size,"%s",e.what());
    return 1;
  }
}
// Serial FP64 Tet4 tangent action. C is MFEM's consistent dP/dF tensor.
// No element matrices, global sparse tangent or parallel worker pool.
extern "C" int mq_tet_hvp(int n, int ndof, const long long *dofs,
                           const double *gradient, const double *C,
                           const double *volume, const double *v, double *out) {
  if (n<1 || ndof<1) return 1;
  std::fill(out,out+ndof,0.0);
  for(int e=0;e<n;++e) {
    const long long *index=dofs+12*e;
    const double *B=gradient+12*e;
    const double *T=C+81*e;
    double dF[9]={},dP[9]={};
    for(int a=0;a<4;++a) for(int i=0;i<3;++i) {
      const long long row=index[3*a+i];
      if(row<0 || row>=ndof) return 2;
      const double value=v[row];
      for(int j=0;j<3;++j) dF[3*i+j]+=value*B[3*a+j];
    }
    for(int i=0;i<9;++i) for(int j=0;j<9;++j) dP[i]+=T[9*i+j]*dF[j];
    for(int a=0;a<4;++a) for(int i=0;i<3;++i) {
      double force=0.0;
      for(int j=0;j<3;++j) force+=dP[3*i+j]*B[3*a+j];
      out[index[3*a+i]]+=volume[e]*force;
    }
  }
  return 0;
}
