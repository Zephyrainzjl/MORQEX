// Each block belongs to a state/voxel/component entry; states may reference
// different immutable geometries. FP64 sums, no atomics or state mixing.
extern "C" __global__ void morqex_node_fields(
 const double* direct,const double* modes,const double* q,const double* z,
 const int* geometry,double* out,int batch,int outputs,int nq,int nr) {
 long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 long long total=(long long)batch*outputs;
 if(k>=total)return;
 int state=k/outputs,entry=k%outputs,g=geometry[state];double value=0.;
 const double* a=direct+(long long)g*outputs*nq+entry;
 const double* b=modes+(long long)g*outputs*nr+entry;
 // Adjacent threads read adjacent output entries at every input column.
 for(int j=0;j<nq;++j)value+=a[(long long)j*outputs]*q[state*nq+j];
 for(int j=0;j<nr;++j)value+=b[(long long)j*outputs]*z[state*nr+j];
 out[k]=value;
}
