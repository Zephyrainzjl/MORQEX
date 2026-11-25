// Periodic harmonic-face finite-volume conductivity, dimensionless cell fields.
extern "C" {
__device__ __forceinline__ long long next(long long i,int axis,int sign){
    long long step=axis==0?1:axis==1?NX:(long long)NX*NY;
    int n=axis==0?NX:axis==1?NY:NZ;int q=(i/step)%n;
    return i+(q+sign<0?(n-1)*step:q+sign>=n?-(n-1)*step:sign*step);
}
__device__ __forceinline__ double face(unsigned char p,unsigned char q,const double* conductivity){
    double a=conductivity[p],b=conductivity[q];return a+b>0?2*a*b/(a+b):0.;
}
__global__ void matrix(const double* u,double* out,const unsigned char* p,const double* c,const double* ratio){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)NX*NY*NZ;if(i>=n)return;
    double sum=0.;for(int a=0;a<3;a++){
        long long j=next(i,a,1),k=next(i,a,-1);sum+=ratio[a]*ratio[a]*(face(p[i],p[j],c)*(u[i]-u[j])+face(p[i],p[k],c)*(u[i]-u[k]));
    }out[i]=sum;
}
__global__ void forcing(double* out,float* scaling,const unsigned char* p,const double* c,const double* ratio,int axis){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)NX*NY*NZ;if(i>=n)return;
    double d=0.;for(int a=0;a<3;a++)d+=ratio[a]*ratio[a]*(face(p[i],p[next(i,a,1)],c)+face(p[i],p[next(i,a,-1)],c));
    scaling[i]=d>0?rsqrt(d):0.;out[i]=ratio[axis]*(face(p[i],p[next(i,axis,1)],c)-face(p[i],p[next(i,axis,-1)],c));
}
__global__ void advance(double* x,double* r,const double* p,const double* Ap,double alpha,long long n){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){x[i]+=alpha*p[i];r[i]-=alpha*Ap[i];}
}
__global__ void conjugate(double* p,const float* z,double beta,long long n){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]=beta==0?(double)z[i]:(double)z[i]+beta*p[i];
}
__global__ void flux(const double* u,double* out,const unsigned char* p,const double* c,const double* ratio,int axis){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)NX*NY*NZ;if(i>=n)return;
    for(int a=0;a<3;a++)out[(long long)a*n+i]=face(p[i],p[next(i,a,1)],c)*((a==axis?1.:0.)+ratio[a]*(u[next(i,a,1)]-u[i]));
}
}
