// Compatible periodic finite differences. Same FP64 stencil on CPU and CUDA.
// Energy: mean .5*(E + sym D u - beta*c - alpha*T):C:(...).
// Kelvin order: xx yy zz yz xz xy. No weak material is added to pore cells.
#ifdef MORQEX_FLOAT
typedef float Real;
#else
typedef double Real;
#endif
#ifdef MORQEX_NX
#define DIMS nx=MORQEX_NX;ny=MORQEX_NY;nz=MORQEX_NZ;
#else
#define DIMS
#endif
#ifdef __CUDACC__
#define INLINE __device__ __forceinline__
#define KERNEL extern "C" __global__ void
#define LOOP(n) long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)
#else
#define INLINE inline
#define KERNEL extern "C" void
#define LOOP(n) for(long long i=0;i<n;i++)
#endif
INLINE long long neighbor(long long i,int a,int sign,int nx,int ny,int nz){
    long long step=a==0?1:a==1?nx:(long long)nx*ny;
    int length=a==0?nx:a==1?ny:nz;int coordinate=(i/step)%length;
    return i+(coordinate+sign<0?(length-1)*step:coordinate+sign>=length?-(length-1)*step:sign*step);
}
INLINE void stress_at(long long i,const Real* u,const unsigned char* phase,const double* C,
    const double* beta,const double* alpha,const double* concentration,const double* E,double T,
    int source,int displacement,int nx,int ny,int nz,const double* ratios,double* s){
    const long long n=(long long)nx*ny*nz;const double root=0.7071067811865475244;
    double g[9]={0.,0.,0.,0.,0.,0.,0.,0.,0.};
    if(displacement)for(int a=0;a<3;a++){long long j=neighbor(i,a,1,nx,ny,nz);
        for(int k=0;k<3;k++)g[3*k+a]=(u[(long long)k*n+j]-u[(long long)k*n+i])*ratios[a];}
    double e[6]={g[0],g[4],g[8],root*(g[5]+g[7]),root*(g[2]+g[6]),root*(g[1]+g[3])};
    int p=phase[i];if(source)for(int k=0;k<6;k++){e[k]+=E[k];if(source&1)e[k]-=beta[p*6+k]*concentration[i]+alpha[p*6+k]*T;}
    const double* c=C+p*36;
    // All phase tensors, including anisotropic ones, use the same contraction.
#ifdef MORQEX_ISOTROPIC
    double tr=e[0]+e[1]+e[2];double l=c[1],m2=c[21];
    for(int k=0;k<3;k++)s[k]=l*tr+m2*e[k];
    for(int k=3;k<6;k++)s[k]=m2*e[k];
#else
    for(int k=0;k<6;k++){double v=0.;for(int j=0;j<6;j++)v+=c[k*6+j]*e[j];s[k]=v;}
#endif
}
KERNEL elastic(const Real* u,Real* out,const unsigned char* phase,const double* C,
    int nx,int ny,int nz,const double* ratios){
    DIMS long long n=(long long)nx*ny*nz;LOOP(n){double here[6],before[6];
        stress_at(i,u,phase,C,nullptr,nullptr,nullptr,nullptr,0.,0,1,nx,ny,nz,ratios,here);
        double f[3]={0.,0.,0.};int indices[9]={0,5,4,5,1,3,4,3,2};
        for(int a=0;a<3;a++){long long j=neighbor(i,a,-1,nx,ny,nz);
            stress_at(j,u,phase,C,nullptr,nullptr,nullptr,nullptr,0.,0,1,nx,ny,nz,ratios,before);
            for(int k=0;k<3;k++){int index=indices[3*k+a];f[k]+=(before[index]-here[index])*ratios[a]*(k==a?1.:0.7071067811865475244);}}
        for(int k=0;k<3;k++)out[(long long)k*n+i]=f[k];
    }
}
KERNEL source(Real* out,const unsigned char* phase,const double* C,const double* beta,
    const double* alpha,const double* concentration,const double* E,double T,int nx,int ny,int nz,const double* ratios){
    DIMS long long n=(long long)nx*ny*nz;LOOP(n){double here[6],before[6];
        stress_at(i,nullptr,phase,C,beta,alpha,concentration,E,T,1,0,nx,ny,nz,ratios,here);
        double f[3]={0.,0.,0.};int indices[9]={0,5,4,5,1,3,4,3,2};
        for(int a=0;a<3;a++){long long j=neighbor(i,a,-1,nx,ny,nz);
            stress_at(j,nullptr,phase,C,beta,alpha,concentration,E,T,1,0,nx,ny,nz,ratios,before);
            for(int k=0;k<3;k++){int index=indices[3*k+a];f[k]+=(here[index]-before[index])*ratios[a]*(k==a?1.:0.7071067811865475244);}}
        for(int k=0;k<3;k++)out[(long long)k*n+i]=f[k];
    }
}
KERNEL residual(const Real* u,Real* out,const unsigned char* phase,const double* C,const double* beta,
    const double* alpha,const double* concentration,const double* E,double T,int nx,int ny,int nz,const double* ratios){
    DIMS long long n=(long long)nx*ny*nz;LOOP(n){double here[6],before[6];
        stress_at(i,u,phase,C,beta,alpha,concentration,E,T,1,1,nx,ny,nz,ratios,here);
        double f[3]={0.,0.,0.};int indices[9]={0,5,4,5,1,3,4,3,2};
        for(int a=0;a<3;a++){long long j=neighbor(i,a,-1,nx,ny,nz);
            stress_at(j,u,phase,C,beta,alpha,concentration,E,T,1,1,nx,ny,nz,ratios,before);
            for(int k=0;k<3;k++){int index=indices[3*k+a];f[k]+=(here[index]-before[index])*ratios[a]*(k==a?1.:0.7071067811865475244);}}
        for(int k=0;k<3;k++)out[(long long)k*n+i]=f[k];
    }
}
KERNEL stress(const Real* u,Real* out,const unsigned char* phase,const double* C,const double* beta,
    const double* alpha,const double* concentration,const double* E,double T,int nx,int ny,int nz,const double* ratios){
    DIMS long long n=(long long)nx*ny*nz;LOOP(n){double s[6];stress_at(i,u,phase,C,beta,alpha,concentration,E,T,1,1,nx,ny,nz,ratios,s);
        for(int k=0;k<6;k++)out[(long long)k*n+i]=s[k];}
}
KERNEL update(Real* x,Real* r,const Real* p,const Real* Ap,double a,long long n){LOOP(n){x[i]+=a*p[i];r[i]-=a*Ap[i];}}
KERNEL direction(Real* p,const Real* z,double b,long long n){LOOP(n){p[i]=z[i]+b*p[i];}}
KERNEL elastic_macro(const Real* u,Real* out,const unsigned char* phase,const double* C,const double* E,
    int nx,int ny,int nz,const double* ratios){
    DIMS long long n=(long long)nx*ny*nz;LOOP(n){double here[6],before[6];
        stress_at(i,u,phase,C,nullptr,nullptr,nullptr,E,0.,2,1,nx,ny,nz,ratios,here);
        double f[3]={0.,0.,0.};int indices[9]={0,5,4,5,1,3,4,3,2};
        for(int a=0;a<3;a++){long long j=neighbor(i,a,-1,nx,ny,nz);
            stress_at(j,u,phase,C,nullptr,nullptr,nullptr,E,0.,2,1,nx,ny,nz,ratios,before);
            for(int k=0;k<3;k++){int index=indices[3*k+a];f[k]+=(before[index]-here[index])*ratios[a]*(k==a?1.:0.7071067811865475244);}}
        for(int k=0;k<3;k++)out[(long long)k*n+i]=f[k];
    }
}
KERNEL mean_stress(const Real* u,double* out,const unsigned char* phase,const double* C,const double* beta,
    const double* alpha,const double* concentration,const double* E,double T,int source_flag,int displacement,
    int nx,int ny,int nz,const double* ratios){
    DIMS long long n=(long long)nx*ny*nz;
#ifdef __CUDACC__
    __shared__ double buffer[6][128];long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;double s[6]={0.,0.,0.,0.,0.,0.};
    if(i<n)stress_at(i,u,phase,C,beta,alpha,concentration,E,T,source_flag,displacement,nx,ny,nz,ratios,s);
    for(int k=0;k<6;k++)buffer[k][threadIdx.x]=s[k];__syncthreads();
    for(int step=64;step;step/=2){if(threadIdx.x<step)for(int k=0;k<6;k++)buffer[k][threadIdx.x]+=buffer[k][threadIdx.x+step];__syncthreads();}
    if(threadIdx.x==0)for(int k=0;k<6;k++)out[(long long)k*gridDim.x+blockIdx.x]=buffer[k][0];
#else
    for(int k=0;k<6;k++)out[k]=0.;for(long long i=0;i<n;i++){double s[6];stress_at(i,u,phase,C,beta,alpha,concentration,E,T,source_flag,displacement,nx,ny,nz,ratios,s);for(int k=0;k<6;k++)out[k]+=s[k];}
#endif
}
