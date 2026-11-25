extern "C" __global__ void matrix32(const float* u,float* out,const unsigned char* p,const double* c,const double* ratio){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)NX*NY*NZ;if(i>=n)return;
    double sum=0.;for(int a=0;a<3;a++){
        long long j=next(i,a,1),k=next(i,a,-1);sum+=ratio[a]*ratio[a]*(face(p[i],p[j],c)*(u[i]-u[j])+face(p[i],p[k],c)*(u[i]-u[k]));
    }out[i]=sum;
}
extern "C" __global__ void restrict_block(const float* fine,double* coarse){
    int cx=(NX+BX-1)/BX,cy=(NY+BX-1)/BX,cz=(NZ+BX-1)/BX;
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cx*cy*cz)return;int x=i%cx,y=(i/cx)%cy,z=i/(cx*cy);double sum=0.;
    for(int k=0;k<BX && z*BX+k<NZ;k++)for(int j=0;j<BX && y*BX+j<NY;j++)for(int a=0;a<BX && x*BX+a<NX;a++)
        sum+=fine[((long long)(z*BX+k)*NY+y*BX+j)*NX+x*BX+a];
    coarse[i]=sum;
}
extern "C" __global__ void prolong_block(const double* coarse,float* fine){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)NX*NY*NZ;if(i>=n)return;int cx=(NX+BX-1)/BX,cy=(NY+BX-1)/BX;
    int x=i%NX,y=(i/NX)%NY,z=i/((long long)NX*NY);fine[i]+=coarse[((z/BX)*cy+y/BX)*cx+x/BX];
}
