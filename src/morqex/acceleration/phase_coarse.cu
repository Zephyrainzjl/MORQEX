__device__ __forceinline__ long long adj(long long i,int a){
    long long s=a==0?1:a==1?NX:(long long)NX*NY;int n=a==0?NX:a==1?NY:NZ;int q=(i/s)%n;
    return i+(q==n-1?-(n-1)*s:s);
}
extern "C" __global__ void count_pairs(const unsigned char* phase,const unsigned char* nonzero,unsigned int* count,int a,int b){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)NX*NY*NZ;if(i>=n || !nonzero[phase[i]])return;
    int x=i%NX,y=(i/NX)%NY,z=i/((long long)NX*NY),cx=(NX+BX-1)/BX,cy=(NY+BX-1)/BX;
    int cell=((z/BX)*cy+y/BX)*cx+x/BX;int q[3]={x,y,z},dims[3]={NX,NY,NZ};
    int flags=((q[a]%BX==BX-1 || q[a]==dims[a]-1)?1:0)+((q[b]%BX==BX-1 || q[b]==dims[b]-1)?2:0);
    long long key=(((long long)cell*4+flags)*NPHASE+phase[i])*NPHASE*NPHASE+phase[adj(i,a)]*NPHASE+phase[adj(i,b)];atomicAdd(count+key,1u);
}
extern "C" __global__ void restrict_block(const float* fine,double* coarse,const unsigned char* phase){
    int cx=(NX+BX-1)/BX,cy=(NY+BX-1)/BX,cz=(NZ+BX-1)/BX;
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cx*cy*cz*NPHASE)return;int p=i%NPHASE,j=i/NPHASE,x=j%cx,y=(j/cx)%cy,z=j/(cx*cy);double sum=0.;
    for(int k=0;k<BX && z*BX+k<NZ;k++)for(int b=0;b<BX && y*BX+b<NY;b++)for(int a=0;a<BX && x*BX+a<NX;a++){
        long long q=((long long)(z*BX+k)*NY+y*BX+b)*NX+x*BX+a;if(phase[q]==p)sum+=fine[q];
    }coarse[i]=sum;
}
extern "C" __global__ void prolong_block(const double* coarse,float* fine,const unsigned char* phase){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)NX*NY*NZ;if(i>=n)return;int cx=(NX+BX-1)/BX,cy=(NY+BX-1)/BX;
    int x=i%NX,y=(i/NX)%NY,z=i/((long long)NX*NY);fine[i]+=coarse[(((z/BX)*cy+y/BX)*cx+x/BX)*NPHASE+phase[i]];
}
