// Positive auxiliary-space FFT preconditioning. Restriction is P transpose.
template<class T> __device__ void interpolate(const T* src,float* out,int nx,int ny,int nz,int count,int axis,long long i){
    long long dims[3]={nx,ny,nz};long long old=dims[axis];dims[axis]=count;
    long long step=axis==0?1:axis==1?dims[0]:dims[0]*dims[1];int q=(i/step)%count;
    double position=(double)q*old/count;int left=(int)position;double weight=position-left;
    long long base=(i/(step*count))*(step*old)+i%step;
    out[i]=(float)((1.-weight)*src[base+left*step]+weight*src[base+((left+1)%old)*step]);
}
extern "C" __global__ void prolong32(const float* src,float* out,int nx,int ny,int nz,int count,int axis,long long n){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)interpolate(src,out,nx,ny,nz,count,axis,i);
}
extern "C" __global__ void prolong64(const double* src,float* out,int nx,int ny,int nz,int count,int axis,long long n){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)interpolate(src,out,nx,ny,nz,count,axis,i);
}
extern "C" __global__ void restrict32(const float* src,float* out,int nx,int ny,int nz,int count,int axis,long long n){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    long long dims[3]={nx,ny,nz};long long old=dims[axis];dims[axis]=count;
    long long step=axis==0?1:axis==1?dims[0]:dims[0]*dims[1];int j=(i/step)%count;
    long long base=(i/(step*count))*(step*old)+i%step;double center=(double)j*old/count;int first=(int)floor(center);
    double sum=0.;for(int d=-3;d<=3;d++){int raw=first+d;double distance=fabs((double)raw*count/old-j);
        double weight=fmax(0.,1.-distance);int index=(raw%(int)old+(int)old)%(int)old;sum+=weight*src[base+(long long)index*step];}
    out[i]=(float)sum;
}
extern "C" __global__ void direction_mixed(double* p,const float* z,double beta,long long n){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]=(double)z[i]+beta*p[i];
}
