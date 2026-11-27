// MORQEX data bridge. Scientific remeshing is unmodified MMG5.8.0 (LGPL-3+).
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <mmg/mmg3d/libmmg3d.h>
static_assert(sizeof(MMG5_int)==4,"MORQEX MMG bridge requires32-bit MMG indices");
struct MeshHandle {
  MMG5_pMesh mesh=nullptr;
  MMG5_pSol metric=nullptr;
  ~MeshHandle(){MMG3D_Free_all(MMG5_ARG_start,MMG5_ARG_ppMesh,&mesh,MMG5_ARG_ppMet,&metric,MMG5_ARG_end);}
};
static void checked(int result,const char* message){if(result!=1)throw std::runtime_error(message);}
extern "C" {
int mq_mmg_abi(){return 1;}
void* mq_mmg_run(int np,double* points,int nt,int* cells,int* regions,
                 int nf,int* faces,int* markers,const double* sizing,const int* flags,
                 char* error,int error_length){
  MeshHandle* h=new MeshHandle;
  try {
    checked(MMG3D_Init_mesh(MMG5_ARG_start,MMG5_ARG_ppMesh,&h->mesh,MMG5_ARG_ppMet,&h->metric,MMG5_ARG_end),"MMG initialization failed");
    checked(MMG3D_Set_iparameter(h->mesh,h->metric,MMG3D_IPARAM_mem,flags[0]),"MMG memory budget rejected");
    checked(MMG3D_Set_iparameter(h->mesh,h->metric,MMG3D_IPARAM_verbose,-1),"MMG verbosity rejected");
    checked(MMG3D_Set_meshSize(h->mesh,np,nt,0,nf,0,0),"MMG input exceeds memory/size budget");
    checked(MMG3D_Set_vertices(h->mesh,points,nullptr),"MMG input vertices rejected");
    checked(MMG3D_Set_tetrahedra(h->mesh,cells,regions),"MMG input tetrahedra rejected");
    if(nf)checked(MMG3D_Set_triangles(h->mesh,faces,markers),"MMG material/interface triangles rejected");
    checked(MMG3D_Set_iparameter(h->mesh,h->metric,MMG3D_IPARAM_nosurf,flags[1]),"MMG surface policy rejected");
    checked(MMG3D_Set_iparameter(h->mesh,h->metric,MMG3D_IPARAM_optim,flags[2]),"MMG optimization policy rejected");
    checked(MMG3D_Set_iparameter(h->mesh,h->metric,MMG3D_IPARAM_opnbdy,flags[3]),"MMG open-boundary policy rejected");
    const int parameters[]={MMG3D_DPARAM_hmin,MMG3D_DPARAM_hmax,MMG3D_DPARAM_hsiz,MMG3D_DPARAM_hausd,MMG3D_DPARAM_hgrad};
    for(int i=0;i<5;i++)if(sizing[i]>0)checked(MMG3D_Set_dparameter(h->mesh,h->metric,parameters[i],sizing[i]),"MMG size/geometry tolerance rejected");
    checked(MMG3D_Set_dparameter(h->mesh,h->metric,MMG3D_DPARAM_angleDetection,sizing[5]),"MMG feature angle rejected");
    int status=MMG3D_mmg3dlib(h->mesh,h->metric);
    if(status!=MMG5_SUCCESS)throw std::runtime_error(status==MMG5_LOWFAILURE?"MMG returned LOWFAILURE; incomplete adaptation is not accepted":"MMG returned STRONGFAILURE; no mesh accepted");
    return h;
  } catch(const std::exception& e){std::snprintf(error,error_length,"%s",e.what());delete h;return nullptr;}
}
int mq_mmg_counts(void* handle,int* counts){
  auto* h=static_cast<MeshHandle*>(handle);MMG5_int np,ne,nprism,nt,nquad,na;
  if(MMG3D_Get_meshSize(h->mesh,&np,&ne,&nprism,&nt,&nquad,&na)!=1||nprism||nquad)return 0;
  counts[0]=np;counts[1]=ne;counts[2]=nt;return 1;
}
int mq_mmg_copy(void* handle,double* points,int* cells,int* regions,int* faces,int* markers){
  auto* h=static_cast<MeshHandle*>(handle);
  return MMG3D_Get_vertices(h->mesh,points,nullptr,nullptr,nullptr)&&
    MMG3D_Get_tetrahedra(h->mesh,cells,regions,nullptr)&&MMG3D_Get_triangles(h->mesh,faces,markers,nullptr);
}
void mq_mmg_free(void* handle){delete static_cast<MeshHandle*>(handle);}
}
#ifdef MORQEX_PYTHON_EXTENSION
#include <Python.h>
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_mmg_native","MMG5.8.0 data bridge; use morqex.modeling APIs.",-1,nullptr};
PyMODINIT_FUNC PyInit__mmg_native(){return PyModule_Create(&module);}
#endif
