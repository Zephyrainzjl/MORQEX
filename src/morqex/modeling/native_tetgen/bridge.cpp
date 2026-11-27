// MORQEX ABI around the source-integrated TetGen1.6.0 (Hang Si, WIAS).
// The original TetGen source and license remain unchanged.
#include "tetgen.h"
#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <stdexcept>

extern "C" int mq_tetgen_abi() { return 1; }
extern "C" void *mq_tetgen_run(int np,const double *points,int nf,const int *faces,
  const int *sizes,const int *markers,int nr,const double *regions,int nh,
  const double *holes,const char *switches,char *error,int capacity) {
  try {
    tetgenio input;auto output=std::make_unique<tetgenio>();
    input.firstnumber=0;input.numberofpoints=np;
    input.pointlist=new REAL[3*np];std::copy(points,points+3*np,input.pointlist);
    input.numberoffacets=nf;input.facetlist=new tetgenio::facet[nf];
    // Initialize ALL facets before any possible exception (RAII destructor).
    for(int j=0;j<nf;++j)tetgenio::init(&input.facetlist[j]);
    input.facetmarkerlist=new int[nf];
    int offset=0;
    for(int j=0;j<nf;++j){
      auto &f=input.facetlist[j];f.numberofpolygons=1;
      f.polygonlist=new tetgenio::polygon[1];tetgenio::init(f.polygonlist);
      auto &p=f.polygonlist[0];p.numberofvertices=sizes[j];
      p.vertexlist=new int[sizes[j]];
      std::copy(faces+offset,faces+offset+sizes[j],p.vertexlist);offset+=sizes[j];
      input.facetmarkerlist[j]=markers[j];
    }
    input.numberofregions=nr;
    if(nr){input.regionlist=new REAL[5*nr];std::copy(regions,regions+5*nr,input.regionlist);}
    input.numberofholes=nh;
    if(nh){input.holelist=new REAL[3*nh];std::copy(holes,holes+3*nh,input.holelist);}
    std::string options(switches);tetrahedralize(options.data(),&input,output.get());
    if(output->numberofcorners!=4)throw std::runtime_error("Only linear Tet4 output is supported");
    return output.release();
  }catch(int code){
    // Meanings follow terminatetetgen() in the bundled TetGen header. In
    // particular code2 is an INTERNAL error, not proof of an invalid PLC.
    const char *cause="native failure";
    switch(code){
      case 1:cause="out of memory";break;
      case 2:cause="internal TetGen error";break;
      case 3:cause="self-intersecting input surface";break;
      case 4:cause="very small input feature";break;
      case 5:cause="very close input facets";break;
      case 10:cause="input error";break;
      case 200:cause="boundary Steiner points conflict with YY";break;
    }
    if(error&&capacity>0)std::snprintf(error,capacity,"TetGen error%d (%s); no mesh accepted, no faces removed",code,cause);
  }
   catch(const std::exception &e){if(error&&capacity>0)std::snprintf(error,capacity,"%s",e.what());}
  return nullptr;
}
extern "C" int mq_tetgen_counts(void *handle,int *counts){
  if(!handle)return 1;auto &o=*static_cast<tetgenio*>(handle);
  counts[0]=o.numberofpoints;counts[1]=o.numberoftetrahedra;
  counts[2]=o.numberoftrifaces;counts[3]=o.numberoftetrahedronattributes;return 0;
}
extern "C" int mq_tetgen_copy(void *handle,double *points,int *cells,double *attr,int *faces,int *markers){
  if(!handle)return 1;auto &o=*static_cast<tetgenio*>(handle);
  std::copy(o.pointlist,o.pointlist+3*o.numberofpoints,points);
  std::copy(o.tetrahedronlist,o.tetrahedronlist+4*o.numberoftetrahedra,cells);
  if(o.numberoftetrahedronattributes)std::copy(o.tetrahedronattributelist,
    o.tetrahedronattributelist+o.numberoftetrahedra*o.numberoftetrahedronattributes,attr);
  if(o.numberoftrifaces){std::copy(o.trifacelist,o.trifacelist+3*o.numberoftrifaces,faces);
    if(o.trifacemarkerlist)std::copy(o.trifacemarkerlist,o.trifacemarkerlist+o.numberoftrifaces,markers);
    else std::fill(markers,markers+o.numberoftrifaces,0);}
  return 0;
}
extern "C" void mq_tetgen_free(void *handle){delete static_cast<tetgenio*>(handle);}

#ifdef MORQEX_PYTHON_EXTENSION
#include <Python.h>
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_tetgen_native","Optional source-built TetGen C ABI",-1,nullptr};
PyMODINIT_FUNC PyInit__tetgen_native(){auto *m=PyModule_Create(&module);if(m)PyModule_AddStringConstant(m,"TETGEN_VERSION","1.6.0");return m;}
#endif
