// SPDX-License-Identifier: GPL-3.0-or-later
// MORQEX adapter to the unmodified, source-integrated CGAL 6.0.1 Mesh_3.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/perturb_mesh_3.h>
#include <CGAL/exude_mesh_3.h>
#include <CGAL/Mesh_error_code.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#ifdef MORQEX_PYTHON_EXTENSION
#include <Python.h>
#endif

using K=CGAL::Exact_predicates_inexact_constructions_kernel;
struct Grid {
  uint64_t n[3],labels;double spacing[3];const float* scores;const float* sizing;
  bool coordinates(const K::Point_3& p,uint64_t lo[3],double f[3])const{
    double xyz[3]={CGAL::to_double(p.x()),CGAL::to_double(p.y()),CGAL::to_double(p.z())};
    for(int a=0;a<3;++a){
      if(xyz[a]<0||xyz[a]>n[a]*spacing[a])return false;
      double q=std::clamp(xyz[a]/spacing[a]-.5,0.,double(n[a]-1));
      lo[a]=n[a]>1?std::min(uint64_t(q),n[a]-2):0;f[a]=n[a]>1?q-lo[a]:0.;
    }return true;
  }
  double interpolate(const float* data,const uint64_t lo[3],const double f[3])const{
    double value=0.;
    for(int z=0;z<2;++z)for(int y=0;y<2;++y)for(int x=0;x<2;++x){
      auto ix=std::min(lo[0]+x,n[0]-1),iy=std::min(lo[1]+y,n[1]-1),iz=std::min(lo[2]+z,n[2]-1);
      value+=data[(iz*n[1]+iy)*n[0]+ix]*(x?f[0]:1-f[0])*(y?f[1]:1-f[1])*(z?f[2]:1-f[2]);
    }return value;
  }
  int operator()(const K::Point_3& p)const{
    uint64_t lo[3];double f[3];if(!coordinates(p,lo,f))return 0;
    int label=0;double best=-1e300;uint64_t count=n[0]*n[1]*n[2];
    for(uint64_t j=0;j<labels;++j){double value=interpolate(scores+j*count,lo,f);if(value>best){best=value;label=int(j+1);}}
    return label;
  }
};

using BaseDomain=CGAL::Labeled_mesh_domain_3<K>;
class Domain:public BaseDomain{
  std::vector<std::pair<K::Point_3,Index>> forced_;
  Grid grid_;
public:
  Domain(const Grid& grid,const double* config,CGAL::Random* random,const double* edges,uint64_t count):
    BaseDomain(grid,K::Sphere_3(K::Point_3(grid.n[0]*grid.spacing[0]/2,grid.n[1]*grid.spacing[1]/2,grid.n[2]*grid.spacing[2]/2),
      .26*(std::pow(grid.n[0]*grid.spacing[0],2)+std::pow(grid.n[1]*grid.spacing[1],2)+std::pow(grid.n[2]*grid.spacing[2],2))),
      CGAL::parameters::relative_error_bound(config[8]).p_rng(random)),grid_(grid){
    // Different rays may hit the same junction to the bisection tolerance.
    // Inserting these as distinct, almost coincident constrained seeds creates
    // artificial short edges. Remove redundant initialization samples only;
    // the continuous domain and every material score remain unchanged.
    const double separation=config[12]*std::min({grid.spacing[0],grid.spacing[1],grid.spacing[2]});
    std::map<std::array<int64_t,3>,std::vector<K::Point_3>> buckets;
    for(uint64_t i=0;i<count;++i){
      const double* e=edges+6*i;K::Point_3 a(e[0],e[1],e[2]),b(e[3],e[4],e[5]);int left=grid(a),right=grid(b);
      if(left==right)continue;
      for(int step=0;step<64;++step){auto middle=CGAL::midpoint(a,b);int label=grid(middle);
        if(label==left)a=middle;else{b=middle;right=label;}
        if(CGAL::to_double(CGAL::squared_distance(a,b))<config[8]*config[8])break;
      }
      if(left!=right){auto point=CGAL::midpoint(a,b);bool duplicate=false;std::array<int64_t,3> key{};
        if(separation>0){
          key={int64_t(std::floor(CGAL::to_double(point.x())/separation)),int64_t(std::floor(CGAL::to_double(point.y())/separation)),int64_t(std::floor(CGAL::to_double(point.z())/separation))};
          for(int x=-1;x<=1&&!duplicate;++x)for(int y=-1;y<=1&&!duplicate;++y)for(int z=-1;z<=1&&!duplicate;++z){
            auto it=buckets.find({key[0]+x,key[1]+y,key[2]+z});if(it!=buckets.end())for(const auto& other:it->second)
              if(CGAL::to_double(CGAL::squared_distance(point,other))<separation*separation){duplicate=true;break;}
          }
        }
        if(!duplicate){auto index=index_from_surface_patch_index({std::min(left,right),std::max(left,right)});point=project_outer(point,index);
          forced_.emplace_back(point,index);if(separation>0)buckets[key].push_back(point);}
      }
    }
  }
  K::Iso_cuboid_3 physical_box()const{return K::Iso_cuboid_3(0,0,0,grid_.n[0]*grid_.spacing[0],grid_.n[1]*grid_.spacing[1],grid_.n[2]*grid_.spacing[2]);}
  bool on_box(const K::Point_3& point)const{
    double xyz[3]={CGAL::to_double(point.x()),CGAL::to_double(point.y()),CGAL::to_double(point.z())};
    for(int a=0;a<3;++a)if(std::abs(xyz[a])<1e-10||std::abs(xyz[a]-grid_.n[a]*grid_.spacing[a])<1e-10)return true;
    return false;
  }
  K::Point_3 project_outer(const K::Point_3& point,const Index& index)const{
    auto pair=surface_patch_index(index);if(pair.first!=0&&pair.second!=0)return point;
    double xyz[3]={CGAL::to_double(point.x()),CGAL::to_double(point.y()),CGAL::to_double(point.z())};int chosen=0;double value=0.,best=1e300;
    for(int a=0;a<3;++a)for(double plane:{0.,grid_.n[a]*grid_.spacing[a]})if(std::abs(xyz[a]-plane)<best){chosen=a;value=plane;best=std::abs(xyz[a]-plane);}
    for(int a=0;a<3;++a)xyz[a]=std::clamp(xyz[a],0.,grid_.n[a]*grid_.spacing[a]);xyz[chosen]=value;
    return K::Point_3(xyz[0],xyz[1],xyz[2]);
  }
  struct Construct_intersection{
    BaseDomain::Construct_intersection base;const Domain* domain;
    template<class Query>Intersection operator()(const Query& query)const{
      auto hit=base(query);if(std::get<2>(hit)!=2)return hit;
      auto pair=domain->surface_patch_index(std::get<1>(hit));if(pair.first!=0&&pair.second!=0)return hit;
      // The observation box is known analytically. Preserve the intersection
      // with the actual query primitive rather than clipping a finished mesh
      // (which could flatten a tiny outside cell). Internal surfaces retain
      // the requested finite-tolerance implicit oracle.
      auto clipped=CGAL::intersection(query,domain->physical_box());std::vector<K::Point_3> candidates;
      if(clipped){if(const auto* s=std::get_if<K::Segment_3>(&*clipped)){candidates.push_back(s->source());candidates.push_back(s->target());}
        else if(const auto* p=std::get_if<K::Point_3>(&*clipped))candidates.push_back(*p);}
      double best=1e300;K::Point_3 chosen;
      for(const auto& p:candidates)if(domain->on_box(p)){double distance=CGAL::to_double(CGAL::squared_distance(p,std::get<0>(hit)));if(distance<best){best=distance;chosen=p;}}
      if(best==1e300)throw std::runtime_error("Observation-box intersection is inconsistent with its query");
      std::get<0>(hit)=domain->project_outer(chosen,std::get<1>(hit));return hit;
    }
  };
  Construct_intersection construct_intersection_object()const{return {BaseDomain::construct_intersection_object(),this};}
  struct Construct_initial_points{
    BaseDomain::Construct_initial_points base;const std::vector<std::pair<K::Point_3,Index>>* forced;const Domain* domain;
    template<class Iterator>Iterator operator()(Iterator out,int count=12)const{
      std::vector<std::pair<K::Point_3,Index>> points;base(std::back_inserter(points),count);
      for(auto& item:points){item.first=domain->project_outer(item.first,item.second);*out++=item;}
      for(const auto& item:*forced)*out++=item;return out;
    }
  };
  Construct_initial_points construct_initial_points_object()const{return {BaseDomain::construct_initial_points_object(),&forced_,this};}
};
using Tr=CGAL::Mesh_triangulation_3<Domain,CGAL::Default,CGAL::Sequential_tag>::type;
using C3t3=CGAL::Mesh_complex_3_in_triangulation_3<Tr>;
using Criteria=CGAL::Mesh_criteria_3<Tr>;

struct SizeField{
  const Grid* grid;const Domain* domain;double fallback,factor,outside;bool surface;
  double operator()(const K::Point_3& p,int dimension,const Domain::Index& index)const{
    // Flat observation-box surfaces may remain coarse; material interfaces
    // and nearby volume cells use the user-supplied graded field.
    if(surface&&dimension==2){auto pair=domain->surface_patch_index(index);if(pair.first==0||pair.second==0)return outside;}
    uint64_t lo[3];double f[3];if(!grid->sizing||!grid->coordinates(p,lo,f))return fallback;
    return std::min(fallback,factor*grid->interpolate(grid->sizing,lo,f));
  }
};

class JointShapeCriterion:public CGAL::Mesh_3::Sliver_criterion<Tr>{
  using Base=CGAL::Mesh_3::Sliver_criterion<Tr>;double target_,limit_;mutable double before_=0.,before_ratio_=0.;
public:
  using Base::operator();using Tet=Base::Tetrahedron_3;using Cells=Base::Cell_vector;
  JointShapeCriterion(double target,double limit,const Tr& tr):Base(target,tr),target_(target),limit_(limit){}
  double get_default_value()const override{return target_;}
  double get_max_value()const override{return 90.;}
  // Start at the requested objective. CGAL's incremental1-degree strategy
  // otherwise stops globally at an unmovable sub-degree junction before
  // attempting ordinary slivers in the rest of a complex image domain.
  double get_perturbation_unit()const override{return target_;}
  double ratio(const Tet& t)const{
    double shortest=1e300;for(int i=0;i<4;++i)for(int j=i+1;j<4;++j)shortest=std::min(shortest,CGAL::to_double(CGAL::squared_distance(t.vertex(i),t.vertex(j))));
    double radius=CGAL::to_double(CGAL::squared_radius(t.vertex(0),t.vertex(1),t.vertex(2),t.vertex(3)));
    if(!(shortest>0)||!(radius>0))return 1e300;
    return std::sqrt(radius/shortest);
  }
  double operator()(const Tet& t)const override{
    double angle=CGAL::to_double(CGAL::Mesh_3::minimum_dihedral_angle(t,this->tr_.geom_traits()));
    return std::min(angle,target_*limit_/ratio(t));
  }
  void before_move(const Cells& cells)const override{CGAL::Mesh_3::Min_value<JointShapeCriterion,Cells> f(*this);before_=f(cells);before_ratio_=0.;
    for(auto c:cells)before_ratio_=std::max(before_ratio_,ratio(this->tr_.tetrahedron(c)));}
  bool valid_move(const Cells& cells,bool soft=false)const override{
    // A minimum over a vertex star alone can improve the worst tiny angle
    // while damaging another cell's radius ratio. Do not introduce a new
    // radius-ratio violation in an already admissible star.
    for(auto c:cells)if(ratio(this->tr_.tetrahedron(c))>std::max(limit_,before_ratio_)*(1+1e-10))return false;
    CGAL::Mesh_3::Min_value<JointShapeCriterion,Cells> f(*this);double value=f(cells);return value>before_||(soft&&value>this->sliver_bound_);
  }
};

struct Result{std::vector<double>points;std::vector<uint32_t>cells;std::vector<int32_t>labels;std::vector<uint64_t>components;std::array<double,6>diagnostics{};};
using Callback=void(*)(int,uint64_t,uint64_t);
static std::mutex native_mutex;
extern "C" {
const char* mq_cgal_source_id(){return MORQEX_SOURCE_ID;}
void* mq_cgal_mesh(const float* scores,const float* sizing,const uint64_t* shape,const double* spacing,
    const double* config,const int64_t* limits,const double* seed_edges,uint64_t seed_count,Callback callback,char* error,int error_size){
  std::lock_guard<std::mutex> guard(native_mutex);
  try{
    Grid grid{{shape[0],shape[1],shape[2]},shape[3],{spacing[0],spacing[1],spacing[2]},scores,sizing};
    if(!scores||!grid.labels||!grid.n[0]||!grid.n[1]||!grid.n[2])throw std::runtime_error("Invalid geometry field");
    CGAL::Random random(static_cast<unsigned>(limits[2]));CGAL::get_default_random()=CGAL::Random(static_cast<unsigned>(limits[2]));
    Domain domain(grid,config,&random,seed_edges,seed_count);
    SizeField facet{&grid,&domain,config[1],config[9],config[11],true},volume{&grid,&domain,config[3],1.,config[3],false};
    Criteria criteria(CGAL::parameters::facet_angle(config[0]).facet_size(facet).facet_distance(config[2]).cell_radius_edge_ratio(config[4]).cell_size(volume));
    CGAL::Mesh_error_code code=CGAL::CGAL_MESH_3_NO_ERROR;
    // Keep ONE named-parameter chain. In CGAL6.0.1 the variadic combine helper
    // accepts a two-template-argument base node and can slice a chained pack,
    // losing no_perturb(). Passing that chain as one pack preserves all flags.
    auto opts=CGAL::parameters::mesh_3_options(CGAL::parameters::maximal_number_of_vertices(uint64_t(limits[0])).number_of_initial_points(int(limits[3])).pointer_to_error_code(&code)).no_exude().no_perturb();
    if(bool(CGAL::parameters::get_parameter(opts,CGAL::internal_np::exude_options_param))||bool(CGAL::parameters::get_parameter(opts,CGAL::internal_np::perturb_options_param)))throw std::runtime_error("Unrequested native optimization is enabled");
    if(callback)callback(0,0,0);
    auto mesh=CGAL::make_mesh_3<C3t3>(domain,criteria,opts);
    if(code!=CGAL::CGAL_MESH_3_NO_ERROR)throw std::runtime_error(CGAL::mesh_error_string(code));
    if(mesh.number_of_cells_in_complex()>uint64_t(limits[1]))throw std::runtime_error("CGAL output exceeds maximum_cells: "+std::to_string(mesh.number_of_cells_in_complex()));
    if(callback)callback(1,mesh.triangulation().number_of_vertices(),mesh.number_of_cells_in_complex());
    auto measure=[&](){std::array<double,2> q{0.,180.};JointShapeCriterion criterion(config[7],config[10],mesh.triangulation());
      for(auto c=mesh.cells_in_complex_begin();c!=mesh.cells_in_complex_end();++c){auto t=mesh.triangulation().tetrahedron(c);
        q[0]=std::max(q[0],criterion.ratio(t));q[1]=std::min(q[1],CGAL::to_double(CGAL::Mesh_3::minimum_dihedral_angle(t,mesh.triangulation().geom_traits())));}
      return q;};
    auto initial=measure(),final=initial;int repairs=0,return_code=-1;
    const int maximum_repairs=int(limits[4])==1?int(limits[5]):0;
    for(int cycle=0;cycle<=maximum_repairs;++cycle){
      if(config[5]>0){
        for(auto c=mesh.triangulation().finite_cells_begin();c!=mesh.triangulation().finite_cells_end();++c)c->reset_cache_validity();
        if(int(limits[4])==1){JointShapeCriterion criterion(config[7],config[10],mesh.triangulation());
          return_code=int(CGAL::perturb_mesh_3(mesh,domain,CGAL::parameters::time_limit(config[5]).sliver_criterion(criterion)));}
        else return_code=int(CGAL::perturb_mesh_3(mesh,domain,CGAL::parameters::time_limit(config[5]).sliver_bound(config[7])));
        if(callback)callback(2,mesh.triangulation().number_of_vertices(),mesh.number_of_cells_in_complex());
      }
      final=measure();
      if((final[0]<=config[10]*(1+1e-7)&&final[1]+1e-7>=config[13])||cycle==maximum_repairs||config[5]<=0)break;
      // Reuse the existing triangulation. Optimize-then-refine is necessary:
      // CGAL explicitly documents that optimization may violate refinement
      // criteria. Restore those criteria by adaptive refinement, rather than
      // throwing away the mesh and rebuilding all good cells.
      Criteria repair_criteria(CGAL::parameters::facet_angle(config[0]).facet_size(facet).facet_distance(config[2]).cell_radius_edge_ratio(config[10]*.98).cell_size(volume));
      code=CGAL::CGAL_MESH_3_NO_ERROR;
      CGAL::refine_mesh_3(mesh,domain,repair_criteria,opts);
      if(code!=CGAL::CGAL_MESH_3_NO_ERROR)throw std::runtime_error(CGAL::mesh_error_string(code));
      if(mesh.number_of_cells_in_complex()>uint64_t(limits[1]))throw std::runtime_error("Local quality refinement exceeds maximum_cells");
      ++repairs;if(callback)callback(10+repairs,mesh.triangulation().number_of_vertices(),mesh.number_of_cells_in_complex());
    }
    if(config[6]>0){CGAL::exude_mesh_3(mesh,CGAL::parameters::time_limit(config[6]).sliver_bound(config[7]));
      if(callback)callback(3,mesh.triangulation().number_of_vertices(),mesh.number_of_cells_in_complex());}
    if(mesh.number_of_cells_in_complex()>uint64_t(limits[1]))throw std::runtime_error("Optimized CGAL mesh exceeds maximum_cells");
    auto result=std::make_unique<Result>();result->components.assign(grid.labels,0);
    if(config[6]>0)final=measure();
    result->diagnostics={initial[0],initial[1],final[0],final[1],double(repairs),double(return_code)};
    std::unordered_set<const void*> visited;
    for(auto c=mesh.cells_in_complex_begin();c!=mesh.cells_in_complex_end();++c){
      Tr::Cell_handle root=c;if(!visited.insert(&*root).second)continue;int label=mesh.subdomain_index(root);
      if(label<1||uint64_t(label)>grid.labels)throw std::runtime_error("Invalid material label");++result->components[label-1];
      std::vector<Tr::Cell_handle> stack{root};
      while(!stack.empty()){auto cell=stack.back();stack.pop_back();for(int j=0;j<4;++j){auto other=cell->neighbor(j);
        if(!mesh.triangulation().is_infinite(other)&&mesh.is_in_complex(other)&&mesh.subdomain_index(other)==label&&visited.insert(&*other).second)stack.push_back(other);}}
    }
    visited.clear();visited.rehash(0);
    std::map<Tr::Vertex_handle,uint32_t> ids;
    for(auto c=mesh.cells_in_complex_begin();c!=mesh.cells_in_complex_end();++c){
      for(int j=0;j<4;++j){auto v=c->vertex(j);auto it=ids.find(v);
        if(it==ids.end()){uint32_t id=uint32_t(ids.size());ids[v]=id;auto p=v->point();
          result->points.insert(result->points.end(),{CGAL::to_double(p.x()),CGAL::to_double(p.y()),CGAL::to_double(p.z())});result->cells.push_back(id);}
        else result->cells.push_back(it->second);}
      result->labels.push_back(mesh.subdomain_index(c));
    }
    if(callback)callback(4,ids.size(),result->labels.size());return result.release();
  }catch(const std::exception& e){if(error&&error_size>0){std::strncpy(error,e.what(),error_size-1);error[error_size-1]=0;}return nullptr;}
  catch(...){if(error&&error_size>0){std::strncpy(error,"CGAL native failure",error_size-1);error[error_size-1]=0;}return nullptr;}
}
int mq_cgal_counts(void* handle,uint64_t* counts){if(!handle||!counts)return 0;auto r=static_cast<Result*>(handle);counts[0]=r->points.size()/3;counts[1]=r->labels.size();counts[2]=r->components.size();return 1;}
int mq_cgal_diagnostics(void* handle,double* values){if(!handle||!values)return 0;auto r=static_cast<Result*>(handle);std::copy(r->diagnostics.begin(),r->diagnostics.end(),values);return 1;}
int mq_cgal_copy(void* handle,double* p,uint32_t* c,int32_t* r,uint64_t* components){if(!handle)return 0;auto v=static_cast<Result*>(handle);
  std::copy(v->points.begin(),v->points.end(),p);std::copy(v->cells.begin(),v->cells.end(),c);std::copy(v->labels.begin(),v->labels.end(),r);std::copy(v->components.begin(),v->components.end(),components);return 1;}
void mq_cgal_free(void* handle){delete static_cast<Result*>(handle);}
}
#ifdef MORQEX_PYTHON_EXTENSION
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_cgal_native",nullptr,-1,nullptr};
PyMODINIT_FUNC PyInit__cgal_native(){return PyModule_Create(&module);}
#endif
