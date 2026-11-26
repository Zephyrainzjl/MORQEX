#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <AMReX_REAL.H>
#include <AMReX.H>
#include <AMReX_PlotFileUtil.H>
static bool gem_used = false;
static void claim_lifecycle() {
    if (gem_used) throw std::runtime_error("Use an isolated process for each AMReX lifecycle");
    gem_used = true;
}
int gem_echem_main(int argc, char **argv);
PYBIND11_MODULE(_gem_echemamr, m) {
    m.doc() = "Native GEM coupled electrochemistry; one solve per isolated process";
    m.def("capabilities", []() {
        pybind11::dict info;
        info["real_bytes"] = sizeof(amrex::Real);
        info["morqex_reference_abi"] = 1;
        info["nonlinear_failure_is_fatal"] = true;
        info["single_level_full_field_reader"] = true;
        info["porous_cbd_dual_potential_qualified"] = false;
#ifdef AMREX_USE_CUDA
        info["cuda"] = true;
#else
        info["cuda"] = false;
#endif
#ifdef AMREX_USE_HYPRE
        info["amg"] = true;
#else
        info["amg"] = false;
#endif
        return info;
    });
    m.def("run", [](std::vector<std::string> args) {
        claim_lifecycle();
        args.insert(args.begin(), "gem-echemamr");
        std::vector<char*> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        pybind11::gil_scoped_release release;
        return gem_echem_main(static_cast<int>(args.size()), argv.data());
    });
    m.def("read_plot", [](const std::string& path, const std::vector<std::string>& variables,
                          long long maximum_voxels) {
#ifdef AMREX_USE_GPU
        throw std::runtime_error("Full-field reader qualified only for CPU build");
#endif
        if (maximum_voxels < 1 || variables.empty()) throw std::invalid_argument("Explicit field selection and voxel budget required");
        claim_lifecycle();
        int argc=1; char program[]="morqex-read-plot"; char* arguments[]={program,nullptr}; char** argv=arguments;
        amrex::Initialize(argc,argv,false);
        struct Finalize { ~Finalize() { amrex::Finalize(); } } finalizer;
        amrex::PlotFileData plot(path);
        if (plot.spaceDim()!=3 || plot.finestLevel()!=0)
            throw std::runtime_error("Reader requires a full 3D single-level reference; no hidden AMR flattening");
        const auto domain=plot.probDomain(0);
        const auto length=domain.length(); const auto low=domain.smallEnd();
        if (domain.numPts()>maximum_voxels) throw std::runtime_error("Plot exceeds explicit voxel budget");
        pybind11::dict fields;
        for (const auto& name:variables) {
            const auto& names=plot.varNames();
            if (std::find(names.begin(),names.end(),name)==names.end()) throw std::invalid_argument("Unknown plot field: "+name);
            auto values=plot.get(0,name);
            pybind11::array_t<double> array({length[2],length[1],length[0]});
            auto output=array.mutable_unchecked<3>();
            for (amrex::MFIter it(values);it.isValid();++it) {
                const auto box=it.validbox(); const auto a=values.const_array(it);
                for (int k=box.smallEnd(2);k<=box.bigEnd(2);++k)
                  for (int j=box.smallEnd(1);j<=box.bigEnd(1);++j)
                    for (int i=box.smallEnd(0);i<=box.bigEnd(0);++i)
                      output(k-low[2],j-low[1],i-low[0])=a(i,j,k,0);
            }
            fields[pybind11::str(name)]=array;
        }
        const auto spacing=plot.cellSize(0); const auto origin=plot.probLo();
        pybind11::dict result;
        result["fields"]=fields;result["time_s"]=plot.time();
        result["spacing_zyx_m"]=pybind11::make_tuple(spacing[2],spacing[1],spacing[0]);
        result["origin_zyx_m"]=pybind11::make_tuple(origin[2],origin[1],origin[0]);
        result["step"]=plot.levelStep(0);
        return result;
    },pybind11::arg("path"),pybind11::arg("variables"),pybind11::arg("maximum_voxels")=1000000);
}
