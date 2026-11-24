#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <AMReX_REAL.H>
int gem_echem_main(int argc, char **argv);
PYBIND11_MODULE(_gem_echemamr, m) {
    m.doc() = "Native GEM coupled electrochemistry; one solve per isolated process";
    m.def("capabilities", []() {
        pybind11::dict info;
        info["real_bytes"] = sizeof(amrex::Real);
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
        static bool used = false;
        if (used) throw std::runtime_error("Use gem.simulate.run for isolated solves");
        used = true;
        args.insert(args.begin(), "gem-echemamr");
        std::vector<char*> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        pybind11::gil_scoped_release release;
        return gem_echem_main(static_cast<int>(args.size()), argv.data());
    });
}
