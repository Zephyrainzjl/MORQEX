#include <echemAMR.H>

#include <Chemistry.H>
#include <Transport.H>
#include <Reactions.H>
#include <bv_utils.H>
#include <cmath>

void echemAMR::solve_potential(Real current_time)
{
    BL_PROFILE("echemAMR::solve_potential()");
    LPInfo info;

    // FIXME: add these as inputs
    int agglomeration = 1;
    int consolidation = 1;
    int max_coarsening_level = linsolve_max_coarsening_level;
    bool semicoarsening = false;
    int max_semicoarsening_level = 0;
    int linop_maxorder = 2;
    int max_fmg_iter = 0;
    int verbose = 1;
    int bottom_verbose = 0;
    int morqex_potential_offset = 0;
    Real morqex_linear_budget_fraction = 0.0;
    amrex::ParmParse morqex_options("echemamr");
    morqex_options.query("morqex_potential_offset",morqex_potential_offset);
    morqex_options.query("morqex_linear_budget_fraction",morqex_linear_budget_fraction);
    if (!std::isfinite(morqex_linear_budget_fraction) || morqex_linear_budget_fraction<0.0 ||
        morqex_linear_budget_fraction>0.25 || (morqex_linear_budget_fraction>0.0 && finest_level!=0))
        amrex::Abort("MORQEX: linear residual budgeting requires one level and fraction in [0,0.25]");
    Real ascalar = 0.0;
    Real bscalar = 1.0;
    int num_nonlinear_iters;

    //FIXME:use get_grads_and_jumps function from Src3d for BVflux

    //==================================================
    // amrex solves
    // read small a as alpha, b as beta

    //(A a - B del.(b del)) phi = f
    //
    // A and B are scalar constants
    // a and b are scalar fields
    // f is rhs
    // in this case: A=0,a=0,B=1,b=conductivity
    // note also the negative sign
    //====================================================

    // FIXME: had to adjust for constant coefficent,
    // this could be due to missing terms in the 
    // intercalation reaction or sign mistakes...
    ProbParm const* localprobparm = d_prob_parm;

    const Real tol_rel = linsolve_reltol;
    const Real tol_abs = linsolve_abstol;

#ifdef AMREX_USE_HYPRE
    if(use_hypre)
    {
        amrex::Print()<<"using hypre\n";
    }
#endif
    info.setAgglomeration(agglomeration);
    info.setConsolidation(consolidation);
    info.setSemicoarsening(semicoarsening);
    info.setMaxCoarseningLevel(max_coarsening_level);
    info.setMaxSemicoarseningLevel(max_semicoarsening_level);


    // set A and B, A=0, B=1
    //
    if (buttler_vohlmer_flux)
    {
        ascalar = bv_relaxfac;
        num_nonlinear_iters=bv_nonlinear_iters;
    } 
    else
    {
        ascalar = 0.0;
        num_nonlinear_iters=1;
    }

    // default to inhomogNeumann since it is defaulted to flux = 0.0 anyways
    std::array<LinOpBCType, AMREX_SPACEDIM> bc_potsolve_lo 
        = {LinOpBCType::inhomogNeumann, LinOpBCType::inhomogNeumann, LinOpBCType::inhomogNeumann};

    std::array<LinOpBCType, AMREX_SPACEDIM> bc_potsolve_hi 
        = {LinOpBCType::inhomogNeumann, LinOpBCType::inhomogNeumann, LinOpBCType::inhomogNeumann};

    bool mixedbc=false;

    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
    {
        if (bc_lo_pot[idim] == BCType::int_dir)
        {
            bc_potsolve_lo[idim] = LinOpBCType::Periodic;
        }
        if (bc_lo_pot[idim] == BCType::ext_dir)
        {
            bc_potsolve_lo[idim] = LinOpBCType::Dirichlet;
        }
        if (bc_lo_pot[idim] == BCType::foextrap)
        {
            bc_potsolve_lo[idim] = LinOpBCType::Neumann;
        }
        if (bc_lo_pot[idim] == BCType::hoextrapcc)
        {
            bc_potsolve_lo[idim] = LinOpBCType::Robin;
            mixedbc=true;
        }

        if (bc_hi_pot[idim] == BCType::int_dir)
        {
            bc_potsolve_hi[idim] = LinOpBCType::Periodic;
        }
        if (bc_hi_pot[idim] == BCType::ext_dir)
        {
            bc_potsolve_hi[idim] = LinOpBCType::Dirichlet;
        }
        if (bc_hi_pot[idim] == BCType::foextrap)
        {
            bc_potsolve_hi[idim] = LinOpBCType::Neumann;
        }
        if (bc_hi_pot[idim] == BCType::hoextrapcc)
        {
            bc_potsolve_hi[idim] = LinOpBCType::Robin;
            mixedbc=true;
        }
    }


    const Real cathode_reference = phi_new[0].max(POT_ID,0,false);
    Vector<MultiFab> potential(finest_level+1);
    Vector<MultiFab> potential_reference(finest_level+1);
    Vector<MultiFab> fluctuation(finest_level+1);
    Vector<MultiFab> acoeff(finest_level+1);
    Vector<MultiFab> bcoeff(finest_level+1);
    Vector<Array<MultiFab, AMREX_SPACEDIM>> gradsoln(finest_level+1);
    Vector<MultiFab> solution(finest_level+1);
    Vector<MultiFab> residual(finest_level+1);
    Vector<MultiFab> rhs(finest_level+1);
    Vector<MultiFab> err(finest_level+1);
    Vector<MultiFab> rhs_res(finest_level+1);

    Vector<MultiFab> robin_a(finest_level+1);
    Vector<MultiFab> robin_b(finest_level+1);
    Vector<MultiFab> robin_f(finest_level+1);

    const int num_grow = 2;
    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        potential[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        potential_reference[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        fluctuation[ilev].define(grids[ilev], dmap[ilev], 1, 0);
        acoeff[ilev].define(grids[ilev], dmap[ilev], 1, 0);
        bcoeff[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        solution[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        residual[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        rhs[ilev].define(grids[ilev], dmap[ilev], 1, 0);
        err[ilev].define(grids[ilev], dmap[ilev], 1, 0);
        rhs_res[ilev].define(grids[ilev], dmap[ilev], 1, 0);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& faceba = amrex::convert(grids[ilev], 
                    IntVect::TheDimensionVector(idim));
            gradsoln[ilev][idim].define(faceba, dmap[ilev], 1, 0);
        }

        robin_a[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        robin_b[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        robin_f[ilev].define(grids[ilev], dmap[ilev], 1, 1);
    }

    Real errnorm_1st_iter = 0.0;
    // MORQEX: an exhausted nonlinear loop must never publish an accepted state.
    if (num_nonlinear_iters < 1) amrex::Abort("MORQEX: positive potential iteration budget required");
    if (buttler_vohlmer_flux) amrex::Print() << "MORQEX BV BEGIN time=" << current_time << "\n";
    
    MLABecLaplacian mlabec(Geom(0, finest_level), boxArray(0, finest_level), 
                           DistributionMap(0, finest_level), info);
    MLABecLaplacian mlabec_res(Geom(0,finest_level), boxArray(0,finest_level), 
                               DistributionMap(0, finest_level), info);

    mlabec.setMaxOrder(linop_maxorder);
    mlabec_res.setMaxOrder(linop_maxorder);
    
    for (int nl_it = 0; nl_it < num_nonlinear_iters; ++nl_it)
    {
        mlabec.setScalars(ascalar, bscalar);
        mlabec_res.setScalars(0.0, bscalar);
    
        mlabec.setDomainBC(bc_potsolve_lo, bc_potsolve_hi);
        mlabec_res.setDomainBC(bc_potsolve_lo, bc_potsolve_hi);
    
        MLMG mlmg(mlabec);
        mlmg.setMaxIter(linsolve_maxiter);
        mlmg.setMaxFmgIter(max_fmg_iter);
        mlmg.setVerbose(verbose);
        mlmg.setBottomVerbose(bottom_verbose);
        mlmg.setBottomTolerance(linsolve_bot_reltol);
        mlmg.setBottomToleranceAbs(linsolve_bot_abstol);
        mlmg.setBottomMaxIter(linsolve_bottom_maxiter);

        mlmg.setPreSmooth(linsolve_num_pre_smooth);
        mlmg.setPostSmooth(linsolve_num_post_smooth);
        mlmg.setFinalSmooth(linsolve_num_final_smooth);
        mlmg.setBottomSmooth(linsolve_num_bottom_smooth);

#ifdef AMREX_USE_HYPRE
        if (use_hypre)
        {
            mlmg.setHypreOptionsNamespace("echemamr.hypre");
            mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
        }
#endif
#ifdef AMREX_USE_PETSC
        if (use_petsc)
        {
            mlmg.setBottomSolver(MLMG::BottomSolver::petsc);
        }
#endif

        MLMG mlmg_res(mlabec_res);
        
        for (int ilev = 0; ilev <= finest_level; ilev++)
        {
            MultiFab Sborder(grids[ilev], dmap[ilev], phi_new[ilev].nComp(), num_grow);
            FillPatch(ilev, current_time, Sborder, 0, Sborder.nComp());
            potential[ilev].setVal(0.0);

            // Copy (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, 
            // int dstcomp, int numcomp, const IntVect& nghost)
            amrex::Copy(potential[ilev], Sborder, POT_ID, 0, 1, num_grow);
            potential_reference[ilev].setVal(0.0);
            if (morqex_potential_offset) {
#if defined(C_AM_ID) && defined(C_CBD_ID)
                for (MFIter mfi(Sborder);mfi.isValid();++mfi) {
                    auto state=Sborder.const_array(mfi);
                    auto ref=potential_reference[ilev].array(mfi);
                    amrex::ParallelFor(amrex::grow(mfi.validbox(),num_grow),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        ref(i,j,k)=cathode_reference*(state(i,j,k,C_AM_ID)+state(i,j,k,C_CBD_ID));
                    });
                }
#else
                amrex::Abort("MORQEX offset requires the Microstructure phase definition");
#endif
                amrex::MultiFab::Subtract(potential[ilev],potential_reference[ilev],0,0,1,num_grow);
                if (nl_it>0) amrex::MultiFab::Copy(potential[ilev],fluctuation[ilev],0,0,1,0);
            }

            acoeff[ilev].setVal(1.0); //will be scaled by ascalar
            mlabec.setACoeffs(ilev, acoeff[ilev]);

            bcoeff[ilev].setVal(1.0);
            solution[ilev].setVal(0.0);
            rhs[ilev].setVal(0.0);

            //default to homogenous Neumann
            robin_a[ilev].setVal(0.0);
            robin_b[ilev].setVal(1.0);
            robin_f[ilev].setVal(0.0);


            // copy current solution for better guess
            if (pot_initial_guess)
            {
                amrex::MultiFab::Copy(solution[ilev], potential[ilev], 0, 0, 1, 0);
                // solution[ilev].copy(potential[ilev], 0, 0, 1);
            }

            // fill cell centered diffusion coefficients and rhs
            for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                const Box& gbx = amrex::grow(bx, 1);
                const auto dx = geom[ilev].CellSizeArray();
                auto prob_lo = geom[ilev].ProbLoArray();
                auto prob_hi = geom[ilev].ProbHiArray();
                const Box& domain = geom[ilev].Domain();

                Real time = current_time; // for GPU capture

                Array4<Real> phi_arr = Sborder.array(mfi);
                Array4<Real> bcoeff_arr = bcoeff[ilev].array(mfi);
                Array4<Real> rhs_arr = rhs[ilev].array(mfi);

                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        electrochem_transport::compute_potential_dcoeff(i, j, k, phi_arr, bcoeff_arr, 
                                                   prob_lo, prob_hi, dx, time, *localprobparm);
                        });
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        electrochem_reactions::compute_potential_source(i, j, k, phi_arr, 
                                              rhs_arr, prob_lo, prob_hi, dx, time, *localprobparm);
                        });
            }

            // average cell coefficients to faces, this includes boundary faces
            Array<MultiFab, AMREX_SPACEDIM> face_bcoeff;
            Array<MultiFab, AMREX_SPACEDIM> face_bcoeff_res;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const BoxArray& ba = amrex::convert(bcoeff[ilev].boxArray(), IntVect::TheDimensionVector(idim));
                face_bcoeff[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
                face_bcoeff_res[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
            }
            // true argument for harmonic averaging
            amrex::average_cellcenter_to_face(GetArrOfPtrs(face_bcoeff), bcoeff[ilev], geom[ilev], 1, true);
            amrex::average_cellcenter_to_face(GetArrOfPtrs(face_bcoeff_res), bcoeff[ilev], geom[ilev], 1, true);
            
            Array<MultiFab, AMREX_SPACEDIM> kdterm;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const BoxArray& ba = amrex::convert(bcoeff[ilev].boxArray(), 
                                                    IntVect::TheDimensionVector(idim));
                kdterm[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
            }

            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const auto dx = geom[ilev].CellSizeArray();
                    const Box& bx = mfi.tilebox();
                    auto prob_lo = geom[ilev].ProbLoArray();
                    auto prob_hi = geom[ilev].ProbHiArray();

                    Real time = current_time; // for GPU capture

                    // face box
                    Box fbox = convert(bx, IntVect::TheDimensionVector(idim));
                    Array4<Real> phi_arr = Sborder.array(mfi);
                    Array4<Real> kdterm_arr = kdterm[idim].array(mfi);
                    int captured_kd_conc_id = kd_conc_id; //public variable

                    amrex::ParallelFor(fbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                        IntVect left(i, j, k);
                        IntVect right(i, j, k);
                        left[idim] -= 1;

                        //kdstar is kd/conc
                        //del.(kd grad(lnc))=del. (kd/c grad(c))=del.(kdstar grad(c))
                        //del.(-sigma grad(phi))+del.(kdstar grad(c))=0
                        amrex::Real kdstar = electrochem_transport::compute_kdstar_atface(i, j, k, idim,
                                                                                          phi_arr, prob_lo, 
                                                                                          prob_hi, dx, time, 
                                                                                          *localprobparm);

                        kdterm_arr(i,j,k)=kdstar*(phi_arr(right,captured_kd_conc_id)
                                                  -phi_arr(left,captured_kd_conc_id))/dx[idim];
                    });
                }
            }


            if (buttler_vohlmer_flux)
            {
                int lset_id = bv_levset_id;
                Real gradctol = lsgrad_tolerance;

                Array<MultiFab, AMREX_SPACEDIM> bv_explicit_terms;
                Array<MultiFab, AMREX_SPACEDIM> bv_explicit_terms_res;
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    const BoxArray& ba = amrex::convert(bcoeff[ilev].boxArray(), 
                                                        IntVect::TheDimensionVector(idim));
                    bv_explicit_terms[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
                    bv_explicit_terms_res[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
                }

                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                    {
                        const auto dx = geom[ilev].CellSizeArray();
                        auto plo = geom[ilev].ProbLoArray();
                        auto phi = geom[ilev].ProbHiArray();
                        const int* domlo_arr = geom[ilev].Domain().loVect();
                        const int* domhi_arr = geom[ilev].Domain().hiVect();
                        GpuArray<int,AMREX_SPACEDIM> domlo={AMREX_D_DECL(domlo_arr[0], domlo_arr[1], domlo_arr[2])};
                        GpuArray<int,AMREX_SPACEDIM> domhi={AMREX_D_DECL(domhi_arr[0], domhi_arr[1], domhi_arr[2])};
                        const Box& bx = mfi.tilebox();
                        Real min_dx = amrex::min(dx[0], amrex::min(dx[1], dx[2]));

                        // face box
                        Box fbox = convert(bx, IntVect::TheDimensionVector(idim));
                        Array4<Real> phi_arr = Sborder.array(mfi);
                        Array4<Real> dcoeff_arr = face_bcoeff[idim].array(mfi);
                        Array4<Real> dcoeff_arr_res = face_bcoeff_res[idim].array(mfi);
                        Array4<Real> explterms_arr = bv_explicit_terms[idim].array(mfi);
                        Array4<Real> explterms_arr_res = bv_explicit_terms_res[idim].array(mfi);
                        Array4<Real> kdterm_arr=kdterm[idim].array(mfi);


                        amrex::ParallelFor(fbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                            int normaldir = idim;
                            Real mod_gradc=0.0;
                            Real facecolor=0.0;
                            Real potjump=0.0;
                            Real gradc_cutoff=0.0;
                            Real dphidn = 0.0;
                            Real dphidt1 = 0.0;
                            Real dphidt2 = 0.0;
                            Real n_ls[AMREX_SPACEDIM];
                            Real intloc[AMREX_SPACEDIM];


                            bv_get_grads_and_jumps(i, j, k, normaldir, lset_id, dx, phi_arr, gradctol,
                                                   mod_gradc, gradc_cutoff, facecolor, potjump, dphidn, 
                                                   dphidt1, dphidt2, n_ls, intloc, 
                                                   plo, phi);


                            explterms_arr(i, j, k) = 0.0;
                            explterms_arr_res(i, j, k) = 0.0;
                            Real activ_func = electrochem_reactions::bv_activation_function(facecolor, mod_gradc, gradc_cutoff);

                            if (mod_gradc > gradc_cutoff && activ_func > 0.0)
                            {

                                //if(fabs(potjump) > 1)
                                //{
                                //   Print()<<"potjump:"<<potjump<<"\t"<<dphidn<<"\t"<<dphidt1<<"\t"<<dphidt2<<"\t"
                                //     <<dcdn<<"\t"<<dcdt1<<"\t"<<dcdt2<<"\t"<<mod_gradc<<"\n";
                                //}

                                // FIXME: pass ion concentration also
                                // FIXME: ideally it should be the ion concentration at the closest electrode cell
                                Real j_bv,jdash_bv;
                                electrochem_reactions::bvcurrent_and_der(i,j,k,normaldir,potjump,phi_arr,
                                                                         *localprobparm,j_bv,jdash_bv);

                                dcoeff_arr(i, j, k) *= (1.0 - activ_func);
                                dcoeff_arr_res(i, j, k) = dcoeff_arr(i, j, k);


                                // dcoeff_arr(i,j,k) += -jdash_bv*activ_func/pow(mod_gradc,3.0) * dcdn*dcdn;
                                dcoeff_arr(i, j, k) += -jdash_bv * activ_func / mod_gradc * n_ls[0] * n_ls[0];


                                // expl term1
                                // explterms_arr(i,j,k) =   j_bv*activ_func*dcdn/mod_gradc;
                                explterms_arr(i, j, k) = j_bv * activ_func * n_ls[0];
                                explterms_arr_res(i, j, k) = explterms_arr(i, j, k);

                                // expl term2
                                // explterms_arr(i,j,k) += -jdash_bv*potjump*activ_func*dcdn/mod_gradc;
                                explterms_arr(i, j, k) += -jdash_bv * potjump * activ_func * n_ls[0];

                                // expl term3 (mix derivative terms from tensor product)
                                // explterms_arr(i,j,k) +=  jdash_bv*activ_func/pow(mod_gradc,3.0)*(dcdn*dcdt1*dphidt1+dcdn*dcdt2*dphidt2);
                                explterms_arr(i, j, k) += jdash_bv * activ_func / mod_gradc * 
                                (n_ls[0] * n_ls[1] * dphidt1 + n_ls[0] * n_ls[2] * dphidt2);
                                kdterm_arr(i,j,k) *= (1.0-activ_func);

                            }
                        });
                    }
                }


                for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const Box& bx = mfi.tilebox();
                    const auto dx = geom[ilev].CellSizeArray();

                    Array4<Real> rhs_arr = rhs[ilev].array(mfi);
                    Array4<Real> rhs_arr_res = rhs_res[ilev].array(mfi);
                    Array4<Real> phi_arr = Sborder.array(mfi);

                    Array4<Real> term_x = bv_explicit_terms[0].array(mfi);
                    Array4<Real> term_y = bv_explicit_terms[1].array(mfi);
                    Array4<Real> term_z = bv_explicit_terms[2].array(mfi);

                    Array4<Real> term_x_res = bv_explicit_terms_res[0].array(mfi);
                    Array4<Real> term_y_res = bv_explicit_terms_res[1].array(mfi);
                    Array4<Real> term_z_res = bv_explicit_terms_res[2].array(mfi);

                    Array4<Real> kdterm_x = kdterm[0].array(mfi);
                    Array4<Real> kdterm_y = kdterm[1].array(mfi);
                    Array4<Real> kdterm_z = kdterm[2].array(mfi);

                    Real relax_fac = bv_relaxfac;

                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                        rhs_arr_res(i,j,k) = rhs_arr(i,j,k);

                        rhs_arr(i, j, k) += (term_x(i, j, k) - term_x(i + 1, j, k)) / dx[0] 
                        + (term_y(i, j, k) - term_y(i, j + 1, k)) / dx[1] +
                        (term_z(i, j, k) - term_z(i, j, k + 1)) / dx[2];

                        rhs_arr(i, j, k) += phi_arr(i, j, k, POT_ID) * relax_fac;

                        rhs_arr(i, j, k) += (kdterm_x(i, j, k) - kdterm_x(i + 1, j, k)) / dx[0] 
                        + (kdterm_y(i, j, k) - kdterm_y(i, j + 1, k)) / dx[1] +
                        (kdterm_z(i, j, k) - kdterm_z(i, j, k + 1)) / dx[2];

                        // MORQEX: include electrolyte concentration-current flux in
                        // the same physical residual, on every face (not only BV faces).
                        rhs_arr_res(i,j,k) += (kdterm_x(i,j,k)-kdterm_x(i+1,j,k))/dx[0]
                          + (kdterm_y(i,j,k)-kdterm_y(i,j+1,k))/dx[1]
                          + (kdterm_z(i,j,k)-kdterm_z(i,j,k+1))/dx[2];

                        rhs_arr_res(i, j, k) += (term_x_res(i, j, k) - term_x_res(i + 1, j, k)) / dx[0] 
                        + (term_y_res(i, j, k) - term_y_res(i, j + 1, k)) / dx[1] +
                        (term_z_res(i, j, k) - term_z_res(i, j, k + 1)) / dx[2];

                    });
                }
            }

            // set boundary conditions
            for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                const auto dx = geom[ilev].CellSizeArray();
                auto prob_lo = geom[ilev].ProbLoArray();
                auto prob_hi = geom[ilev].ProbHiArray();
                const Box& domain = geom[ilev].Domain();

                Array4<Real> phi_arr = Sborder.array(mfi);
                Array4<Real> bc_arr = potential[ilev].array(mfi);

                Array4<Real> robin_a_arr = robin_a[ilev].array(mfi);
                Array4<Real> robin_b_arr = robin_b[ilev].array(mfi);
                Array4<Real> robin_f_arr = robin_f[ilev].array(mfi);

                Real time = current_time; // for GPU capture

                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    const amrex::Real bclo = host_global_storage->pot_bc_lo[idim];
                    const amrex::Real bchi = host_global_storage->pot_bc_hi[idim];

                    if (!geom[ilev].isPeriodic(idim))
                    {
                        if (bx.smallEnd(idim) == domain.smallEnd(idim))
                        {
                            amrex::ParallelFor(amrex::bdryLo(bx, idim), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                electrochem_transport::potential_bc(i, j, k, idim, -1, phi_arr, bc_arr, 
                                                                    prob_lo, prob_hi, dx, time, bclo, bchi);
                            });
                        }
                        if (bx.bigEnd(idim) == domain.bigEnd(idim))
                        {
                            amrex::ParallelFor(amrex::bdryHi(bx, idim), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                electrochem_transport::potential_bc(i, j, k, idim, +1, phi_arr, 
                                                                    bc_arr, prob_lo, prob_hi, dx, time, bclo, bchi);
                            });
                        }

                        if(mixedbc)
                        {
                            if (bx.smallEnd(idim) == domain.smallEnd(idim))
                            {
                                amrex::ParallelFor(amrex::bdryLo(bx, idim), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                    electrochem_transport::potential_mixedbc(i, j, k, idim, -1, phi_arr, robin_a_arr, 
                                                                             robin_b_arr, robin_f_arr, prob_lo, prob_hi, dx, time, bclo, bchi);
                                });
                            }
                            if (bx.bigEnd(idim) == domain.bigEnd(idim))
                            {
                                amrex::ParallelFor(amrex::bdryHi(bx, idim), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                    electrochem_transport::potential_mixedbc(i, j, k, idim, +1, phi_arr, robin_a_arr, 
                                                                             robin_b_arr, robin_f_arr, prob_lo, prob_hi, dx, time, bclo, bchi);
                                });
                            }
                        }
                    }
                }
            }

            // Exact change of variable phi = psi + phi_ref. Transform both
            // linearized and physical right-hand sides; no weak conductivity,
            // changed gauge, altered BV law or relaxed stopping criterion.
            if (morqex_potential_offset) {
                for (MFIter mfi(rhs[ilev]);mfi.isValid();++mfi) {
                    auto ref=potential_reference[ilev].const_array(mfi);
                    auto f=rhs[ilev].array(mfi);auto fr=rhs_res[ilev].array(mfi);
                    auto dx=geom[ilev].CellSizeArray();Real relaxation=ascalar;
                    GpuArray<Array4<Real const>,AMREX_SPACEDIM> b{AMREX_D_DECL(face_bcoeff[0].const_array(mfi),face_bcoeff[1].const_array(mfi),face_bcoeff[2].const_array(mfi))};
                    GpuArray<Array4<Real const>,AMREX_SPACEDIM> br{AMREX_D_DECL(face_bcoeff_res[0].const_array(mfi),face_bcoeff_res[1].const_array(mfi),face_bcoeff_res[2].const_array(mfi))};
                    amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        IntVect c(AMREX_D_DECL(i,j,k));
                        f(c)-=relaxation*ref(c);
                        for(int axis=0;axis<AMREX_SPACEDIM;++axis) {
                            IntVect lo=c,hi=c;lo[axis]-=1;hi[axis]+=1;
                            f(c)+=(b[axis](hi)*(ref(hi)-ref(c))-b[axis](c)*(ref(c)-ref(lo)))/(dx[axis]*dx[axis]);
                            fr(c)+=(br[axis](hi)*(ref(hi)-ref(c))-br[axis](c)*(ref(c)-ref(lo)))/(dx[axis]*dx[axis]);
                        }
                    });
                }
            }
            // set b with diffusivities
            mlabec.setBCoeffs(ilev, amrex::GetArrOfConstPtrs(face_bcoeff));
            mlabec_res.setBCoeffs(ilev, amrex::GetArrOfConstPtrs(face_bcoeff_res));

            // bc's are stored in the ghost cells of potential
            mlabec.setLevelBC(ilev, &potential[ilev], &(robin_a[ilev]), &(robin_b[ilev]), &(robin_f[ilev]));
            mlabec_res.setLevelBC(ilev, &(potential[ilev]), &(robin_a[ilev]), &(robin_b[ilev]), &(robin_f[ilev]));
        }

        // Evaluate the actual current iterate, with BV fluxes and concentration
        // current assembled from this SAME state. Never combine a new potential
        // with old nonlinear reaction fluxes and call that the nonlinear residual.
        if (buttler_vohlmer_flux) {
            mlmg_res.apply(GetVecOfPtrs(residual),GetVecOfPtrs(potential));
            Real total_nl_res = 0.0;
            Real physical_source_norm = 0.0;
            for (int ilev=0; ilev<=finest_level; ++ilev) {
                amrex::MultiFab mask;
                if (ilev<finest_level) {
                    mask=makeFineMask(grids[ilev],dmap[ilev],grids[ilev+1],refRatio(ilev),1.0,0.0);
                } else {
                    mask.define(grids[ilev],dmap[ilev],1,0); mask.setVal(1.0);
                }
                amrex::MultiFab::Subtract(residual[ilev],rhs_res[ilev],0,0,1,0);
                amrex::MultiFab::Multiply(residual[ilev],mask,0,0,1,0);
                total_nl_res += residual[ilev].norm2();
                physical_source_norm += rhs_res[ilev].norm2();
            }
            if (!std::isfinite(total_nl_res) || !std::isfinite(physical_source_norm))
                amrex::Abort("MORQEX: nonfinite BV residual");
            if (nl_it==0) {
                errnorm_1st_iter=amrex::max(total_nl_res,physical_source_norm);
                amrex::Print()<<"MORQEX BV SCALE "<<errnorm_1st_iter<<"\n";
            }
            const Real relative=errnorm_1st_iter>0 ? total_nl_res/errnorm_1st_iter : 0.0;
            amrex::Print()<<"BV NON-LINEAR RESIDUAL (rel,abs): "<<relative<<' '<<total_nl_res<<"\n";
            // At least one solve is needed to populate the field-gradient output.
            if (nl_it>0 && (total_nl_res<bv_nonlinear_abstol || relative<bv_nonlinear_reltol)) {
                for (int ilev=0;ilev<=finest_level;++ilev)
                    amrex::MultiFab::Copy(solution[ilev],potential[ilev],0,0,1,0);
                amrex::Print()<<"MORQEX BV ACCEPT "<<relative<<' '<<total_nl_res<<"\n";
                break;
            }
            if (nl_it==num_nonlinear_iters-1)
                amrex::Abort("MORQEX: BV nonlinear iteration budget exhausted without convergence");
        }
        Real inner_relative=tol_rel, inner_absolute=tol_abs;
        if (buttler_vohlmer_flux && morqex_linear_budget_fraction>0.0) {
            // On a single uniform grid ||r||_2 <= sqrt(N)*||r||_infinity.
            // Allocate a fraction of the unchanged OUTER physical residual
            // tolerance to the inner linear system. This avoids oversolving an
            // absolute potential equation past floating-point cancellation.
            // This is an accuracy budget, not an Eisenstat-Walker forcing law
            // or a field-error bound. The next iterate's actual nonlinear
            // residual is still rebuilt and independently accepted above.
            const Real cells=static_cast<Real>(geom[0].Domain().numPts());
            inner_relative=0.0;
            inner_absolute=morqex_linear_budget_fraction *
                amrex::max(bv_nonlinear_abstol,bv_nonlinear_reltol*errnorm_1st_iter)/std::sqrt(cells);
            amrex::Print()<<"MORQEX LINEAR ABSOLUTE BUDGET "<<inner_absolute<<"\n";
        }
        mlmg.solve(GetVecOfPtrs(solution),GetVecOfConstPtrs(rhs),inner_relative,inner_absolute);
        mlmg.getGradSolution(GetVecOfArrOfPtrs(gradsoln));
        for (int ilev=0;ilev<=finest_level;++ilev) {
            amrex::MultiFab::Copy(fluctuation[ilev],solution[ilev],0,0,1,0);
            amrex::MultiFab::Copy(phi_new[ilev],solution[ilev],0,POT_ID,1,0);
            if (morqex_potential_offset)
                amrex::MultiFab::Add(phi_new[ilev],potential_reference[ilev],0,POT_ID,1,0);
        }

    }


    // copy solution back to phi_new
    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        amrex::MultiFab::Copy(phi_new[ilev], solution[ilev], 0, POT_ID, 1, 0);
        if (morqex_potential_offset) {
            amrex::MultiFab::Add(phi_new[ilev],potential_reference[ilev],0,POT_ID,1,0);
            for (int axis=0;axis<AMREX_SPACEDIM;++axis)
                for (MFIter mfi(gradsoln[ilev][axis]);mfi.isValid();++mfi) {
                    auto g=gradsoln[ilev][axis].array(mfi);auto ref=potential_reference[ilev].const_array(mfi);
                    Real h=geom[ilev].CellSize(axis);
                    amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        IntVect r(AMREX_D_DECL(i,j,k)),l=r;l[axis]-=1;
                        g(i,j,k)+=(ref(r)-ref(l))/h;
                    });
                }
        }
        const Array<const MultiFab*, AMREX_SPACEDIM> allgrad = {&gradsoln[ilev][0], 
            &gradsoln[ilev][1], &gradsoln[ilev][2]};
        average_face_to_cellcenter(phi_new[ilev], EFX_ID, allgrad);
        phi_new[ilev].mult(-1.0, EFX_ID, 3);
    }

    //clear
    potential.clear();
    acoeff.clear();
    bcoeff.clear();
    gradsoln.clear();
    solution.clear();
    residual.clear();
    rhs.clear();
    err.clear();
    rhs_res.clear();

    robin_a.clear();
    robin_b.clear();
    robin_f.clear();
}

void echemAMR::compute_current_den(Vector<Array<MultiFab, AMREX_SPACEDIM>>& currentden,Real cur_time)
{
    int num_grow=2;
    for(int lev=0;lev<=finest_level;lev++)
    {
        MultiFab Sborder(grids[lev], dmap[lev], phi_new[lev].nComp(), num_grow);
        FillPatch(lev, cur_time, Sborder, 0, Sborder.nComp());
        compute_current_density_at_level(lev, Sborder, currentden[lev],cur_time);
    }

    // =======================================================
    // Average down the current at coarse fine interfaces
    // =======================================================
    for (int lev = finest_level; lev > 0; lev--)
    {
        average_down_faces(
            amrex::GetArrOfConstPtrs(currentden[lev]),
            amrex::GetArrOfPtrs(currentden[lev - 1]), refRatio(lev - 1),
            Geom(lev - 1));
    }
    // =======================================================
}

void echemAMR::compute_current_density_at_level(
    int lev,
    MultiFab& Sborder,
    Array<MultiFab, AMREX_SPACEDIM>& currden,Real current_time)
{

    const auto dx = geom[lev].CellSizeArray();
    auto prob_lo = geom[lev].ProbLoArray();
    auto prob_hi = geom[lev].ProbHiArray();
    ProbParm const* localprobparm = d_prob_parm;
    MultiFab conductivity;
    conductivity.define(grids[lev], dmap[lev], 1, 1);
    conductivity.setVal(1.0);

    int ncomp = Sborder.nComp();
    const int* domlo = geom[lev].Domain().loVect();
    const int* domhi = geom[lev].Domain().hiVect();
    int lset_id = bv_levset_id;
    Real time=current_time;
    int captured_kd_conc_id = kd_conc_id; //public variable

    GpuArray<int, AMREX_SPACEDIM> domlo_arr = {
        AMREX_D_DECL(domlo[0], domlo[1], domlo[2])};
    GpuArray<int, AMREX_SPACEDIM> domhi_arr = {
        AMREX_D_DECL(domhi[0], domhi[1], domhi[2])};

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx, 1);
            Array4<Real> phi_arr = Sborder.array(mfi);
            Array4<Real> conductivity_arr = conductivity.array(mfi);
            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                electrochem_transport::compute_potential_dcoeff(i, j, k, phi_arr, conductivity_arr, 
                                                                prob_lo, prob_hi, dx, time, *localprobparm);
            });
        }
    }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            Array<Box, AMREX_SPACEDIM> face_boxes;
            face_boxes[0] = mfi.nodaltilebox(0);
#if AMREX_SPACEDIM > 1
            face_boxes[1] = mfi.nodaltilebox(1);
#if AMREX_SPACEDIM == 3
            face_boxes[2] = mfi.nodaltilebox(2);
#endif
#endif
            Array4<Real> sborder_arr = Sborder.array(mfi);
            Array4<Real> conductivity_arr = conductivity.array(mfi);

            GpuArray<Array4<Real>, AMREX_SPACEDIM> j_arr{AMREX_D_DECL(
                    currden[0].array(mfi), currden[1].array(mfi),
                    currden[2].array(mfi))};

            for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
            {
                amrex::ParallelFor(
                    face_boxes[idim],
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        IntVect face{AMREX_D_DECL(i, j, k)};
                        IntVect lcell{AMREX_D_DECL(i, j, k)};
                        IntVect rcell{AMREX_D_DECL(i, j, k)};
                        lcell[idim] -= 1;

                        j_arr[idim](face) = 0.0;

                        int ls_L = int(sborder_arr(lcell,lset_id));
                        int ls_R = int(sborder_arr(rcell,lset_id));

                        int regular_interface=ls_L*ls_R + (!ls_L)*(!ls_R);
                        
                        int left_phys_boundary =
                            (face[idim] == domlo_arr[idim]);
                        int right_phys_boundary =
                            (face[idim] == (domhi_arr[idim] + 1));

                        if(regular_interface)
                        {
                            Real efield=0.0;
                            Real gradc=0.0;
                            if(!left_phys_boundary && !right_phys_boundary)
                            {
                              efield=-(sborder_arr(rcell,POT_ID)-sborder_arr(lcell,POT_ID))/dx[idim];
                              gradc=(sborder_arr(rcell,captured_kd_conc_id)
                                                         -sborder_arr(lcell,captured_kd_conc_id))/dx[idim];
                            }
                            else if(left_phys_boundary)
                            {
                                IntVect rcellp1=rcell;
                                rcellp1[idim]+=1;
                                efield=-(sborder_arr(rcellp1,POT_ID)-sborder_arr(rcell,POT_ID))/dx[idim];
                                gradc=(sborder_arr(rcellp1,captured_kd_conc_id)
                                                         -sborder_arr(rcell,captured_kd_conc_id))/dx[idim];
                            }
                            else //right phys boundary
                            {
                                IntVect lcellm1=lcell;
                                lcellm1[idim]-=1;
                                efield=-(sborder_arr(lcell,POT_ID)-sborder_arr(lcellm1,POT_ID))/dx[idim];
                                gradc=(sborder_arr(lcell,captured_kd_conc_id)
                                                         -sborder_arr(lcellm1,captured_kd_conc_id))/dx[idim];
                            }
                            // Match the harmonic face coefficient used by the potential operator.
                            const Real kl=conductivity_arr(lcell),kr=conductivity_arr(rcell);
                            j_arr[idim](face) += (kl>0 && kr>0 ? 2.0*kl*kr/(kl+kr) : 0.0)*efield;

                            amrex::Real kdstar = electrochem_transport::compute_kdstar_atface(i, j, k, idim,
                                                                                              sborder_arr, prob_lo, 
                                                                                              prob_hi, dx, time, 
                                                                                              *localprobparm);
                            j_arr[idim](face) += kdstar*gradc;
                        }

                    });
            }
        }
    }
}
