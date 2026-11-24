"""Nested residual-driven enrichment for the fixed quadratic coupled system.

Each enrichment explicitly queries one full concentration BE solve relative to
the current approximate state. It is not a hidden exact trajectory reset. The
previous certified error, lithium inventory and energy are carried unchanged
through the nested state transfer; every full query/setup is counted.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cho_solve, lu_factor, lu_solve
from .concentration import ConcentrationBasis, _orthogonal, component_constants
from .gradient_transport import PortGradientModel, EnergyAccuracyError, EnergyErrorBound, GradientState
from .transport import ConcentrationInfeasible


class AdaptiveBudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdaptiveGradientDiagnostics:
    dynamics: object
    events: tuple
    high_fidelity_queries: int
    enrichment_seconds: float
    total_seconds: float
    used_full_fallback: bool
    full_space_steps: int
    resolvent_right_hand_sides: int
    resolvent_factorizations: int


class AdaptiveGradientTransport:
    def __init__(self, system, initial_basis, *, maximum_modes, maximum_enrichments_per_step=3,
                 allow_full_fallback=False, enrichment_strategy="snapshot", resolvent_shifts=(.025,.1,.5,2.),
                 block_rank=2, channel_scales=None, macro_strain_scale=1e-3, restart_dimension=None,
                 retain_forcing_directions=False, certificate_method="graph", error_modes=16):
        basis = system.conservative_basis(initial_basis)
        valid_initial = basis.values.shape[1] <= maximum_modes or (allow_full_fallback and basis.values.shape[1] == system.n)
        if not valid_initial or not 1 <= maximum_modes <= system.n or maximum_enrichments_per_step < 1:
            raise ValueError("Invalid adaptive mode/query budget")
        self.system = system
        self.certificate_method,self.error_modes = certificate_method,int(error_modes)
        self.model = PortGradientModel(system,basis,certificate_method=certificate_method,error_modes=error_modes)
        self.maximum_modes, self.maximum_enrichments = int(maximum_modes), int(maximum_enrichments_per_step)
        self.allow_full_fallback = bool(allow_full_fallback)
        if enrichment_strategy not in {"snapshot","resolvent-block","compressed-resolvent-block","state-preserving-restart"}: raise ValueError("Unknown enrichment strategy")
        shifts = np.asarray(resolvent_shifts,dtype=float)
        if shifts.ndim != 1 or not len(shifts) or not np.isfinite(shifts).all() or np.any(shifts <= 0): raise ValueError("Positive resolvent shifts required")
        self.enrichment_strategy, self.resolvent_shifts = enrichment_strategy,tuple(float(s) for s in shifts)
        if enrichment_strategy in {"compressed-resolvent-block","state-preserving-restart"} and channel_scales is None:
            raise ValueError("Declare physical channel scales for compressed response blocks")
        scales = np.ones(system.b.shape[1]) if channel_scales is None else np.asarray(channel_scales,dtype=float)
        if scales.shape != (system.b.shape[1],) or not np.isfinite(scales).all() or np.any(scales <= 0) or block_rank < 1 or not np.isfinite(macro_strain_scale) or macro_strain_scale <= 0:
            raise ValueError("Positive block rank, flux scales and macro strain scale required")
        self.channel_scales,self.macro_strain_scale,self.block_rank = scales.copy(),float(macro_strain_scale),int(block_rank)
        self.restart_dimension = min(int(restart_dimension or 24),self.maximum_modes,system.n)
        self.retain_forcing_directions = bool(retain_forcing_directions)
        constants = ConcentrationBasis(component_constants(system.diffusion),system.weights.copy(),"mass-seed",system.diffusion.n_components)
        self._restart_seed = system.conservative_basis(constants).values
        if enrichment_strategy == "state-preserving-restart" and self.restart_dimension < min(system.n,self._restart_seed.shape[1]+3):
            raise ValueError("Restart dimension cannot contain conserved directions and current state/load")
        self._shift_factors, self._responses = {},{}
        self.total_full_queries, self.total_updates = 0, 0
        self.total_full_space_steps = 0
        self.total_resolvent_rhs, self.total_resolvent_factorizations = 0,0

    def _block_directions(self, channel_flux, macro_strain):
        u,e = np.asarray(channel_flux),np.asarray(macro_strain)
        ports = [("flux",int(i)) for i in np.flatnonzero(np.abs(u) > 1e-14*max(np.max(np.abs(u)),1e-30))]
        ports += [("macro",int(i)) for i in np.flatnonzero(np.abs(e) > 1e-14*max(np.max(np.abs(e)),1e-30))]
        columns = []
        for shift in self.resolvent_shifts:
            missing = [(kind,i) for kind,i in ports if (shift,kind,i) not in self._responses]
            if missing:
                if shift not in self._shift_factors:
                    self._shift_factors[shift] = lu_factor(shift*np.eye(self.system.n)+self.system.operator)
                    self.total_resolvent_factorizations += 1
                rhs = np.column_stack([self.system.b[:,i] if kind == "flux" else -self.system.macro_forcing[:,i] for kind,i in missing])
                values = lu_solve(self._shift_factors[shift],rhs)
                self.total_resolvent_rhs += len(missing)
                for j,(kind,i) in enumerate(missing): self._responses[shift,kind,i] = values[:,j].copy()
            columns.extend(self._responses[shift,kind,i]*(self.channel_scales[i] if kind == "flux" else self.macro_strain_scale) for kind,i in ports)
        return np.column_stack(columns) if columns else np.empty((self.system.n,0))

    def _forcing_directions(self, channel_flux, macro_strain):
        u,e = np.asarray(channel_flux),np.asarray(macro_strain)
        selected_u = np.flatnonzero(np.abs(u) > 1e-14*max(np.max(np.abs(u)),1e-30))
        selected_e = np.flatnonzero(np.abs(e) > 1e-14*max(np.max(np.abs(e)),1e-30))
        return np.column_stack([self.system.b[:,selected_u],-self.system.macro_forcing[:,selected_e],-self.system.offset_forcing])

    def _compress_block(self, block, core_basis, rank=None):
        if not block.shape[1]: return block
        h = self.system.H
        gram = core_basis.T @ h @ core_basis
        innovation = block-core_basis @ np.linalg.solve(gram,core_basis.T @ h @ block)
        metric = self.system.chol.T @ innovation
        _,singular,right = np.linalg.svd(metric,full_matrices=False)
        if not len(singular) or singular[0] == 0: return np.empty((self.system.n,0))
        retained = min(self.block_rank if rank is None else rank,int(np.sum(singular > singular[0]*1e-12)))
        return innovation @ right[:retained].T

    def initialize(self, initial_deviation, *, error_tolerance, macro_strain=None):
        return self.model.initialize(initial_deviation,error_tolerance=error_tolerance,macro_strain=macro_strain)

    def step(self, previous, dt, channel_flux, *, error_tolerance, macro_strain=None):
        if previous.model_fingerprint != self.model.fingerprint: raise ValueError("Adaptive state does not match the active model")
        started = time.perf_counter()
        model, state = self.model, previous
        events, queries, setup_seconds, fallback = [], 0, 0., False
        rhs_before, factors_before = self.total_resolvent_rhs,self.total_resolvent_factorizations
        for attempt in range(self.maximum_enrichments+1):
            candidate, diagnostic = model.trial(state,dt,channel_flux,macro_strain=macro_strain)
            try:
                model.accept(candidate,error_tolerance)
                # Only an accepted candidate commits a model/state change.
                self.model = model; self.total_updates += len(events)
                full_steps = int(model.dimension == self.system.n)
                self.total_full_space_steps += full_steps
                return candidate, AdaptiveGradientDiagnostics(diagnostic,tuple(events),queries,
                    setup_seconds,time.perf_counter()-started,fallback,full_steps,
                    self.total_resolvent_rhs-rhs_before,self.total_resolvent_factorizations-factors_before)
            except (EnergyAccuracyError, ConcentrationInfeasible) as failure:
                if attempt == self.maximum_enrichments:
                    raise AdaptiveBudgetError(f"No certified step within enrichment budget: {failure}") from failure
                setup = time.perf_counter()
                old_basis = model.basis
                v = old_basis.values
                if model.dimension >= self.maximum_modes and self.enrichment_strategy != "state-preserving-restart":
                    if not self.allow_full_fallback or model.dimension == self.system.n:
                        raise AdaptiveBudgetError(f"Mode budget reached at {model.dimension}; state not advanced: {failure}") from failure
                    updated = _orthogonal(np.eye(self.system.n),self.system.weights,initial=v,count=self.system.n)
                    fallback = True
                    reason = "explicit full-space fallback; accumulated bound retained"
                else:
                    # This query uses only the current approximate state/input;
                    # no exact reference trajectory is passed to the algorithm.
                    current = v @ state.coefficients
                    exact_relative_step = self.system.full_step(current,dt,channel_flux,
                        macro_strain=candidate.macro_strain)
                    queries += 1; self.total_full_queries += 1
                    correction = exact_relative_step-v @ candidate.coefficients
                    lifting = cho_solve((self.system.chol,True), self.system.G @ candidate.macro_strain+self.system.g0)
                    if self.enrichment_strategy == "state-preserving-restart":
                        mandatory = np.column_stack([current,exact_relative_step,lifting])
                        if self.retain_forcing_directions:
                            mandatory = np.column_stack([mandatory,self._forcing_directions(channel_flux,candidate.macro_strain)])
                        core_update = _orthogonal(mandatory,self.system.weights,initial=self._restart_seed,relative_tolerance=1e-12)
                        if core_update.shape[1] > self.restart_dimension:
                            raise AdaptiveBudgetError("Mandatory state/load/conservation span exceeds restart budget")
                        block = self._block_directions(channel_flux,candidate.macro_strain)
                        directions = self._compress_block(block,core_update,self.restart_dimension-core_update.shape[1])
                        updated = _orthogonal(directions,self.system.weights,initial=core_update)
                        # Old directions may fill remaining budget but cannot
                        # displace the exact current state or proposed full step.
                        updated = _orthogonal(v,self.system.weights,initial=updated)[:,:self.restart_dimension]
                    else:
                        candidates = np.column_stack([lifting,correction])
                        core_update = _orthogonal(candidates,self.system.weights,initial=v)
                        updated = core_update
                    if self.enrichment_strategy in {"resolvent-block","compressed-resolvent-block"}:
                        block = self._block_directions(channel_flux,candidate.macro_strain)
                        if self.enrichment_strategy == "compressed-resolvent-block": block = self._compress_block(block,core_update)
                        updated = _orthogonal(block,self.system.weights,initial=core_update)
                    if updated.shape[1] > self.maximum_modes:
                        if self.enrichment_strategy != "snapshot" and core_update.shape[1] <= self.maximum_modes:
                            # Current correction/lifting precede optional block
                            # directions, so truncation preserves the core fix.
                            updated = updated[:,:self.maximum_modes]
                        elif not self.allow_full_fallback:
                            raise AdaptiveBudgetError("Enrichment would exceed mode budget; state not advanced")
                        else:
                            updated = _orthogonal(np.eye(self.system.n),self.system.weights,initial=v,count=self.system.n)
                            fallback = True
                    if updated.shape[1] == model.dimension and (self.enrichment_strategy != "state-preserving-restart" or np.array_equal(updated,v)):
                        raise AdaptiveBudgetError("No independent certified correction direction; bound not reset")
                    reason = "nested residual correction and current energy lifting; "+self.enrichment_strategy
                basis = ConcentrationBasis(updated,self.system.weights.copy(),"adaptive-energy-port",old_basis.component_count)
                enriched = PortGradientModel(self.system,basis,certificate_method=self.certificate_method,error_modes=self.error_modes)
                if self.enrichment_strategy == "state-preserving-restart":
                    transferred,transfer = model.transfer_preserving_state(state,enriched)
                else:
                    transferred, transfer = model.transfer_to(state,enriched)
                event = {**transfer,"reason":reason,"time_s":state.time,
                    "rejected_trial_error_bound":candidate.error.total_bound,
                    "tolerance":float(error_tolerance),"full_queries_this_step":queries,
                    "resolvent_rhs_cumulative":self.total_resolvent_rhs,
                    "resolvent_factorizations_cumulative":self.total_resolvent_factorizations}
                events.append(event)
                state, model = transferred, enriched
                setup_seconds += time.perf_counter()-setup
        raise RuntimeError("Unreachable adaptive branch")

    @staticmethod
    def _implementation():
        here = Path(__file__).resolve().parent
        return hashlib.sha256(b"".join((here/name).read_bytes() for name in ("adaptive_transport.py","gradient_transport.py","error_dynamics.py"))).hexdigest()

    def save_checkpoint(self, state, workspace, name):
        self.model.accept(state,max(1.,float(state.error.total_bound)*1.01))
        path = workspace.path("checkpoints",name); path.mkdir(parents=True,exist_ok=True)
        destination = path/"state.npz"
        np.savez_compressed(destination,basis=self.model.basis.values,coefficients=state.coefficients,
            macro_strain=state.macro_strain,time=state.time,energy=state.energy_density,
            inventory=state.component_inventory,error_orthogonal=state.error.orthogonal_bound,
            error_null=state.error.null_coordinates,error_total=state.error.total_bound,error_residual=state.error.residual_norm,
            error_proxy=np.empty(0) if state.error.spectral_proxy is None else state.error.spectral_proxy,
            error_spectral_uncertainty=state.error.spectral_uncertainty,error_spectral_tail=state.error.spectral_tail)
        metadata = {"schema":"morqex-adaptive-gradient-1","system_fingerprint":self.system.fingerprint,
            "model_fingerprint":self.model.fingerprint,"implementation_sha256":self._implementation(),
            "sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),"maximum_modes":self.maximum_modes,
            "maximum_enrichments":self.maximum_enrichments,"allow_full_fallback":self.allow_full_fallback,
            "total_full_queries":self.total_full_queries,"total_updates":self.total_updates,
            "total_full_space_steps":self.total_full_space_steps,"enrichment_strategy":self.enrichment_strategy,
            "resolvent_shifts":list(self.resolvent_shifts),"total_resolvent_rhs":self.total_resolvent_rhs,
            "channel_scales":self.channel_scales.tolist(),"macro_strain_scale":self.macro_strain_scale,"block_rank":self.block_rank,
            "restart_dimension":self.restart_dimension,
            "retain_forcing_directions":self.retain_forcing_directions,
            "certificate_method":self.certificate_method,"error_modes":self.error_modes,"certificate_key":state.error.certificate_key,
            "total_resolvent_factorizations":self.total_resolvent_factorizations,
            "cache_restart":"response/factor caches are cold after restart; additional work is counted"}
        (path/"manifest.json").write_text(json.dumps(metadata,indent=2))
        return path

    @classmethod
    def load_checkpoint(cls, system, path):
        path = Path(path); metadata = json.loads((path/"manifest.json").read_text())
        if metadata["schema"] != "morqex-adaptive-gradient-1" or metadata["system_fingerprint"] != system.fingerprint or metadata["implementation_sha256"] != cls._implementation():
            raise ValueError("Checkpoint does not match coupled system/implementation")
        if hashlib.sha256((path/"state.npz").read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Adaptive checkpoint checksum mismatch")
        with np.load(path/"state.npz",allow_pickle=False) as a:
            basis = ConcentrationBasis(a["basis"].copy(),system.weights.copy(),"checkpoint",system.diffusion.n_components)
            model = cls(system,basis,maximum_modes=metadata["maximum_modes"],
                maximum_enrichments_per_step=metadata["maximum_enrichments"],allow_full_fallback=metadata["allow_full_fallback"],
                enrichment_strategy=metadata["enrichment_strategy"],resolvent_shifts=metadata["resolvent_shifts"],
                channel_scales=metadata["channel_scales"],macro_strain_scale=metadata["macro_strain_scale"],block_rank=metadata["block_rank"],restart_dimension=metadata["restart_dimension"],
                retain_forcing_directions=metadata["retain_forcing_directions"],certificate_method=metadata["certificate_method"],error_modes=metadata["error_modes"])
            if model.model.fingerprint != metadata["model_fingerprint"]: raise ValueError("Checkpoint basis mismatch")
            proxy = a["error_proxy"] if metadata["certificate_method"] == "spectral" else None
            error = EnergyErrorBound(float(a["error_orthogonal"]),a["error_null"],float(a["error_total"]),float(a["error_residual"]),system.fingerprint,
                proxy,float(a["error_spectral_uncertainty"]),float(a["error_spectral_tail"]),metadata["certificate_key"])
            state = GradientState(a["coefficients"],a["macro_strain"],float(a["time"]),float(a["energy"]),a["inventory"],error,model.model.fingerprint)
        model.model.accept(state,max(1.,float(error.total_bound)*1.01))
        if not np.isclose(state.energy_density,model.model.energy(state.coefficients,state.macro_strain),rtol=1e-12,atol=1e-12) or not np.allclose(state.component_inventory,system.baseline_inventory+model.model.inventory @ state.coefficients,rtol=1e-12,atol=0):
            raise ValueError("Checkpoint energy/inventory inconsistent with its state")
        model.total_full_queries,model.total_updates = metadata["total_full_queries"],metadata["total_updates"]
        model.total_full_space_steps = metadata["total_full_space_steps"]
        model.total_resolvent_rhs,model.total_resolvent_factorizations = metadata["total_resolvent_rhs"],metadata["total_resolvent_factorizations"]
        return model,state
