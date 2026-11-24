"""Energy-metric reduction/certification of a fixed quadratic coupled system.

Established port-Hamiltonian projection is used, not claimed as a new method.
This qualified small-system implementation keeps a dense full concentration
Hessian OFFLINE (N <= 512); it is not a scalable arbitrary-N certification yet.
Prescribed flux ports and macro strain only; nonlinear BV is not certified here.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular, lu_factor, lu_solve
from .concentration import ConcentrationBasis, component_constants, _orthogonal
from .certification import tree_poincare_lower_bound
from .energy import _digest
from .transport import ConcentrationInfeasible


def immutable(value):
    a = np.array(value, dtype=float, copy=True); a.setflags(write=False)
    return a


class EnergyAccuracyError(RuntimeError):
    def __init__(self, error_bound, tolerance):
        self.error_bound, self.tolerance = float(error_bound), float(tolerance)
        super().__init__(f"Coupled energy-norm error bound {error_bound:.6g} exceeds {tolerance:.6g} sqrt(J/m^3); reject or enrich")


@dataclass(frozen=True)
class EnergyErrorBound:
    orthogonal_bound: float
    null_coordinates: np.ndarray
    total_bound: float
    residual_norm: float
    system_fingerprint: str
    spectral_proxy: object = None
    spectral_uncertainty: float = 0.
    spectral_tail: float = 0.
    certificate_key: str = "graph"

    def __post_init__(self):
        object.__setattr__(self, "null_coordinates", immutable(self.null_coordinates))
        if self.spectral_proxy is not None: object.__setattr__(self,"spectral_proxy",immutable(self.spectral_proxy))


@dataclass(frozen=True)
class GradientState:
    coefficients: np.ndarray
    macro_strain: np.ndarray
    time: float
    energy_density: float
    component_inventory: np.ndarray
    error: EnergyErrorBound
    model_fingerprint: str

    def __post_init__(self):
        for name in ("coefficients", "macro_strain", "component_inventory"):
            object.__setattr__(self, name, immutable(getattr(self, name)))


@dataclass(frozen=True)
class GradientDiagnostics:
    diffusion_dissipation: float
    external_chemical_power: float
    mechanical_work: float
    numerical_dissipation: float
    energy_balance: float
    component_mass_error: np.ndarray
    time_step_residual: float


class QuadraticGradientSystem:
    def __init__(self, response, diffusion, coordinate_basis, patterns, *, chemical_curvature,
                 reference_concentration, maximum_concentration):
        n = len(diffusion.weights)
        if n > 512: raise ValueError("Dense coupled certification exceeds the qualified N<=512 budget")
        if coordinate_basis.values.shape != (n, n) or not np.array_equal(coordinate_basis.values, np.eye(n)):
            raise ValueError("Full mechanical response must use active-voxel identity concentration coordinates")
        if not np.allclose(coordinate_basis.weights, diffusion.weights, rtol=1e-12, atol=0) or response.dimension != n+6:
            raise ValueError("Incompatible full concentration/energy coordinates")
        if not np.isfinite(chemical_curvature) or chemical_curvature <= 0:
            raise ValueError("Positive quadratic chemical curvature required")
        if not np.isfinite([reference_concentration, maximum_concentration]).all() or not 0 < reference_concentration < maximum_concentration:
            raise ValueError("Physical constant reference concentration required")
        f = np.asarray(patterns, dtype=float)
        if f.ndim != 2 or f.shape[0] != len(diffusion.face_areas) or not np.isfinite(f).all():
            raise ValueError("Finite fixed boundary-face flux pattern library required")
        self.response, self.diffusion = response, diffusion
        self.n, self.kappa = n, float(chemical_curvature)
        self.reference, self.maximum = float(reference_concentration), float(maximum_concentration)
        self.weights = immutable(diffusion.weights)
        self.H = immutable(.5*(response.H[6:, 6:]+response.H[6:, 6:].T)+self.kappa*np.diag(self.weights))
        self.chol = cholesky(self.H, lower=True)
        # H >= kappa M is needed for the graph-gap comparison used below.
        elastic = self.H-self.kappa*np.diag(self.weights)
        if np.linalg.eigvalsh(elastic).min() < -1e-10*max(np.linalg.norm(self.H, 2), 1e-30):
            raise ValueError("Mechanical energy Hessian is not positive semidefinite")
        self.A = immutable(diffusion.stiffness.toarray()/self.kappa/self.weights[:, None]/self.weights[None, :])
        self.b = immutable(np.asarray(diffusion.flux_operator @ f)/self.weights[:, None])
        self.G, self.g0 = immutable(response.H[6:, :6]), immutable(response.h[6:])
        self.operator = immutable(self.A @ self.H)
        self.macro_forcing = immutable(self.A @ self.G)
        self.offset_forcing = immutable(self.A @ self.g0)
        self.mass_constants = component_constants(diffusion)
        self.covectors = self.weights[:, None]*self.mass_constants
        self.mass_riesz = cho_solve((self.chol, True), self.covectors)
        self.null_basis = np.linalg.qr(solve_triangular(self.chol, self.covectors, lower=True), mode="reduced")[0]
        self.gap_lower = tree_poincare_lower_bound(diffusion)
        self.pointwise_constant = float(np.sqrt(np.max(np.diag(cho_solve((self.chol, True), np.eye(n))))))*(1+1e-12)
        self.inventory_rows = np.array([self.weights*(diffusion.components == j) for j in range(diffusion.n_components)])*diffusion.rve_volume
        self.baseline_inventory = self.inventory_rows @ np.full(n, self.reference)
        self.fingerprint = _digest(self.H, self.A, self.b, self.G, self.g0, self.weights,
            diffusion.voxel_ids, np.array([self.reference, self.maximum]),
            np.frombuffer(response.metadata.get("fingerprint", "").encode(), dtype=np.uint8))
        self._full_dt, self._full_factor = None, None
        self.full_solve_count = 0
        self._error_spaces = {}

    def error_space(self, count):
        from .error_dynamics import SpectralErrorSpace
        if count not in self._error_spaces: self._error_spaces[count] = SpectralErrorSpace(self,count)
        return self._error_spaces[count]

    def full_step(self, concentration_deviation, dt, channel_flux, *, macro_strain=None):
        c, u = np.asarray(concentration_deviation), np.asarray(channel_flux)
        e = np.zeros(6) if macro_strain is None else np.asarray(macro_strain)
        if c.shape != (self.n,) or u.shape != (self.b.shape[1],) or e.shape != (6,) or not np.isfinite(np.r_[c,u,e,dt]).all() or dt <= 0:
            raise ValueError("Invalid full coupled step")
        if self._full_dt != dt:
            self._full_factor = lu_factor(np.eye(self.n)+dt*self.operator); self._full_dt = dt
        self.full_solve_count += 1
        updated = lu_solve(self._full_factor, c+dt*(self.b @ u-self.macro_forcing @ e-self.offset_forcing))
        if np.any(self.reference+updated < 0) or np.any(self.reference+updated > self.maximum):
            raise ConcentrationInfeasible("Full coupled discrete concentration outside physical bounds")
        return updated

    def energy(self, c, macro_strain):
        c = np.asarray(c)
        return float(self.response.energy_density(np.r_[macro_strain, c])+.5*self.kappa*np.dot(self.weights*c, c))

    def norm(self, c): return float(np.linalg.norm(self.chol.T @ c))

    def conservative_basis(self, basis):
        v = np.asarray(basis.values)
        if v.shape[0] != self.n or not np.allclose(basis.weights, self.weights, rtol=1e-12, atol=0):
            raise ValueError("Basis does not match full transport domain")
        if not np.allclose(basis.mass, np.eye(v.shape[1]), atol=1e-10):
            q, r = np.linalg.qr(np.sqrt(self.weights)[:, None]*v, mode="reduced")
            if np.min(np.abs(np.diag(r))) < np.max(np.abs(np.diag(r)))*1e-12: raise ValueError("Dependent initial basis")
            v = q/np.sqrt(self.weights)[:, None]
        # Retaining constants alone is insufficient for H-weighted testing.
        candidates = np.column_stack([self.mass_constants, self.mass_riesz])
        augmented = _orthogonal(candidates, self.weights, initial=v)
        return ConcentrationBasis(augmented, self.weights.copy(), basis.method+"+mass-Riesz", self.diffusion.n_components)


class PortGradientModel:
    def __init__(self, system, basis, *, projection="energy", certificate_method="graph", error_modes=16):
        if projection not in {"energy","mass"}: raise ValueError("Projection must be energy or mass")
        if certificate_method not in {"graph","spectral"}: raise ValueError("Unknown coupled certificate")
        self.projection = projection
        self.certificate_method,self.error_modes = certificate_method,int(error_modes)
        self.error_space = system.error_space(self.error_modes) if certificate_method == "spectral" else None
        self.system = system
        self.basis = ConcentrationBasis(immutable(basis.values), immutable(basis.weights), basis.method, basis.component_count)
        v, h = self.basis.values, system.H
        if v.ndim != 2 or v.shape[0] != system.n or not np.isfinite(v).all(): raise ValueError("Invalid concentration basis")
        if self.basis.weights.shape != system.weights.shape or not np.allclose(self.basis.weights,system.weights,rtol=1e-12,atol=0):
            raise ValueError("Concentration mass measure does not match coupled system")
        self.H = v.T @ h @ v
        hr_chol = cholesky(self.H, lower=True)
        if projection == "energy":
            self.W = cho_solve((hr_chol, True), (h @ v).T).T
        else:
            mass = v.T @ (system.weights[:,None]*v)
            self.W = np.linalg.solve(mass,(system.weights[:,None]*v).T).T
        if np.linalg.norm(self.W @ (v.T @ system.covectors)-system.covectors) > 1e-9*np.linalg.norm(system.covectors):
            raise ValueError("Projection omits a mass Riesz/constant direction; inventories would drift")
        self.mobility = .5*(self.W.T @ system.A @ self.W+(self.W.T @ system.A @ self.W).T)
        self.source = self.W.T @ system.b
        self.G, self.g0 = v.T @ system.G, v.T @ system.g0
        self.macro_H, self.macro_h, self.constant = system.response.H[:6, :6], system.response.h[:6], system.response.constant
        self.inventory = system.inventory_rows @ v
        self.inventory_flux = system.inventory_rows @ system.b
        self.suprema = np.max(np.abs(v), axis=0)
        self.margin = min(system.reference, system.maximum-system.reference)
        columns = np.column_stack([v, system.operator @ v, -system.b, system.macro_forcing, system.offset_forcing])
        metric = system.chol.T @ columns
        self.null_operator = system.null_basis.T @ metric
        self.residual_root = np.linalg.qr(metric-system.null_basis @ self.null_operator, mode="r")
        self.roundoff_norm = float(np.linalg.norm(metric, ord="fro"))
        if self.error_space is not None:
            self.slow_residual = self.error_space.slow.T @ metric
            tail = metric-system.null_basis @ self.null_operator-self.error_space.slow @ self.slow_residual
            self.tail_residual_root = np.linalg.qr(tail,mode="r")
        self.certificate_key = self.error_space.key if self.error_space is not None else "graph"
        self.fingerprint = _digest(v, np.frombuffer((system.fingerprint+projection+self.certificate_key).encode(), dtype=np.uint8))
        self._dt, self._factor = None, None

    @property
    def dimension(self): return len(self.H)

    def energy(self, a, e):
        return float(.5*a @ self.H @ a+(self.G @ e+self.g0) @ a+
            .5*e @ self.macro_H @ e+self.macro_h @ e+self.constant)

    def initialize(self, initial_deviation, *, macro_strain=None, error_tolerance):
        c = np.asarray(initial_deviation, dtype=float)
        e = np.zeros(6) if macro_strain is None else np.asarray(macro_strain, dtype=float)
        if c.shape != (self.system.n,) or e.shape != (6,) or not np.isfinite(np.r_[c,e]).all(): raise ValueError("Invalid initial field")
        a = np.linalg.solve(self.H, self.basis.values.T @ (self.system.H @ c))
        error = self.system.chol.T @ (c-self.basis.values @ a)
        if self.error_space is None:
            null = self.system.null_basis.T @ error
            orthogonal = float(np.linalg.norm(error-self.system.null_basis @ null))
            bound = EnergyErrorBound(orthogonal, null, float(np.hypot(orthogonal, np.linalg.norm(null))), 0., self.system.fingerprint)
        else:
            null,proxy,tail = self.error_space.decompose(error,self.system.null_basis)
            orthogonal = float(np.hypot(np.linalg.norm(proxy),tail))
            bound = EnergyErrorBound(orthogonal,null,float(np.hypot(orthogonal,np.linalg.norm(null))),0.,self.system.fingerprint,
                proxy,0.,tail,self.certificate_key)
        state = GradientState(a, e, 0., self.energy(a,e), self.system.baseline_inventory+self.inventory @ a, bound, self.fingerprint)
        self.accept(state, error_tolerance)
        return state

    def accept(self, state, tolerance):
        if state.model_fingerprint != self.fingerprint or state.error.system_fingerprint != self.system.fingerprint: raise ValueError("Incompatible certified state")
        if state.error.certificate_key != self.certificate_key: raise ValueError("Incompatible error-coordinate basis")
        if state.coefficients.shape != (self.dimension,) or state.macro_strain.shape != (6,) or state.error.null_coordinates.shape != (self.system.null_basis.shape[1],):
            raise ValueError("Invalid coupled state dimensions")
        if not np.isfinite(np.r_[state.coefficients,state.macro_strain,state.error.null_coordinates,state.error.total_bound,state.error.orthogonal_bound,state.energy_density,state.time]).all():
            raise ValueError("Nonfinite coupled state or certificate")
        if min(state.error.total_bound,state.error.orthogonal_bound,state.time) < 0: raise ValueError("Negative error bound or time")
        required = float(np.hypot(state.error.orthogonal_bound,np.linalg.norm(state.error.null_coordinates)))
        if self.error_space is not None:
            proxy = state.error.spectral_proxy
            if proxy is None or proxy.shape != (self.error_space.count,) or not np.isfinite(np.r_[proxy,state.error.spectral_uncertainty,state.error.spectral_tail]).all():
                raise ValueError("Invalid signed error coordinates")
            if min(state.error.spectral_uncertainty,state.error.spectral_tail) < 0:
                raise ValueError("Negative error uncertainty")
            required_orthogonal = float(np.hypot(np.linalg.norm(proxy)+state.error.spectral_uncertainty,state.error.spectral_tail))
            if state.error.orthogonal_bound+1e-12*max(required_orthogonal,1.) < required_orthogonal:
                raise ValueError("Inconsistent spectral error enclosure")
        if state.error.total_bound+1e-12*max(required,1.) < required:
            raise ValueError("Inconsistent total error enclosure")
        if not np.isfinite(tolerance) or tolerance <= 0: raise ValueError("Positive energy-norm error tolerance required")
        if state.error.total_bound > tolerance: raise EnergyAccuracyError(state.error.total_bound, tolerance)
        if self.suprema @ np.abs(state.coefficients)+self.system.pointwise_constant*state.error.total_bound > self.margin:
            raise ConcentrationInfeasible("Coupled energy error enclosure reaches concentration bounds")

    def trial(self, previous, dt, channel_flux, *, macro_strain=None):
        """Unaccepted candidate for analysis/adaptivity; use step for a qualified state."""
        if previous.model_fingerprint != self.fingerprint: raise ValueError("State/basis mismatch")
        u = np.asarray(channel_flux, dtype=float)
        e = previous.macro_strain if macro_strain is None else np.asarray(macro_strain, dtype=float)
        old = previous.coefficients
        if old.shape != (self.dimension,) or u.shape != (self.source.shape[1],) or e.shape != (6,) or not np.isfinite(np.r_[old,u,e,dt]).all() or dt <= 0: raise ValueError("Invalid coupled control/step")
        if self._dt != dt:
            self._factor = lu_factor(np.eye(self.dimension)+dt*self.mobility @ self.H); self._dt = dt
        g = self.G @ e+self.g0
        a = lu_solve(self._factor, old+dt*(self.source @ u-self.mobility @ g))
        bound = self.error_after(previous.error, old, a, dt, u, e)
        energy = self.energy(a,e); inventory = self.system.baseline_inventory+self.inventory @ a
        state = GradientState(a,e,previous.time+dt,energy,inventory,bound,self.fingerprint)
        gradient = self.H @ a+g
        loss = float(gradient @ self.mobility @ gradient)
        power = float(gradient @ self.source @ u)
        work = self.energy(old,e)-self.energy(old,previous.macro_strain)
        numerical = float(.5*(a-old) @ self.H @ (a-old))
        balance = energy-previous.energy_density-work-dt*(power-loss)+numerical
        residual = (a-old)/dt+self.mobility @ gradient-self.source @ u
        diagnostic = GradientDiagnostics(loss,power,float(work),numerical,float(balance),
            immutable(inventory-previous.component_inventory-dt*self.inventory_flux @ u),float(np.linalg.norm(residual)))
        return state, diagnostic

    def error_after(self, previous_error, old, a, dt, u, e):
        """Full-generator residual bound; also applicable to other approximations in this V."""
        if previous_error.system_fingerprint != self.system.fingerprint: raise ValueError("Error bound belongs to another system")
        if np.shape(old) != (self.dimension,) or np.shape(a) != (self.dimension,) or np.shape(u) != (self.source.shape[1],) or np.shape(e) != (6,) or not np.isfinite(np.r_[old,a,u,e,dt]).all() or dt <= 0:
            raise ValueError("Invalid coupled residual coordinates")
        z = np.r_[np.asarray(a)-old, dt*np.asarray(a), dt*np.asarray(u), dt*np.asarray(e), dt]
        rn = self.null_operator @ z; rp = float(np.linalg.norm(self.residual_root @ z))
        padding = 128*np.finfo(float).eps*self.roundoff_norm*np.linalg.norm(z)
        if self.error_space is not None:
            if previous_error.certificate_key != self.certificate_key: raise ValueError("Cannot silently change error dynamics")
            proxy,uncertainty,tail = self.error_space.advance(previous_error,self.slow_residual @ z,
                float(np.linalg.norm(self.tail_residual_root @ z)),dt,padding)
            orthogonal = float(np.hypot(np.linalg.norm(proxy)+uncertainty,tail))
            null = previous_error.null_coordinates-rn
            total = float(np.hypot(orthogonal,np.linalg.norm(null)))
            return EnergyErrorBound(orthogonal,null,total,float(np.hypot(rp,np.linalg.norm(rn))),self.system.fingerprint,
                proxy,uncertainty,tail,self.certificate_key)
        orthogonal = (previous_error.orthogonal_bound+rp+padding)/(1+dt*self.system.gap_lower)+padding
        null = previous_error.null_coordinates-rn
        total = float(np.hypot(orthogonal, np.linalg.norm(null)))
        return EnergyErrorBound(float(orthogonal), null, total, float(np.hypot(rp, np.linalg.norm(rn))), self.system.fingerprint)

    def step(self, previous, dt, channel_flux, *, error_tolerance, macro_strain=None):
        state, diagnostic = self.trial(previous,dt,channel_flux,macro_strain=macro_strain)
        self.accept(state,error_tolerance)
        return state, diagnostic

    def transfer_to(self, previous, new_model):
        if previous.model_fingerprint != self.fingerprint or new_model.system.fingerprint != self.system.fingerprint:
            raise ValueError("Cannot transfer across a different physical system")
        if new_model.projection != self.projection: raise ValueError("Nested transfer cannot silently change projection method")
        if new_model.certificate_key != self.certificate_key: raise ValueError("Nested transfer cannot change certificate coordinates")
        old = self.basis.values
        if new_model.dimension < self.dimension or not np.array_equal(new_model.basis.values[:, :self.dimension], old):
            raise ValueError("Exact transfer requires a nested basis with preserved columns")
        a = np.r_[previous.coefficients, np.zeros(new_model.dimension-self.dimension)]
        e = previous.macro_strain
        energy = new_model.energy(a,e)
        inventory = self.system.baseline_inventory+new_model.inventory @ a
        state = GradientState(a,e,previous.time,energy,inventory,previous.error,new_model.fingerprint)
        return state, {"energy_jump":energy-previous.energy_density,
            "mass_jump":(inventory-previous.component_inventory).tolist(),
            "error_bound_before":previous.error.total_bound,"error_bound_after":state.error.total_bound,
            "old_modes":self.dimension,"new_modes":new_model.dimension}

    def transfer_preserving_state(self, previous, new_model):
        """Non-nested restart only when its space contains the current field.

        Any floating transfer defect is added to the old certificate, including
        its null coordinates; accumulated error is never reset to zero.
        """
        if previous.model_fingerprint != self.fingerprint or new_model.system.fingerprint != self.system.fingerprint or new_model.projection != self.projection or new_model.certificate_key != self.certificate_key:
            raise ValueError("Restart must retain the same physical system/projection")
        concentration = self.basis.values @ previous.coefficients
        a = np.linalg.solve(new_model.H,new_model.basis.values.T @ (self.system.H @ concentration))
        defect = self.system.chol.T @ (concentration-new_model.basis.values @ a)
        defect_norm = float(np.linalg.norm(defect))
        if defect_norm > 1e-10*max(self.system.norm(concentration),1.):
            raise ValueError("Restart space does not contain the current physical field")
        null_defect = self.system.null_basis.T @ defect
        orthogonal_defect = float(np.linalg.norm(defect-self.system.null_basis @ null_defect))
        if self.error_space is not None:
            proxy_defect = self.error_space.slow.T @ defect
            tail_defect = float(np.linalg.norm(defect-self.system.null_basis @ null_defect-self.error_space.slow @ proxy_defect))
            proxy = previous.error.spectral_proxy+proxy_defect
            uncertainty = previous.error.spectral_uncertainty
            tail = previous.error.spectral_tail+tail_defect
            orthogonal = float(np.hypot(np.linalg.norm(proxy)+uncertainty,tail))
            null = previous.error.null_coordinates+null_defect
            bound = EnergyErrorBound(orthogonal,null,float(np.hypot(orthogonal,np.linalg.norm(null))),previous.error.residual_norm,
                self.system.fingerprint,proxy,uncertainty,tail,self.certificate_key)
        else:
            orthogonal = previous.error.orthogonal_bound+orthogonal_defect
            null = previous.error.null_coordinates+null_defect
            # A scalar floor avoids reducing the enclosure by rounding-induced
            # cancellation of null coordinates during a graph-bound restart.
            floor = previous.error.total_bound+defect_norm
            orthogonal = max(orthogonal,float(np.sqrt(max(floor**2-float(null @ null),0.))))
            bound = EnergyErrorBound(orthogonal,null,float(np.hypot(orthogonal,np.linalg.norm(null))),previous.error.residual_norm,self.system.fingerprint)
        e = previous.macro_strain; energy = new_model.energy(a,e)
        inventory = self.system.baseline_inventory+new_model.inventory @ a
        state = GradientState(a,e,previous.time,energy,inventory,bound,new_model.fingerprint)
        return state,{"energy_jump":energy-previous.energy_density,"mass_jump":(inventory-previous.component_inventory).tolist(),
            "error_bound_before":previous.error.total_bound,"error_bound_after":bound.total_bound,
            "transfer_defect_H_norm":defect_norm,"old_modes":self.dimension,"new_modes":new_model.dimension,
            "transfer_kind":"state-containing restart; defect added, no error reset"}
