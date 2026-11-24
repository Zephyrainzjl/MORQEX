"""Signed low-dimensional error dynamics with a bounded unresolved complement.

This small-system diagnostic constructs its spectral subspaces densely OFFLINE.
Numerical eigenvectors select a space; complementary coercivity uses an explicit
matrix Gershgorin lower bound, not unverified numerical eigenvalues. Off-block
coupling is retained in the norm inequalities. FP64 padding is not directed
interval arithmetic. No generic large-N certification or novelty is claimed.
"""
import numpy as np
import hashlib
from scipy.linalg import null_space, eigh, cho_factor, cho_solve


class SpectralErrorSpace:
    def __init__(self, system, retained_modes):
        null = system.null_basis
        complement = null_space(null.T)
        q = system.chol.T @ system.A @ system.chol
        q = .5*(q+q.T)
        positive = .5*(complement.T @ q @ complement+(complement.T @ q @ complement).T)
        if len(positive):
            _,vectors = eigh(positive)
            ordered = complement @ vectors
        else: ordered = np.empty((system.n,0))
        self.count = min(int(retained_modes),ordered.shape[1])
        if self.count < 0: raise ValueError("Nonnegative error-coordinate count required")
        self.slow,self.tail = ordered[:,:self.count],ordered[:,self.count:]
        self.configure_bounds(system)

    def configure_bounds(self, system):
        combined = np.column_stack([system.null_basis,self.slow,self.tail])
        if combined.shape != (system.n,system.n) or np.linalg.norm(combined.T @ combined-np.eye(system.n)) > 1e-10:
            raise ValueError("Error subspaces must form an orthonormal decomposition")
        q = system.chol.T @ system.A @ system.chol
        q = .5*(q+q.T)
        self.slow_operator = .5*(self.slow.T @ q @ self.slow+(self.slow.T @ q @ self.slow).T)
        scale = max(float(np.linalg.norm(q,ord="fro")),1e-30)
        padding = 256*np.finfo(float).eps*system.n*scale
        if self.tail.shape[1]:
            tail_operator = self.tail.T @ q @ self.tail
            diagonal = np.diag(tail_operator)
            gershgorin = float(np.min(diagonal-(np.sum(np.abs(tail_operator),axis=1)-np.abs(diagonal))))-padding
            self.tail_lower = max(float(system.gap_lower),gershgorin)
            self.coupling = float(np.linalg.norm(self.slow.T @ q @ self.tail,ord="fro"))+padding
        else: self.tail_lower,self.coupling = np.inf,0.
        digest = hashlib.sha256(self.slow.tobytes()+self.tail.tobytes()+self.slow_operator.tobytes()).hexdigest()
        self.key = f"spectral-error-{self.count}-{digest[:16]}"
        self._dt,self._factor = None,None

    def decompose(self, error, null_basis):
        null = null_basis.T @ error
        proxy = self.slow.T @ error
        tail = float(np.linalg.norm(error-null_basis @ null-self.slow @ proxy))
        return null,proxy,tail

    def advance(self, previous, slow_residual, tail_residual_norm, dt, padding):
        proxy_before = previous.spectral_proxy
        if proxy_before is None or len(proxy_before) != self.count: raise ValueError("Error coordinates do not match retained space")
        if self._dt != dt:
            self._factor = cho_factor(np.eye(self.count)+dt*self.slow_operator) if self.count else None;self._dt = dt
        proxy = cho_solve(self._factor,proxy_before-slow_residual) if self.count else np.empty(0)
        if not self.tail.shape[1]:
            return proxy,previous.spectral_uncertainty+padding,0.
        alpha = 1/(1+dt*self.tail_lower)
        coupling = self.coupling
        denominator = 1-alpha*dt**2*coupling**2
        if denominator <= 0: raise ValueError("Error-subspace coupling too large; use graph bound or reduce dt")
        slow_previous = previous.spectral_uncertainty
        tail = alpha*(previous.spectral_tail+tail_residual_norm+padding+
            dt*coupling*(np.linalg.norm(proxy)+slow_previous+padding))/denominator
        slow = slow_previous+dt*coupling*tail+padding
        return proxy,float(slow),float(tail)
