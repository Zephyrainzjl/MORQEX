"""Explicit property conditioning for a pretrained categorical diffusion prior.

This is a sampling constraint, not evidence of learned counterfactual physics.
For phase scores s and target fractions f, solve the convex dual
  mean(logsumexp((s+b)/T)) - dot(f,b)/T
for global chemical potentials b. It is the entropic relaxation of assigning
labels with prescribed phase marginals. Pairwise score differences are preserved
up to a phase-wise bias; no voxel is independently shuffled or resampled.

Unlike DPS, this operation does not backpropagate through the denoiser. It is
inspired by measurement-conditioned sampling, not an implementation of DPS.
"""
import torch


@torch.no_grad()
def composition_potential(scores, fractions, temperature=.1, iterations=18,
                          max_points=65536):
    """Return biased scores and auditable soft/hard composition diagnostics.

    scores: (1, phases, z, y, x). Fractions must be strictly positive and sum to
    one. Dual fitting uses a deterministic strided subset for large volumes;
    measured full-volume hard fractions are always reported. No exact-count
    guarantee is made. Temperature is in the same units as scores.
    """
    if scores.ndim != 5 or scores.shape[0] != 1:
        raise ValueError('Expected scores with shape (1,phases,z,y,x)')
    if temperature <= 0 or iterations < 1 or max_points < 1:
        raise ValueError('Positive temperature, iterations and max_points required')
    target = torch.as_tensor(fractions, device=scores.device, dtype=torch.float32)
    k = scores.shape[1]
    if target.shape != (k,) or not torch.isfinite(target).all() or (target <= 0).any() or abs(float(target.sum())-1) > 1e-5:
        raise ValueError('Strictly positive phase fractions summing to one required')
    flat = scores[0].float().reshape(k,-1)
    stride = max(1,(flat.shape[1]+max_points-1)//max_points)
    subset = flat[:,::stride]
    # Start from log-ratio balancing, then damped Newton on k-1 free potentials.
    bias = torch.zeros(k, device=scores.device)
    eye = torch.eye(k-1,device=scores.device)
    def objective(b):
        return torch.logsumexp((subset+b[:,None])/temperature,dim=0).mean()-(target*b).sum()/temperature
    for _ in range(iterations):
        p = ((subset+bias[:,None])/temperature).softmax(0)
        mean = p.mean(1)
        residual = mean-target
        if float(residual.abs().max()) < 2e-5:
            break
        hessian = (torch.diag(mean)-(p@p.T)/p.shape[1])/temperature
        delta = torch.linalg.solve(hessian[:-1,:-1]+eye*1e-5,residual[:-1])
        delta = delta.clamp(-2.,2.)
        old = objective(bias)
        accepted = False
        for scale in (1.,.5,.25,.125,.0625):
            proposal = bias.clone();proposal[:-1] -= scale*delta
            if objective(proposal) < old:
                bias = proposal;accepted = True;break
        if not accepted:
            bias += temperature*(target.log()-mean.clamp_min(1e-10).log())
            bias -= bias[-1].clone()
    adjusted = scores+bias[None,:,None,None,None].to(scores.dtype)
    labels = adjusted.argmax(1)
    hard = torch.stack([(labels==i).float().mean() for i in range(k)])
    soft = ((subset+bias[:,None])/temperature).softmax(0).mean(1)
    return adjusted,dict(phase_bias=bias.cpu().tolist(),soft_fractions=soft.cpu().tolist(),
                         hard_fractions=hard.cpu().tolist(),temperature=float(temperature),
                         fitted_points=int(subset.shape[1]),method='entropic composition potential')
