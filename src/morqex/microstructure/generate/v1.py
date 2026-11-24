"""Research implementation; not a pretrained model or a proven speed/quality claim.

Train a shared orientation-conditioned 2D DDPM, synthesize a coherent 3D field
by averaging denoising scores in the three orthogonal planes at each DDIM step.
Uses upstream Diffusers UNet2DModel/DDPMScheduler/DDIMScheduler.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import time
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from morqex.microstructure.volume import Volume
SOURCE_DIGEST=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

@dataclass(frozen=True)
class DiffusionConfig:
    image_size: int = 64
    phases: int = 3
    channels: tuple = (32, 64, 96)
    layers: int = 2
    batch_size: int = 16
    steps: int = 20000
    time_budget_seconds: float = 3300.
    learning_rate: float = 2e-4
    diffusion_steps: int = 1000
    condition_dropout: float = .15
    reference_dropout: float = 0.
    ema_decay: float = .995
    checkpoint_every: int = 500
    seed: int = 42
    device: str = 'cuda'
    mixed_precision: bool = True
    fraction_weight: float = .05
    attention: bool = True
    min_snr_gamma: float = 5.
    morphology_variants: int = 0
    morphology_sigma: float = 1.5

    def __post_init__(self):
        factor = 2**(len(self.channels)-1)
        if self.image_size < factor or self.image_size % factor:
            raise ValueError('image_size must be divisible by UNet downsampling factor')
        if self.phases < 2 or self.batch_size < 1 or self.steps < 1 or self.time_budget_seconds <= 0:
            raise ValueError('Invalid training configuration')
        if any(c % 8 for c in self.channels) or not 0 <= self.condition_dropout < 1:
            raise ValueError('Channels must be multiples of 8; dropout in [0,1)')
        if not 0 <= self.reference_dropout < 1:
            raise ValueError('reference_dropout must be in [0,1)')
        if self.diffusion_steps < 2 or self.checkpoint_every < 1 or not 0 <= self.ema_decay < 1:
            raise ValueError('Invalid diffusion/checkpoint/EMA settings')

def _model(cfg):
    blocks = len(cfg.channels)
    return UNet2DModel(sample_size=cfg.image_size,
        in_channels=3*cfg.phases+3, out_channels=cfg.phases,
        block_out_channels=tuple(cfg.channels), layers_per_block=cfg.layers,
        down_block_types=tuple('AttnDownBlock2D' if cfg.attention and i >= blocks-2 else 'DownBlock2D' for i in range(blocks)),
        up_block_types=tuple('AttnUpBlock2D' if cfg.attention and i < 2 else 'UpBlock2D' for i in range(blocks)), norm_num_groups=8,
        add_attention=cfg.attention)

def _read(path):
    import tifffile
    from PIL import Image
    path = Path(path)
    if path.suffix == '.npy': a = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in ('.tif', '.tiff'): a = tifffile.imread(path)
    else: a = np.asarray(Image.open(path))
    if a.ndim != 2:
        raise ValueError(f'{path}: supply segmented scalar 2D slices, got {a.shape}')
    return a

def _images(paths, cfg, labels=None):
    if len(paths) not in (1, 3):
        raise ValueError('Supply one isotropic image, or three ordered (xy,xz,yz) images')
    arrays = [_read(p) for p in paths]
    actual = np.unique(np.concatenate([np.unique(a) for a in arrays]))
    labels = actual if labels is None else np.asarray(labels)
    if len(labels) != cfg.phases or not np.isin(actual, labels).all():
        raise ValueError(f'Expected {cfg.phases} labels {labels.tolist()}, got {actual.tolist()}')
    tensors = []
    for a in arrays:
        if min(a.shape) < cfg.image_size:
            raise ValueError('Training image smaller than image_size')
        a = np.searchsorted(labels, a)
        tensors.append(torch.as_tensor(a.astype(np.int64), device=cfg.device))
    return tensors if len(tensors) == 3 else tensors*3, labels.tolist()

def _crop(a, n, size, phases):
    rows = torch.randint(a.shape[0]-size+1, (n,), device=a.device)
    cols = torch.randint(a.shape[1]-size+1, (n,), device=a.device)
    r = torch.arange(size, device=a.device)
    x = a[rows[:,None,None]+r[None,:,None], cols[:,None,None]+r[None,None,:]]
    return F.one_hot(x, phases).permute(0,3,1,2).float()*2-1

def _input(noisy, reference, fractions, axis):
    b, _, h, w = noisy.shape
    fraction_map = fractions[:,:,None,None].expand(b,-1,h,w)
    orientation = F.one_hot(torch.full((b,), axis, device=noisy.device), 3).float()
    orientation = orientation[:,:,None,None].expand(b,-1,h,w)
    return torch.cat((noisy, reference, fraction_map, orientation), 1)

def _atomic_save(state, path):
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)

def train(paths, run_dir, config=None, resume=None):
    """Crops on device, BF16 AMP, EMA, deadline, full optimizer/RNG resumption.

    Input images are segmented labels; grayscale CT must be segmented explicitly.
    A patch's observed fractions supervise the continuous composition condition.
    Multiple fraction datasets must be used to learn a broad composition range.
    """
    cfg = config or DiffusionConfig()
    run = Path(run_dir).resolve()
    package = Path(__file__).resolve().parents[4]
    if run == package or package in run.parents:
        raise ValueError('run_dir must be outside gem package')
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; explicitly select device=cpu for smoke tests')
    images, labels = _images(paths, cfg)
    variants = [images]
    if cfg.morphology_variants:
        # Only 2D transformations of supplied slices. No held-out 3D information.
        from scipy.ndimage import distance_transform_edt
        rng = np.random.default_rng(cfg.seed)
        distances = []
        for img in images:
            a = img.cpu().numpy()
            distances.append(np.stack([distance_transform_edt(a == i)-distance_transform_edt(a != i)
                                       for i in range(cfg.phases)]))
        for _ in range(cfg.morphology_variants):
            bias = rng.normal(0,cfg.morphology_sigma,(cfg.phases,1,1))
            variants.append([torch.as_tensor((d+bias).argmax(0),device=device) for d in distances])
    net = _model(cfg).to(device, memory_format=torch.channels_last)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    optim = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate)
    scheduler = DDPMScheduler(num_train_timesteps=cfg.diffusion_steps, beta_schedule='squaredcos_cap_v2',
                              prediction_type='epsilon', clip_sample=False)
    start = 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        for key in ('image_size','phases','channels','layers','diffusion_steps'):
            if state['config'][key] != asdict(cfg)[key]: raise ValueError(f'Resume mismatch: {key}')
        if state['labels'] != labels: raise ValueError('Resume label mismatch')
        net.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        optim.load_state_dict(state['optimizer']); start = state['step']
        torch.set_rng_state(state['rng'].cpu())
        if device.type == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    amp = device.type == 'cuda' and cfg.mixed_precision
    started = time.monotonic()
    (run/'config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf8')
    step = start
    completed=start
    try:
        with (run/'metrics.jsonl').open('a', encoding='utf8') as log:
            for step in range(start+1, cfg.steps+1):
                axis = (step-1) % 3
                source = variants[torch.randint(len(variants),()).item()][axis]
                clean = _crop(source, cfg.batch_size, cfg.image_size, cfg.phases)
                ref = _crop(source, cfg.batch_size, cfg.image_size, cfg.phases)
                # Flips preserve orientation; no rotations for anisotropic training.
                if torch.rand(()) < .5: clean = clean.flip(-1); ref = ref.flip(-1)
                fractions = ((clean+1)/2).mean((2,3))
                keep = (torch.rand(cfg.batch_size, device=device) >= cfg.condition_dropout).float()
                # Additional image-only dropout forces use of the numerical
                # condition, with the same single denoiser forward/backward.
                image_keep=keep
                if cfg.reference_dropout:
                    image_keep=keep*(torch.rand(cfg.batch_size,device=device)>=cfg.reference_dropout)
                ref = ref * image_keep[:,None,None,None]
                condition = fractions * keep[:,None]
                t = torch.randint(cfg.diffusion_steps, (cfg.batch_size,), device=device)
                noise = torch.randn_like(clean)
                noisy = scheduler.add_noise(clean, noise, t)
                optim.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    pred = net(_input(noisy, ref, condition, axis), t).sample
                    alpha = scheduler.alphas_cumprod.to(device)[t,None,None,None]
                    snr = alpha.flatten()/(1-alpha.flatten()).clamp_min(1e-8)
                    weights = snr.clamp(max=cfg.min_snr_gamma)/snr.clamp_min(1e-8)
                    loss_noise = ((pred.float()-noise).square().mean((1,2,3))*weights).mean()
                    x0 = (noisy-(1-alpha).sqrt()*pred.float())/alpha.sqrt()
                    estimate = ((x0+1)/2).clamp(0,1).mean((2,3))
                    loss_fraction = ((estimate-fractions)**2).mean()
                    loss = loss_noise + cfg.fraction_weight*loss_fraction
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                optim.step()
                with torch.no_grad():
                    for p, q in zip(ema.parameters(), net.parameters()): p.lerp_(q, 1-cfg.ema_decay)
                completed=step
                elapsed = time.monotonic()-started
                log.write(json.dumps({'step':step, 'loss':float(loss.detach()),
                    'noise_loss':float(loss_noise.detach()), 'fraction_loss':float(loss_fraction.detach()),
                    'elapsed_seconds':elapsed})+'\n'); log.flush()
                if step % cfg.checkpoint_every == 0 or elapsed >= cfg.time_budget_seconds:
                    _save(run, cfg, net, ema, optim, step, labels, paths, elapsed)
                if elapsed >= cfg.time_budget_seconds: break
    finally:
        _save(run, cfg, net, ema, optim, completed, labels, paths, time.monotonic()-started)
    return run/'checkpoint.pt'

def _save(run, cfg, net, ema, optim, step, labels, paths, elapsed):
    _atomic_save({'config':asdict(cfg), 'model':net.state_dict(), 'ema':ema.state_dict(),
        'optimizer':optim.state_dict(), 'step':step, 'labels':labels,
        'paths':[str(Path(p).resolve()) for p in paths], 'rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'elapsed_seconds':elapsed, 'torch_version':str(torch.__version__),
        'source_sha256':SOURCE_DIGEST,'fraction_estimator':'linear_clipped'}, run/'checkpoint.pt')

def _fractions(values, phases, device):
    a = torch.as_tensor(values, dtype=torch.float32, device=device)
    if a.shape != (phases,) or not torch.isfinite(a).all() or (a < 0).any() or abs(float(a.sum())-1) > 1e-5:
        raise ValueError('fractions must be nonnegative, length phases, and sum to 1')
    return a

def _starts(n,patch,overlap):
    starts = list(range(0,max(1,n-patch+1),patch-overlap))
    if starts[-1] != n-patch: starts.append(n-patch)
    return starts

def _plane_score(model,planes,reference,frac,axis,t,cfg,guidance,slice_batch,overlap):
    """Overlap-add scores, permitting volumes larger than the trained 2D crop."""
    n, channels, h, w = planes.shape
    patch = cfg.image_size
    scores = torch.zeros_like(planes)
    weights = torch.zeros((1,1,h,w),device=planes.device)
    window = torch.hann_window(patch+2,periodic=False,device=planes.device)[1:-1]
    window = (window[:,None]*window[None,:])[None,None].clamp_min(1e-4)
    amp = planes.device.type == 'cuda' and cfg.mixed_precision
    for row in _starts(h,patch,overlap):
        for col in _starts(w,patch,overlap):
            for start in range(0,n,slice_batch):
                batch = planes[start:start+slice_batch,:,row:row+patch,col:col+patch]
                b = len(batch)
                ref = reference.expand(b,-1,-1,-1)
                f = frac.expand(b,-1)
                with torch.autocast(device_type=planes.device.type,dtype=torch.bfloat16,enabled=amp):
                    cond = model(_input(batch,ref,f,axis),t).sample
                    if guidance != 1:
                        uncond = model(_input(batch,torch.zeros_like(ref),torch.zeros_like(f),axis),t).sample
                        cond = uncond+guidance*(cond-uncond)
                scores[start:start+b,:,row:row+patch,col:col+patch] += cond.float()*window
            weights[:,:,row:row+patch,col:col+patch] += window
    return scores/weights

@torch.inference_mode()
def sample(checkpoint, paths=None, fractions=None, size=None, steps=50, guidance=1.5,
           seed=42, device='cuda', slice_batch=16, spacing=(1.,1.,1.), exact_fractions=False,
           prior=None, strength=1., overlap=16, composition_guidance=False,
           composition_temperature=.05, composition_start=.35):
    """One/three 2D conditions -> 3D labels. Optional GAN volume as diffusion prior.

    exact_fractions is explicit quota postprocessing, not learned accuracy.
    Inputs are statistical exemplars, not registered spatial constraints.
    composition_guidance applies explicit chemical-potential conditioning inside
    DDIM, without retraining or additional denoiser evaluations. Its measured
    hard fractions are reported; it is not learned condition accuracy.
    """
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = DiffusionConfig(**{**state['config'], 'device':device})
    size = size or cfg.image_size
    if size < cfg.image_size:
        raise ValueError('Volume size must be >= training patch size')
    overlap = min(overlap,cfg.image_size//2)
    if overlap < 0 or overlap >= cfg.image_size: raise ValueError('Invalid overlap')
    if steps < 1 or steps > cfg.diffusion_steps or slice_batch < 1 or not 0 < strength <= 1:
        raise ValueError('Invalid sampling settings')
    if not 0 <= composition_start <= 1 or composition_temperature <= 0:
        raise ValueError('Invalid composition guidance settings')
    torch.manual_seed(seed)
    model = _model(cfg).to(device).eval()
    model.load_state_dict(state['ema'])
    images, labels = _images(paths or state['paths'], cfg, state['labels'])
    if fractions is None:
        fractions = [float(np.mean([(a == i).float().mean().item() for a in images])) for i in range(cfg.phases)]
    frac = _fractions(fractions, cfg.phases, device)
    refs = [_crop(a, 1, cfg.image_size, cfg.phases) for a in images]
    scheduler = DDIMScheduler(num_train_timesteps=cfg.diffusion_steps, beta_schedule='squaredcos_cap_v2',
                              prediction_type='epsilon', clip_sample=False)
    scheduler.set_timesteps(steps, device=device)
    times = scheduler.timesteps[-max(1, round(steps*strength)):]
    x = torch.randn((1,cfg.phases,size,size,size), device=device)
    if prior is not None:
        a = np.asarray(prior.data if isinstance(prior, Volume) else prior)
        if a.shape != (size,size,size) or not np.isin(a, labels).all():
            raise ValueError('Prior shape/labels mismatch')
        p = torch.as_tensor(np.searchsorted(labels,a), device=device)
        clean = F.one_hot(p, cfg.phases).permute(3,0,1,2)[None].float()*2-1
        x = scheduler.add_noise(clean, x, times[:1])
    elif strength != 1:
        raise ValueError('strength<1 requires prior')
    amp = torch.device(device).type == 'cuda' and cfg.mixed_precision
    composition_trace = []
    for index,t in enumerate(times):
        score = torch.zeros_like(x)
        for axis in range(3):
            # plane axis 0: xy (normal z), 1: xz, 2: yz.
            planes = x[0].movedim(axis+1, 0).contiguous()
            prediction = _plane_score(model,planes,refs[axis],frac,axis,t,cfg,guidance,slice_batch,overlap)
            score[0] += prediction.movedim(0,axis+1)
        prediction = score/3
        if composition_guidance and (index+1)/len(times) >= composition_start:
            from .conditioning import composition_potential
            alpha = scheduler.alphas_cumprod[t].to(x.device)
            clean = (x-(1-alpha).sqrt()*prediction)/alpha.sqrt()
            temperature = max(composition_temperature,.3*(1-(index+1)/len(times)))
            conditioned,diagnostic = composition_potential(clean,frac,temperature)
            # Re-express the constrained x0 as epsilon for the same DDIM step.
            prediction = (x-alpha.sqrt()*conditioned)/(1-alpha).sqrt().clamp_min(1e-8)
            composition_trace.append(dict(timestep=int(t),**diagnostic))
        x = scheduler.step(prediction, t, x).prev_sample
    logits = x[0].float().cpu().numpy()
    phase = logits.argmax(0)
    if exact_fractions:
        phase = _quota(logits, frac.cpu().numpy())
    return Volume(_label_array(labels)[phase], spacing=spacing, metadata={
        'model':'v1', 'checkpoint':str(Path(checkpoint).resolve()), 'seed':seed,
        'steps':steps, 'guidance':guidance, 'requested_fractions':frac.cpu().tolist(),
        'quota_postprocessing':exact_fractions, 'training_step':state['step'],
        'composition_guidance':composition_guidance,'composition_trace':composition_trace})

def _quota(logits, fractions):
    scores = logits.reshape(len(fractions), -1)
    n = scores.shape[1]
    raw = fractions*n
    counts = np.floor(raw).astype(int)
    counts[np.argsort(-(raw-counts))[:n-counts.sum()]] += 1
    out = np.full(n, -1, np.int32)
    # Greedy confidence ordering, exact counts to within one voxel.
    for phase in np.argsort(counts)[:-1]:
        available = np.flatnonzero(out < 0)
        count = counts[phase]
        if count:
            margin = scores[phase,available] - np.max(np.delete(scores[:,available], phase, axis=0), axis=0)
            chosen = available[np.argpartition(-margin, count-1)[:count]]
            out[chosen] = phase
    out[out < 0] = np.argsort(counts)[-1]
    return out.reshape(logits.shape[1:])

def _label_array(labels):
    dtype=np.result_type(np.min_scalar_type(min(labels)),np.min_scalar_type(max(labels)))
    return np.asarray(labels,dtype=dtype)

def tune_batch_size(config, target_fraction=.88, maximum=512):
    """Measure actual forward/backward peak; leave headroom for optimizer/EMA.

    Call before the real run. No dummy persistent memory allocations.
    """
    from dataclasses import replace
    if not .2 <= target_fraction <= .95: raise ValueError('target_fraction must be .2..95')
    device = torch.device(config.device)
    if device.type != 'cuda': raise ValueError('CUDA required')
    model = _model(config).to(device,memory_format=torch.channels_last)
    optim = torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    ema = copy.deepcopy(model).requires_grad_(False)
    total = torch.cuda.get_device_properties(device).total_memory
    records = []
    best = 1
    batch = 8
    upper=None
    cap=min(total*target_fraction,torch.cuda.mem_get_info(device)[0]*target_fraction)
    while batch <= maximum:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(batch,3*config.phases+3,config.image_size,config.image_size,device=device)
            started = time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.mixed_precision):
                loss = model(x,torch.ones(batch,device=device,dtype=torch.long)).sample.float().square().mean()
            loss.backward(); optim.step(); optim.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            records.append({'batch':batch,'peak_bytes':peak,'seconds':time.monotonic()-started})
            del loss,x
            if peak > cap:
                upper=batch
                break
            best = batch
            batch = int(batch*1.5)//8*8
            if batch <= best: batch = best+8
        except torch.cuda.OutOfMemoryError:
            optim.zero_grad(set_to_none=True)
            upper=batch
            break
    if upper is not None:
        # Refine the coarse 1.5x search instead of leaving a large VRAM gap.
        for _ in range(4):
            trial=((best+upper)//2)//8*8
            if trial<=best:break
            try:
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                x=torch.randn(trial,3*config.phases+3,config.image_size,config.image_size,device=device)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.mixed_precision):
                    loss=model(x,torch.ones(trial,device=device,dtype=torch.long)).sample.float().square().mean()
                loss.backward();optim.step();optim.zero_grad(set_to_none=True)
                torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
                records.append({'batch':trial,'peak_bytes':peak,'refinement':True})
                del x,loss
                if peak<=cap:best=trial
                else:upper=trial
            except torch.cuda.OutOfMemoryError:
                optim.zero_grad(set_to_none=True);upper=trial
    del model,optim,ema
    torch.cuda.empty_cache()
    return replace(config,batch_size=best),records
