
import torch as th
import numpy as np

from diff_rl.common.helpers import append_dims, append_zero, mean_flat


def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings

class Consistency_Model:
    def __init__(
        self,
        action_dim,
        device,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        ts=None,
        sampler="onestep", 
        clip_denoised=True,
    ):
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.num_timesteps = 40

        self.device = device
        
        self.sampler = sampler
        self.steps = steps
        self.ts = [0, 20, 40]
        # this is for sample
        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings_for_boundary_condition(self, sigma): # get the values of c_...
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)

    def consistency_losses(
        self,
        model,
        x_start,
        num_scales,
        state=None,
        target_model=None,
        noise=None,
    ):

        if noise is None:
            noise = th.randn_like(x_start) # make this noise a \nabla[Q(s, a)]

        dims = x_start.ndim

        def denoise_fn(x, t, state=None):
            return self.denoise(model, x, t, state)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None):
            return self.denoise(target_model, x, t, state)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0): # x_t (x_t_n+1), t (t_n+1), t2 (t_n), x_start
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims) # according to Training Consistency Models in Isolation

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        ) # random 

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        ) # t (t_n+1) and t2(t_n) here is randomly generated based on indices
        t = t**self.rho # t_n+1 

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho # t_n
        
        x_t = x_start + noise * append_dims(t, dims) # basically is x_start + noise

        dropout_state = th.get_rng_state() # random number generator
        distiller = denoise_fn(x_t, t, state) # # predicted target based on t = t_n+1

        # x_t2 = euler_solver(x_t, t, t2, x_start).detach()
        # equals to: x_t2 = x_start + noise * append_dims(t2, dims)
        x_t2 = x_start + noise * append_dims(t2, dims)

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state) # predicted target based on t2=t_n
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t) # sigmas**-2
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # lambda(t_n), get different weights based on snrs: snrs + 1.0 / sigma_data**2

        consistency_diffs = (distiller - distiller_target) ** 2 # get the consistency difference
        consistency_loss = mean_flat(consistency_diffs) * weights # weighted average as loss

        recon_diffs = (distiller - x_start) ** 2 # this is reconstruction loss
        recon_loss = mean_flat(recon_diffs) * weights

        terms = {}
        terms["loss"] = recon_loss
        terms["consistency_loss"] = consistency_loss
        terms["recon_loss"] = recon_loss

        return terms

    def denoise(self, model, x_t, sigmas, state):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        
        model_output = model(c_in * x_t, rescaled_t, state)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return model_output, denoised

    def sample(self, model, state):
        x_0 = self.sample_onestep(model, state).clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, model, state):
        x_T = th.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]]) # stands for sigma input
        return self.denoise(model, x_T, self.sigmas[0] * s_in, state)[1]
    
    def sample_multistep(self, model, state):
        x_T = th.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        x = self.denoise(model, x_T, self.sigmas[0] * s_in, state)[1]

        for i in range(len(self.ts)):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            t = np.clip(t, self.sigma_min, self.sigma_max)
            x = self.denoise(model, x, t * s_in, state)[1]

        return x