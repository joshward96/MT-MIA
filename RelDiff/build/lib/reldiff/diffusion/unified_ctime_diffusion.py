from itertools import chain
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from reldiff.diffusion.noise_schedule import (
    PowerMeanNoise,
    PowerMeanNoise_PerColumn,
    PowerMeanUnified,
    LogLinearNoise,
    LogLinearNoise_PerColumn,
    LogLinearUnified,
    NoNoise,
)
from reldiff.data.utils import TensorDequantizer
from reldiff.data.dataloader import get_subgraph_dataloader

"""
“Our implementation of the continuous-time masked diffusion is inspired by https://arxiv.org/abs/2406.07524's implementation at [https://github.com/kuleshov-group/mdlm], with modifications to support data distributions that include categorical dimensions of different sizes.”
"""

S_churn = 1
S_min = 0
S_max = float("inf")
S_noise = 1


class UnifiedCtimeDiffusion(torch.nn.Module):
    def __init__(
        self,
        num_classes: np.array,
        num_numerical_features: int,
        denoise_fn,
        y_only_model,
        num_timesteps=1000,
        scheduler="power_mean",
        cat_scheduler="log_linear",
        noise_dist="uniform",
        edm_params={},
        noise_dist_params={},
        noise_schedule_params={},
        sampler_params={},
        device=torch.device("cpu"),
        calibrate_losses=False,
        proportions=None,
        **kwargs,
    ):
        super(UnifiedCtimeDiffusion, self).__init__()

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes  # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = (
            torch.from_numpy(
                np.concatenate(
                    [
                        num_classes[i].repeat(num_classes[i])
                        for i in range(len(num_classes))
                    ]
                )
            ).to(device)
            if len(num_classes) > 0
            else torch.tensor([]).to(device).int()
        )
        self.mask_index = torch.tensor(self.num_classes).long().to(device)
        self.neg_infinity = -1000000.0
        self.num_classes_w_mask = tuple(self.num_classes + 1)

        offsets = np.cumsum(self.num_classes)
        offsets = np.append([0], offsets)
        self.slices_for_classes = []
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(offsets).to(device)

        offsets = np.cumsum(self.num_classes) + np.arange(1, len(self.num_classes) + 1)
        offsets = np.append([0], offsets)
        self.slices_for_classes_with_mask = []
        for i in range(1, len(offsets)):
            self.slices_for_classes_with_mask.append(
                np.arange(offsets[i - 1], offsets[i])
            )

        self._denoise_fn = denoise_fn
        self.y_only_model = y_only_model
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.cat_scheduler = cat_scheduler
        self.noise_dist = noise_dist
        self.edm_params = edm_params
        self.noise_dist_params = noise_dist_params
        self.sampler_params = sampler_params
        self.calibrate_losses = calibrate_losses

        self.w_num = 0.0
        self.w_cat = 0.0
        self.num_mask_idx = []
        self.cat_mask_idx = []

        if self.calibrate_losses:
            assert proportions is not None
            entropy = torch.tensor(
                [Categorical(probs=p).entropy() for p in proportions]
            ).float()
            self.register_buffer("normal_const", entropy)
        else:
            self.register_buffer("normal_const", torch.ones((len(num_classes),)))

        self.device = device

        if self.scheduler == "power_mean":
            self.num_schedule = PowerMeanNoise(**noise_schedule_params)
        elif self.scheduler == "power_mean_per_column":
            self.num_schedule = PowerMeanNoise_PerColumn(
                num_numerical=num_numerical_features, **noise_schedule_params
            )
        elif self.scheduler == "power_mean_unified":
            self.num_schedule = PowerMeanNoise_PerColumn(
                num_numerical=num_numerical_features, **noise_schedule_params
            )
        else:
            raise NotImplementedError(
                f"The noise schedule--{self.scheduler}-- is not implemented for contiuous data at CTIME "
            )

        if self.cat_scheduler == "log_linear":
            self.cat_schedule = LogLinearNoise(**noise_schedule_params)
        elif self.cat_scheduler == "log_linear_per_column":
            self.cat_schedule = LogLinearNoise_PerColumn(
                num_categories=len(num_classes), **noise_schedule_params
            )
        elif self.cat_scheduler == "log_linear_unified":
            self.cat_schedule = LogLinearNoise_PerColumn(
                num_categories=len(num_classes), **noise_schedule_params
            )
        else:
            raise NotImplementedError(
                f"The noise schedule--{self.cat_scheduler}-- is not implemented for discrete data at CTIME "
            )

    def mixed_loss(self, x):
        b = x.shape[0]
        device = x.device

        x_num = x[:, : self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features :].long()
        # Sample noise level
        if self.noise_dist == "uniform_t":
            t = torch.rand(b, device=device, dtype=x_num.dtype)
            t = t[:, None]
            sigma_num = self.num_schedule.total_noise(t)
            sigma_cat = self.cat_schedule.total_noise(t)
            dsigma_cat = self.cat_schedule.rate_noise(t)
        else:
            sigma_num = self.sample_ctime_noise(x)
            t = self.num_schedule.inverse_to_t(sigma_num)
            while torch.any((t < 0) + (t > 1)):
                # restrict t to [0,1]
                # this iterative approach is equivalent to sampling from a truncated version of the orignal noise distribution
                invalid_idx = ((t < 0) + (t > 1)).nonzero().squeeze(-1)
                sigma_num[invalid_idx] = self.sample_ctime_noise(x[: len(invalid_idx)])
                t = self.num_schedule.inverse_to_t(sigma_num)
            assert not torch.any((t < 0) + (t > 1))
            sigma_cat = self.cat_schedule.total_noise(t)
        # Convert sigma_cat to the corresponding alpha and move_chance
        # alpha = torch.exp(-sigma_cat)
        move_chance = -torch.expm1(
            -sigma_cat
        )  # torch.expm1 gives better numertical stability

        # Continuous forward diff
        x_num_t = x_num
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = x_num + noise * sigma_num

        # Discrete forward diff
        x_cat_t = x_cat
        if x_cat.shape[1] > 0:
            is_learnable = self.cat_scheduler == "log_linear_per_column"
            strategy = "soft" if is_learnable else "hard"
            x_cat_t, x_cat_t_soft = self.q_xt(x_cat, move_chance, strategy=strategy)

        # Predict orignal data (distribution)
        model_out_num, model_out_cat = self._denoise_fn(
            x_num_t, x_cat_t_soft, t.squeeze(), sigma=sigma_num
        )

        d_loss = torch.zeros((1,)).float()
        c_loss = torch.zeros((1,)).float()

        if x_num.shape[1] > 0:
            c_loss = self._edm_loss(model_out_num, x_num, sigma_num)
        if x_cat.shape[1] > 0:
            logits = self._subs_parameterization(
                model_out_cat, x_cat_t
            )  # log normalized probabilities, with the entry mask category being set to -inf
            d_loss = self._absorbed_closs(logits, x_cat, sigma_cat, dsigma_cat)

        return d_loss.mean(), c_loss.mean()

    @torch.no_grad()
    def sample(self, num_samples):
        b = num_samples
        device = self.device
        dtype = torch.float32

        # Create the chain of t
        t = torch.linspace(
            0, 1, self.num_timesteps, dtype=dtype, device=device
        )  # times = 0.0,...,1.0
        t = t[:, None]

        # Compute the chains of sigma
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[0:-1]

        # Prepare sigma_hat for stochastic sampling mode
        if self.sampler_params["stochastic_sampler"]:
            gamma = (
                min(S_churn / self.num_timesteps, np.sqrt(2) - 1)
                * (S_min <= sigma_num_cur)
                * (sigma_num_cur <= S_max)
            )
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat)
            t_hat = torch.min(
                t_hat, dim=-1, keepdim=True
            ).values  # take the samllest t_hat induced by sigma_num
            zero_gamma = (gamma == 0).any()
            t_hat[zero_gamma] = t[zero_gamma]
            out_of_bound = (t_hat > 1).squeeze()
            sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
            t_hat[out_of_bound] = t[out_of_bound]
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat = t
            sigma_num_hat = sigma_num_cur
            sigma_cat_hat = sigma_cat_cur

        # Sample priors for the continuous dimensions
        z_norm = (
            torch.randn((b, self.num_numerical_features), device=device)
            * sigma_num_cur[-1]
        )

        # Sample priors for the discrete dimensions
        has_cat = len(self.num_classes) > 0
        z_cat = torch.zeros(
            (b, 0), device=device
        ).float()  # the default values for categorical sample if the dataset has no categorical entry
        if has_cat:
            z_cat = self._sample_masked_prior(
                b,
                len(self.num_classes),
            )

        pbar = tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps)
        pbar.set_description("Sampling Progress")
        for i in pbar:
            z_norm, z_cat, q_xs = self.edm_update(
                z_norm,
                z_cat,
                i,
                t[i],
                t[i - 1] if i > 0 else None,
                t_hat[i],
                sigma_num_cur[i],
                sigma_num_next[i],
                sigma_num_hat[i],
                sigma_cat_cur[i],
                sigma_cat_next[i],
                sigma_cat_hat[i],
            )

        if not torch.all(
            z_cat < self.mask_index
        ):  # catch any update result in the mask class or the dummy classes
            error_index = torch.any(z_cat >= self.mask_index, dim=-1).nonzero()
            error_z_cat = z_cat[error_index]
            error_q_xs = q_xs[error_index]
            print(error_index)
            print(error_z_cat)
            print(error_q_xs)
        assert torch.all(z_cat < self.mask_index)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample

    def sample_all(self, num_samples, batch_size, keep_nan_samples=False):
        b = batch_size

        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            print(f"Samples left to generate: {num_samples - num_generated}")
            sample = self.sample(b)
            mask_nan = torch.any(sample.isnan(), dim=1)
            if keep_nan_samples:
                # If the sample instances that contains Nan are decided to be kept, the row with Nan will be foreced to all zeros
                sample = sample * (~mask_nan)[:, None]
            else:
                # Otherwise the instances with Nan will be eliminated
                sample = sample[~mask_nan]

            all_samples.append(sample)
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]

        return x_gen

    def q_xt(self, x, move_chance, strategy="hard"):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        if strategy == "hard":
            move_indices = torch.rand(*x.shape, device=x.device) < move_chance
            xt = torch.where(move_indices, self.mask_index.to(x.device), x)
            xt_soft = self.to_one_hot(xt).to(move_chance.dtype)
            return xt, xt_soft
        elif strategy == "soft":
            bs = x.shape[0]
            xt_soft = torch.zeros(bs, torch.sum(self.mask_index + 1), device=x.device)
            xt = torch.zeros_like(x)
            for i in range(len(self.num_classes)):
                slice_i = self.slices_for_classes_with_mask[i]
                # set the bernoulli probabilities, which determines the "coin flip" transition to the mask class
                prob_i = torch.zeros(bs, 2, device=x.device)
                prob_i[:, 0] = 1 - move_chance[:, i]
                prob_i[:, -1] = move_chance[:, i]
                log_prob_i = torch.log(prob_i)
                # draw soft samples and place them back to the corresponding columns
                soft_sample_i = F.gumbel_softmax(log_prob_i, tau=0.01, hard=True)
                idx = torch.stack(
                    (x[:, i] + slice_i[0], torch.ones_like(x[:, i]) * slice_i[-1]),
                    dim=-1,
                )
                xt_soft[torch.arange(len(idx)).unsqueeze(1), idx] = soft_sample_i
                # retrieve the hard samples
                xt[:, i] = torch.where(
                    soft_sample_i[:, 1] > soft_sample_i[:, 0],
                    self.mask_index[i],
                    x[:, i],
                )
            return xt, xt_soft

    def _subs_parameterization(self, unormalized_prob, xt):
        # log prob at the mask index = - infinity
        unormalized_prob = self.pad(unormalized_prob, self.neg_infinity)

        unormalized_prob[
            :, range(unormalized_prob.shape[1]), self.mask_index.to(xt.device)
        ] += self.neg_infinity

        # Take log softmax on the unnormalized probabilities to the logits
        logits = unormalized_prob - torch.logsumexp(
            unormalized_prob, dim=-1, keepdim=True
        )
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_index.to(xt.device)  # (bs, K)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def pad(self, x, pad_value):
        """
        Converts a concatenated tensor of class probabilities into a padded matrix,
        where each sub-tensor is padded along the last dimension to match the largest
        category size (max number of classes).

        Args:
            x (Tensor): The input tensor containing concatenated probabilities for all the categories in x_cat.
                        [bs, sum(num_classes_w_mask)]
            pad_value (float): The value filled into the dummy entries, which are padded to ensure all sub-tensors have equal size
                            along the last dimension.

        Returns:
            Tensor: A new tensorwith
                    [bs, len(num_classes_w_mask), max(num_classes_w_mask)), num_categories]
        """
        splited = torch.split(x, self.num_classes_w_mask, dim=-1)
        max_K = max(self.num_classes_w_mask)
        padded_ = [
            torch.cat(
                (
                    t,
                    pad_value
                    * torch.ones(
                        *(t.shape[:-1]),
                        max_K - t.shape[-1],
                        dtype=t.dtype,
                        device=t.device,
                    ),
                ),
                dim=-1,
            )
            for t in splited
        ]
        out = torch.stack(padded_, dim=-2)
        return out

    def to_one_hot(self, x_cat):
        x_cat_oh = torch.cat(
            [
                F.one_hot(
                    x_cat[:, i],
                    num_classes=self.num_classes[i] + 1,
                )
                for i in range(len(self.num_classes))
            ],
            dim=-1,
        )
        return x_cat_oh

    def _absorbed_closs(self, model_output, x0, sigma, dsigma):
        """
        alpha: (bs,)
        """
        log_p_theta = torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)
        alpha = torch.exp(-sigma)
        if self.cat_scheduler in ["log_linear_unified", "log_linear_per_column"]:
            elbo_weight = -dsigma / torch.expm1(sigma)
        else:
            elbo_weight = -1 / (1 - alpha)
        # normalize log_p_theta
        log_p_theta /= self.normal_const
        loss = elbo_weight * log_p_theta
        return loss

    def _sample_masked_prior(self, *batch_dims):
        return self.mask_index[None, :] * torch.ones(
            *batch_dims, dtype=torch.int64, device=self.mask_index.device
        )

    def _mdlm_update(self, log_p_x0, x, alpha_t, alpha_s):
        """
        # t: (bs,)
        log_p_x0: (bs, K, K_max)
        # alpha_t: (bs,)
        # alpha_s: (bs,)
        alpha_t: (bs, 1/K_cat)
        alpha_s: (bs,1/K_cat)
        """
        move_chance_t = 1 - alpha_t
        move_chance_s = 1 - alpha_s
        move_chance_t = move_chance_t.unsqueeze(-1)
        move_chance_s = move_chance_s.unsqueeze(-1)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        # There is a noremalizing term is (1-\alpha_t) who's responsility is to ensure q_xs is normalized.
        # However, omiting it won't make a difference for the Gumbel-max sampling trick in  _sample_categorical()
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, range(q_xs.shape[1]), self.mask_index] = move_chance_s[:, :, 0]

        # Important: make sure that prob of dummy classes are exactly 0
        dummy_mask = torch.tensor(
            [
                [(1 if i <= mask_idx else 0) for i in range(max(self.mask_index + 1))]
                for mask_idx in self.mask_index
            ],
            device=q_xs.device,
        )
        dummy_mask = torch.ones_like(q_xs) * dummy_mask
        q_xs *= dummy_mask

        _x = self._sample_categorical(q_xs)

        copy_flag = (x != self.mask_index.to(x.device)).to(x.dtype)

        z_cat = copy_flag * x + (1 - copy_flag) * _x
        if not torch.all(
            z_cat <= self.mask_index.to(x.device)
        ):  # catch any update result in the dummy classes
            error_index = torch.any(
                z_cat > self.mask_index.to(x.device), dim=-1
            ).nonzero()
            error_z_cat = z_cat[error_index]
            error_q_xs = q_xs[error_index]
            print(error_index)
            print(error_z_cat)
            print(error_q_xs)
        return copy_flag * x + (1 - copy_flag) * _x, q_xs

    def _sample_categorical(self, categorical_probs):
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    def sample_ctime_noise(self, batch):
        if self.noise_dist == "log_norm":
            rnd_normal = torch.randn(batch.shape[0], device=batch.device)
            sigma = (
                rnd_normal * self.noise_dist_params["P_std"]
                + self.noise_dist_params["P_mean"]
            ).exp()
        else:
            raise NotImplementedError(
                f"The noise distribution--{self.noise_dist}-- is not implemented for CTIME "
            )
        return sigma

    def _edm_loss(self, D_yn, y, sigma):
        weight = (sigma**2 + self.edm_params["sigma_data"] ** 2) / (
            sigma * self.edm_params["sigma_data"]
        ) ** 2

        target = y
        loss = weight * ((D_yn - target) ** 2)

        return loss

    def edm_update(
        self,
        x_num_cur,
        x_cat_cur,
        i,
        t_cur,
        t_next,
        t_hat,
        sigma_num_cur,
        sigma_num_next,
        sigma_num_hat,
        sigma_cat_cur,
        sigma_cat_next,
        sigma_cat_hat,
    ):
        """
        i = T-1,...,0
        """
        cfg = self.y_only_model is not None

        b = x_num_cur.shape[0]
        has_cat = len(self.num_classes) > 0

        # Get x_num_hat by move towards the noise by a small step
        x_num_hat = x_num_cur + (
            sigma_num_hat**2 - sigma_num_cur**2
        ).sqrt() * S_noise * torch.randn_like(x_num_cur)
        # Get x_cat_hat
        move_chance = -torch.expm1(
            sigma_cat_cur - sigma_cat_hat
        )  # the incremental move change is 1 - alpha_t/alpha_s = 1 - exp(sigma_s - sigma_t)
        x_cat_hat, _ = (
            self.q_xt(x_cat_cur, move_chance)
            if has_cat
            else torch.zeros_like(x_cat_cur)
        )

        # Get predictions
        x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_hat.dtype)
        denoised, raw_logits = self._denoise_fn(
            x_num_hat.float(),
            x_cat_hat_oh,
            t_hat.squeeze().repeat(b),
            sigma=sigma_num_hat.unsqueeze(0).repeat(b, 1),  # sigma accepts (bs, K_num)
        )

        # Apply cfg updates, if is in cfg mode
        is_bin_class = len(self.num_mask_idx) == 0
        is_learnable = self.scheduler == "power_mean_per_column"
        if cfg:
            if not is_learnable:
                sigma_cond = sigma_num_hat
            else:
                if is_bin_class:
                    sigma_cond = (
                        0.002 ** (1 / 7) + t_hat * (80 ** (1 / 7) - 0.002 ** (1 / 7))
                    ).pow(7)
                else:
                    sigma_cond = sigma_num_hat[self.num_mask_idx]
            y_num_hat = x_num_hat.float()[:, self.num_mask_idx]
            idx = list(
                chain(
                    *[self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]
                )
            )
            y_cat_hat = x_cat_hat_oh[:, idx]
            y_only_denoised, y_only_raw_logits = self.y_only_model(
                y_num_hat,
                y_cat_hat,
                t_hat.squeeze().repeat(b),
                sigma=sigma_cond.unsqueeze(0).repeat(b, 1),  # sigma accepts (bs, K_num)
            )

            denoised[:, self.num_mask_idx] *= 1 + self.w_num
            denoised[:, self.num_mask_idx] -= self.w_num * y_only_denoised

            mask_logit_idx = [
                self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx
            ]
            mask_logit_idx = (
                np.concatenate(mask_logit_idx)
                if len(mask_logit_idx) > 0
                else np.array([])
            )

            raw_logits[:, mask_logit_idx] *= 1 + self.w_cat
            raw_logits[:, mask_logit_idx] -= self.w_cat * y_only_raw_logits

        # Euler step
        d_cur = (x_num_hat - denoised) / sigma_num_hat
        x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * d_cur

        # Unmasking
        x_cat_next = x_cat_cur
        q_xs = torch.zeros_like(x_cat_cur).float()
        if has_cat:
            logits = self._subs_parameterization(raw_logits, x_cat_hat)
            alpha_t = torch.exp(-sigma_cat_hat).unsqueeze(0).repeat(b, 1)
            alpha_s = torch.exp(-sigma_cat_next).unsqueeze(0).repeat(b, 1)
            x_cat_next, q_xs = self._mdlm_update(logits, x_cat_hat, alpha_t, alpha_s)

        # Apply 2nd order correction.
        if self.sampler_params["second_order_correction"]:
            if i > 0:
                x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_next.dtype)
                denoised, raw_logits = self._denoise_fn(
                    x_num_next.float(),
                    x_cat_hat_oh,
                    t_next.squeeze().repeat(b),
                    sigma=sigma_num_next.unsqueeze(0).repeat(b, 1),
                )
                if cfg:
                    if not is_learnable:
                        sigma_cond = sigma_num_next
                    else:
                        if is_bin_class:
                            sigma_cond = (
                                0.002 ** (1 / 7)
                                + t_next * (80 ** (1 / 7) - 0.002 ** (1 / 7))
                            ).pow(7)
                        else:
                            sigma_cond = sigma_num_next[self.num_mask_idx]
                    y_num_next = x_num_next.float()[:, self.num_mask_idx]
                    idx = list(
                        chain(
                            *[
                                self.slices_for_classes_with_mask[i]
                                for i in self.cat_mask_idx
                            ]
                        )
                    )
                    y_cat_hat = x_cat_hat_oh[:, idx]
                    y_only_denoised, y_only_raw_logits = self.y_only_model(
                        y_num_next,
                        y_cat_hat,
                        t_next.squeeze().repeat(b),
                        sigma=sigma_cond.unsqueeze(0).repeat(
                            b, 1
                        ),  # sigma accepts (bs, K_num)
                    )
                    denoised[:, self.num_mask_idx] *= 1 + self.w_num
                    denoised[:, self.num_mask_idx] -= self.w_num * y_only_denoised

                d_prime = (x_num_next - denoised) / sigma_num_next
                x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

        return x_num_next, x_cat_next, q_xs

    def sample_impute(
        self,
        x_num,
        x_cat,
        num_mask_idx,
        cat_mask_idx,
        resample_rounds,
        impute_condition,
        w_num,
        w_cat,
    ):
        self.w_num = w_num
        self.w_cat = w_cat
        self.num_mask_idx = num_mask_idx
        self.cat_mask_idx = cat_mask_idx

        b = x_num.size(0)
        device = self.device
        dtype = torch.float32

        # Create masks, true for the missing columns
        num_mask = [i in num_mask_idx for i in range(self.num_numerical_features)]
        cat_mask = [i in cat_mask_idx for i in range(len(self.num_classes))]
        num_mask = torch.tensor(num_mask).to(x_num.device).to(x_num.dtype)
        cat_mask = torch.tensor(cat_mask).to(x_cat.device).to(x_cat.dtype)

        # Create the chain of t
        t = torch.linspace(
            0, 1, self.num_timesteps, dtype=dtype, device=device
        )  # times = 0.0,...,1.0
        t = t[:, None]

        # Compute the chains of sigma
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[0:-1]

        # Prepare sigma_hat for stochastic sampling mode
        if self.sampler_params["stochastic_sampler"]:
            gamma = (
                min(S_churn / self.num_timesteps, np.sqrt(2) - 1)
                * (S_min <= sigma_num_cur)
                * (sigma_num_cur <= S_max)
            )
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat)
            t_hat = torch.min(
                t_hat, dim=-1, keepdim=True
            ).values  # take the samllest t_hat induced by sigma_num
            zero_gamma = (gamma == 0).any()
            t_hat[zero_gamma] = t[zero_gamma]
            out_of_bound = (t_hat > 1).squeeze()
            sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
            t_hat[out_of_bound] = t[out_of_bound]
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat = t
            sigma_num_hat = sigma_num_cur
            sigma_cat_hat = sigma_cat_cur

        # Sample priors for the continuous dimensions
        if impute_condition == "x_t":
            z_norm = (
                x_num
                + torch.randn((b, self.num_numerical_features), device=device)
                * sigma_num_cur[-1]
            )  # z_{t_max} = x_0(masked) + sigma_max*epsilon
        elif impute_condition == "x_0":
            z_norm = x_num

        # Sample priors for the discrete dimensions
        has_cat = len(self.num_classes) > 0
        z_cat = torch.zeros(
            (b, 0), device=device
        ).float()  # the default values for categorical sample if the dataset has no categorical entry
        if has_cat:
            if impute_condition == "x_t":
                z_cat = self._sample_masked_prior(
                    b,
                    len(self.num_classes),
                )  # z_{t_max} is still all pushed to [MASK]
            elif impute_condition == "x_0":
                z_cat = x_cat

        pbar = tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps)
        pbar.set_description("Sampling Progress")
        for i in pbar:
            for u in range(resample_rounds):
                # Get known parts by Forward Flow
                if impute_condition == "x_t":
                    z_norm_known = (
                        x_num
                        + torch.randn((b, self.num_numerical_features), device=device)
                        * sigma_num_next[i]
                    )
                    move_chance = (
                        1 - torch.exp(-sigma_cat_next[i])
                        if i < (self.num_timesteps - 1)
                        else torch.ones_like(sigma_cat_next[i])
                    )  # force move_chance to be 1 for the first iteration
                    z_cat_known, _ = self.q_xt(x_cat, move_chance)
                elif impute_condition == "x_0":
                    z_norm_known = x_num
                    z_cat_known = x_cat

                # Get unknown by Reverse Step
                z_norm_unknown, z_cat_unknown, q_xs = self.edm_update(
                    z_norm,
                    z_cat,
                    i,
                    t[i],
                    t[i - 1] if i > 0 else None,
                    t_hat[i],
                    sigma_num_cur[i],
                    sigma_num_next[i],
                    sigma_num_hat[i],
                    sigma_cat_cur[i],
                    sigma_cat_next[i],
                    sigma_cat_hat[i],
                )
                z_norm = (1 - num_mask) * z_norm_known + num_mask * z_norm_unknown
                z_cat = (1 - cat_mask) * z_cat_known + cat_mask * z_cat_unknown

                # Resample x_t from x_{t-1} by Foward Step
                if u < resample_rounds - 1:
                    z_norm = z_norm + (
                        sigma_num_cur[i] ** 2 - sigma_num_next[i] ** 2
                    ).sqrt() * S_noise * torch.randn_like(z_norm)
                    move_chance = -torch.expm1(sigma_cat_next[i] - sigma_cat_cur[i])
                    z_cat, _ = self.q_xt(z_cat, move_chance)

        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample


class MultiTableUnifiedCtimeDiffusion(UnifiedCtimeDiffusion):
    def __init__(
        self,
        num_classes: dict[np.array],
        num_numerical_features: dict[int],
        denoise_fn: torch.nn.Module,
        num_timesteps=1000,
        scheduler="power_mean",
        cat_scheduler="log_linear",
        noise_dist="uniform",
        edm_params={},
        noise_dist_params={},
        noise_schedule_params={},
        sampler_params={},
        device=torch.device("cpu"),
        root_table: str | None = None,
        n_hops_dataloader: int = 2,
        calibrate_losses: bool = False,
        proportions_dict: dict = {},
        dequantize: bool = False,
        is_disjoint: bool = False,
        num_seed_nodes: int = 10,
        num_neighbors: int = -1,
        dimension_tables: list | None = None,
        timestep_sampling: str = "uniform",
        **kwargs,
    ):
        super(UnifiedCtimeDiffusion, self).__init__()

        self.active_table = None
        self.num_numerical_features_dict = num_numerical_features
        self.num_classes_dict = dict()
        self.num_classes_expanded_dict = dict()
        self.mask_index_dict = dict()
        self.neg_infinity = -1000000.0
        self.num_classes_w_mask_dict = dict()
        self.slices_for_classes_dict = dict()
        self.offsets_dict = dict()
        self.slices_for_classes_with_mask_dict = dict()
        self.root_table = root_table
        self.n_hops_dataloader = n_hops_dataloader
        self.is_disjoint = is_disjoint
        self.num_seed_nodes = num_seed_nodes
        self.num_neighbors = num_neighbors
        self.dimension_tables = dimension_tables
        self.timestep_sampling = timestep_sampling

        self._denoise_fn = denoise_fn
        # self.y_only_model = y_only_model
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.cat_scheduler = cat_scheduler
        self.noise_dist = noise_dist
        self.edm_params = edm_params
        self.noise_dist_params = noise_dist_params
        self.sampler_params = sampler_params
        self.calibrate_losses = calibrate_losses
        self.dequantize = dequantize

        self.device = device

        self.num_schedule_dict = nn.ModuleDict()
        self.cat_schedule_dict = nn.ModuleDict()
        self.normal_const_dict = nn.ParameterDict()
        if self.dequantize:
            dataset: HeteroData = kwargs.get("dataset", None)
            assert dataset is not None
            self.dequantizers = dict()

        for table_name, num_classes in num_classes.items():
            num_classes = np.array(num_classes)  # it as a vector [K1, K2, ..., Km]
            self.num_classes_dict[table_name] = num_classes
            self.num_classes_expanded_dict[table_name] = (
                torch.from_numpy(
                    np.concatenate(
                        [
                            num_classes[i].repeat(num_classes[i])
                            for i in range(len(num_classes))
                        ]
                    )
                ).to(device)
                if len(num_classes) > 0
                else torch.tensor([]).to(device).int()
            )
            self.mask_index_dict[table_name] = (
                torch.tensor(num_classes).long().to(device)
            )
            self.num_classes_w_mask_dict[table_name] = tuple(
                [c.item() for c in num_classes + 1]
            )

            offsets = np.cumsum(num_classes)
            offsets = np.append([0], offsets)
            slices_for_classes = []
            for i in range(1, len(offsets)):
                slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
            self.offsets_dict[table_name] = torch.from_numpy(offsets).to(device)

            offsets = np.cumsum(num_classes) + np.arange(1, len(num_classes) + 1)
            offsets = np.append([0], offsets)
            slices_for_classes_with_mask = []
            for i in range(1, len(offsets)):
                slices_for_classes_with_mask.append(
                    np.arange(offsets[i - 1], offsets[i])
                )
            self.slices_for_classes_with_mask_dict[table_name] = (
                slices_for_classes_with_mask
            )

            if self.calibrate_losses:
                proportions = proportions_dict.get(table_name, None)
                assert proportions is not None
                normal_const = torch.tensor(
                    [Categorical(probs=p).entropy() for p in proportions]
                )
            else:
                normal_const = torch.ones((len(num_classes),))
            self.normal_const_dict[table_name] = nn.Parameter(
                normal_const, requires_grad=False
            )

            # Set up noise schedules
            if (
                self.dimension_tables is not None
                and table_name in self.dimension_tables
            ):
                if "per_column" in self.scheduler:
                    dim_num = num_numerical_features[table_name]
                    dim_cat = len(num_classes)
                else:
                    dim_num = 1
                    dim_cat = 1
                self.num_schedule_dict[table_name] = NoNoise(dim=dim_num)
                self.cat_schedule_dict[table_name] = NoNoise(dim=dim_cat)
                continue

            if self.scheduler == "power_mean":
                self.num_schedule_dict[table_name] = PowerMeanNoise(
                    **noise_schedule_params
                )
            elif self.scheduler == "power_mean_per_column":
                self.num_schedule_dict[table_name] = PowerMeanNoise_PerColumn(
                    num_numerical=num_numerical_features[table_name],
                    **noise_schedule_params,
                )
            elif self.scheduler == "power_mean_unified":
                self.num_schedule_dict[table_name] = PowerMeanUnified(
                    num_numerical=num_numerical_features[table_name],
                    **noise_schedule_params,
                )
            else:
                raise NotImplementedError(
                    f"The noise schedule--{self.scheduler}-- is not implemented for contiuous data at CTIME "
                )

            # Override the scheduler with the default if there are no numerical features.
            # This way we have a placeholder sigma_num for stochastic sampling.
            if (
                num_numerical_features[table_name] == 0
                and sampler_params["stochastic_sampler"]
            ):
                self.num_schedule_dict[table_name] = PowerMeanNoise(
                    **noise_schedule_params
                )

            if self.cat_scheduler == "log_linear":
                self.cat_schedule_dict[table_name] = LogLinearNoise(
                    **noise_schedule_params
                )
            elif self.cat_scheduler == "log_linear_per_column":
                self.cat_schedule_dict[table_name] = LogLinearNoise_PerColumn(
                    num_categories=len(num_classes), **noise_schedule_params
                )
            elif self.cat_scheduler == "log_linear_unified":
                self.cat_schedule_dict[table_name] = LogLinearUnified(
                    num_categories=len(num_classes), **noise_schedule_params
                )
            else:
                raise NotImplementedError(
                    f"The noise schedule--{self.cat_scheduler}-- is not implemented for discrete data at CTIME "
                )

            if self.dequantize:
                # Dimenstion tables are skipped as stochastic transforms don't make sense there. (continue above)
                # Tables with no numerical features are skipped as well.
                if num_numerical_features[table_name] == 0:
                    continue
                self.dequantizers[table_name] = TensorDequantizer(
                    scale=self.edm_params["sigma_data"]
                ).fit(dataset[table_name].x_num)
                self.dequantizers[table_name].to(self.device)

    @contextmanager
    def activate_table(self, table):
        yield table
        self.active_table = None

    @property
    def mask_index(self):
        return self.mask_index_dict[self.active_table]

    @property
    def num_classes(self):
        return self.num_classes_dict[self.active_table]

    @property
    def slices_for_classes_with_mask(self):
        return self.slices_for_classes_with_mask_dict[self.active_table]

    @property
    def num_classes_w_mask(self):
        return self.num_classes_w_mask_dict[self.active_table]

    @property
    def normal_const(self):
        return self.normal_const_dict[self.active_table]

    def mixed_loss(self, batch: HeteroData, t_batch: torch.Tensor | None = None):
        x_num_dict = batch.x_num_dict
        x_cat_dict = batch.x_cat_dict
        time_dict = dict()
        sigma_num_dict = dict()
        sigma_cat_dict = dict()
        dsigma_cat_dict = dict()
        if self.is_disjoint:
            num_subgraphs = max(
                {nodes[0] for _, nodes in batch.num_sampled_nodes_dict.items()}
            )
        else:
            num_subgraphs = 1
        if t_batch is None:
            if self.timestep_sampling == "uniform":
                t_batch = torch.rand(
                    num_subgraphs, device=self.device, dtype=torch.float32
                )
            elif self.timestep_sampling == "low_discrepancy":
                t_batch = low_discrepancy_sampler(num_subgraphs, device=self.device)
            elif self.timestep_sampling == "antithetic":
                t_batch = antithetic_sampler(num_subgraphs, device=self.device)

        else:
            t_batch = t_batch.unsqueeze(-1).to(self.device)
        x_num_t_dict = dict()
        x_cat_t_dict = dict()
        x_cat_t_soft_dict = dict()
        for table_name in x_num_dict.keys():
            if (
                self.dequantize
                and table_name in self.dequantizers
                and x_num_dict[table_name].size(1) > 0
            ):
                # Dequantize the numerical features
                x_num_dict[table_name] = self.dequantizers[table_name].transform(
                    x_num_dict[table_name]
                )
            with self.activate_table(table_name):
                x_num = x_num_dict[table_name]
                x_cat = x_cat_dict[table_name].long()
                # Sample noise level
                if self.noise_dist == "uniform_t":
                    if self.is_disjoint:
                        subgraph_ids = batch.batch_dict[table_name]
                    else:
                        subgraph_ids = torch.zeros(
                            x_num.shape[0], device=self.device
                        ).long()
                    t = t_batch[subgraph_ids]
                    t = t[:, None]
                    sigma_num = self.num_schedule_dict[table_name].total_noise(t)
                    sigma_cat = self.cat_schedule_dict[table_name].total_noise(t)
                    dsigma_cat = self.cat_schedule_dict[table_name].rate_noise(t)
                else:
                    raise NotImplementedError(
                        f"The noise distribution--{self.noise_dist}-- is not implemented for CTIME "
                    )
                # Convert sigma_cat to the corresponding alpha and move_chance
                # alpha = torch.exp(-sigma_cat)
                move_chance = -torch.expm1(
                    -sigma_cat
                )  # torch.expm1 gives better numertical stability

                # Continuous forward diff
                x_num_t = x_num
                if x_num.shape[1] > 0:
                    noise = torch.randn_like(x_num)
                    x_num_t = x_num + noise * sigma_num

                # Discrete forward diff
                x_cat_t = x_cat
                if x_cat.shape[1] > 0:
                    is_learnable = self.cat_scheduler == "log_linear_per_column"
                    strategy = "soft" if is_learnable else "hard"

                    with self.activate_table(table_name) as self.active_table:
                        x_cat_t, x_cat_t_soft = self.q_xt(
                            x_cat, move_chance, strategy=strategy
                        )
                else:
                    x_cat_t_soft = torch.zeros_like(x_cat).float()
                x_num_t_dict[table_name] = x_num_t
                x_cat_t_dict[table_name] = x_cat_t
                x_cat_t_soft_dict[table_name] = x_cat_t_soft
                # it is possible for only one node of a given type is sampled
                time_dict[table_name] = t.squeeze(-1)
                sigma_num_dict[table_name] = sigma_num
                sigma_cat_dict[table_name] = sigma_cat
                dsigma_cat_dict[table_name] = dsigma_cat

        # Predict orignal data (distribution)
        model_out_num_dict, model_out_cat_dict = self._denoise_fn(
            x_num_t_dict,
            x_cat_t_soft_dict,
            time_dict,
            sigma=sigma_num_dict,
            batch=batch,
        )

        d_loss_dict = dict()
        c_loss_dict = dict()

        c_loss = torch.tensor(0.0, device=self.device)
        d_loss = torch.tensor(0.0, device=self.device)
        for table_name in x_num_dict.keys():
            mask = torch.isin(batch[table_name].n_id, batch[table_name].input_id)
            c_loss_dict[table_name] = torch.tensor(0.0, device=self.device)
            d_loss_dict[table_name] = torch.tensor(0.0, device=self.device)
            if mask.sum() == 0 or table_name in self.dimension_tables:
                continue
            model_out_num = model_out_num_dict[table_name]
            model_out_cat = model_out_cat_dict[table_name]
            x_num = x_num_dict[table_name][mask]
            x_cat = x_cat_dict[table_name][mask]
            x_cat_t = x_cat_t_dict[table_name][mask]
            sigma_num = sigma_num_dict[table_name][mask]
            sigma_cat = sigma_cat_dict[table_name][mask]
            dsigma_cat = dsigma_cat_dict[table_name][mask]

            if x_num.shape[1] > 0:
                c_loss_table = self._edm_loss(model_out_num, x_num, sigma_num)
                c_loss_dict[table_name] += c_loss_table.mean()
            if x_cat.shape[1] > 0:
                with self.activate_table(table_name) as self.active_table:
                    logits = self._subs_parameterization(
                        model_out_cat, x_cat_t
                    )  # log normalized probabilities, with the entry mask category being set to -inf
                    d_loss_table = self._absorbed_closs(
                        logits, x_cat, sigma_cat, dsigma_cat
                    )
                d_loss_dict[table_name] += d_loss_table.mean()
            c_loss += c_loss_dict[table_name]
            d_loss += d_loss_dict[table_name]
        c_loss_dict["total"] = c_loss
        d_loss_dict["total"] = d_loss

        return d_loss_dict, c_loss_dict

    @torch.no_grad()
    def sample_all(self, dataset: HeteroData, device=None, batch_size: int = 20000):
        if device is None:
            device = self.device
        dtype = torch.float32

        z_norm_dict = dict()
        z_cat_dict = dict()
        sigma_num_cur_dict = dict()
        sigma_num_next_dict = dict()
        sigma_num_hat_dict = dict()
        sigma_cat_cur_dict = dict()
        sigma_cat_next_dict = dict()
        sigma_cat_hat_dict = dict()

        # Create the chain of t
        t = torch.linspace(
            0, 1, self.num_timesteps, dtype=dtype, device=device
        )  # times = 0.0,...,1.0
        t = t[:, None]

        for table_name in self.num_classes_dict.keys():
            b = dataset[table_name].num_nodes

            # Compute the chains of sigma
            sigma_num_cur = self.num_schedule_dict[table_name].total_noise(t)
            sigma_cat_cur = self.cat_schedule_dict[table_name].total_noise(t)
            sigma_num_next = torch.zeros_like(sigma_num_cur)
            sigma_num_next[1:] = sigma_num_cur[0:-1]
            sigma_cat_next = torch.zeros_like(sigma_cat_cur)
            sigma_cat_next[1:] = sigma_cat_cur[0:-1]

            if self.sampler_params["stochastic_sampler"]:
                # Prepare sigma_hat for stochastic sampling mode
                assert sigma_num_cur.shape[1] > 0
                gamma = (
                    min(S_churn / self.num_timesteps, np.sqrt(2) - 1)
                    * (S_min <= sigma_num_cur)
                    * (sigma_num_cur <= S_max)
                )
                sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
                if (
                    self.dimension_tables is not None
                    and table_name in self.dimension_tables
                ):
                    s_cur_ = PowerMeanNoise().total_noise(t)
                    s_hat_ = s_cur_ + gamma * s_cur_
                    t_hat = PowerMeanNoise().inverse_to_t(s_hat_)
                else:
                    t_hat = self.num_schedule_dict[table_name].inverse_to_t(
                        sigma_num_hat
                    )
                t_hat = torch.min(
                    t_hat, dim=-1, keepdim=True
                ).values  # take the samllest t_hat induced by sigma_num
                zero_gamma = (gamma == 0).any()
                t_hat[zero_gamma] = t[zero_gamma]
                out_of_bound = (t_hat > 1).squeeze()
                sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
                t_hat[out_of_bound] = t[out_of_bound]
                sigma_cat_hat = self.cat_schedule_dict[table_name].total_noise(t_hat)
            else:
                t_hat = t
                sigma_num_hat = sigma_num_cur
                sigma_cat_hat = sigma_cat_cur

            # Sample priors for the continuous dimensions
            with self.activate_table(table_name) as self.active_table:
                z_norm = (
                    torch.randn(
                        (b, self.num_numerical_features_dict[table_name]), device=device
                    )
                    * sigma_num_cur[-1]
                )

                # Sample priors for the discrete dimensions
                num_classes = self.num_classes_dict[table_name]
                has_cat = len(num_classes) > 0
                z_cat = torch.zeros(
                    (b, 0), device=device
                ).float()  # the default values for categorical sample if the dataset has no categorical entry
                if table_name in self.dimension_tables:
                    z_norm = dataset[table_name].x_num.to(device)
                    z_cat = dataset[table_name].x_cat.to(device)
                elif has_cat:
                    z_cat = self._sample_masked_prior(
                        b,
                        len(num_classes),
                    ).to(device)

            z_norm_dict[table_name] = z_norm
            z_cat_dict[table_name] = z_cat
            sigma_num_cur_dict[table_name] = sigma_num_cur
            sigma_num_next_dict[table_name] = sigma_num_next
            sigma_num_hat_dict[table_name] = sigma_num_hat
            sigma_cat_cur_dict[table_name] = sigma_cat_cur
            sigma_cat_next_dict[table_name] = sigma_cat_next
            sigma_cat_hat_dict[table_name] = sigma_cat_hat

        print("Starting denoising...")
        for i in reversed(range(0, self.num_timesteps)):
            z_norm_dict, z_cat_dict, q_xs_dict = self.edm_update(
                dataset,
                z_norm_dict,
                z_cat_dict,
                i,
                t[i],
                t[i - 1] if i > 0 else None,
                t_hat[i],
                sigma_num_cur_dict,
                sigma_num_next_dict,
                sigma_num_hat_dict,
                sigma_cat_cur_dict,
                sigma_cat_next_dict,
                sigma_cat_hat_dict,
                device=device,
                batch_size=batch_size,
            )

        sample_dict = dict()
        for table_name, q_xs in q_xs_dict.items():
            z_cat = z_cat_dict[table_name]
            z_norm = z_norm_dict[table_name]
            with self.activate_table(table_name) as self.active_table:
                if not torch.all(
                    z_cat < self.mask_index.to(z_cat.device)
                ):  # catch any update result in the mask class or the dummy classes
                    error_index = torch.any(
                        z_cat >= self.mask_index.to(z_cat.device), dim=-1
                    ).nonzero()
                    error_z_cat = z_cat[error_index]
                    error_q_xs = q_xs[error_index]
                    print(error_index)
                    print(error_z_cat)
                    print(error_q_xs)
                assert torch.all(z_cat < self.mask_index.to(z_cat.device))
            if (
                self.dequantize
                and table_name in self.dequantizers
                and z_norm.size(1) > 0
            ):
                z_norm = (
                    self.dequantizers[table_name]
                    .to(z_norm.device)
                    .inverse_transform(z_norm.cpu())
                )
            sample_dict[table_name] = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample_dict

    def edm_update(
        self,
        data: HeteroData,
        x_num_cur_dict,
        x_cat_cur_dict,
        i,
        t_cur,
        t_next,
        t_hat,
        sigma_num_cur_dict,  # [i]
        sigma_num_next_dict,  # [i]
        sigma_num_hat_dict,  # [i]
        sigma_cat_cur_dict,  # [i]
        sigma_cat_next_dict,  # [i]
        sigma_cat_hat_dict,  # [i]
        device=torch.device("cuda"),
        batch_size: int = 20000,
    ):
        """
        i = T-1,...,0
        """

        x_cat_hat_dict = dict()
        for table_name in x_num_cur_dict.keys():
            x_num_cur = x_num_cur_dict[table_name]
            x_cat_cur = x_cat_cur_dict[table_name]
            sigma_num_cur = sigma_num_cur_dict[table_name][i]
            sigma_num_next = sigma_num_next_dict[table_name][i]
            sigma_num_hat = sigma_num_hat_dict[table_name][i]
            sigma_cat_cur = sigma_cat_cur_dict[table_name][i]
            sigma_cat_next = sigma_cat_next_dict[table_name][i]
            sigma_cat_hat = sigma_cat_hat_dict[table_name][i]

            b = x_num_cur.shape[0]
            has_cat = len(self.num_classes_dict[table_name]) > 0

            # Get x_num_hat by move towards the noise by a small step
            x_num_hat = x_num_cur + (
                sigma_num_hat**2 - sigma_num_cur**2
            ).sqrt() * S_noise * torch.randn_like(x_num_cur)
            # Get x_cat_hat
            move_chance = -torch.expm1(
                sigma_cat_cur - sigma_cat_hat
            )  # the incremental move change is 1 - alpha_t/alpha_s = 1 - exp(sigma_s - sigma_t)
            if table_name in self.dimension_tables and has_cat:
                try:
                    x_cat_hat = data[table_name].x_cat
                    x_cat_hat_oh = data[table_name].x_cat_hat_oh
                except AttributeError:
                    with self.activate_table(table_name) as self.active_table:
                        x_cat_hat, _ = self.q_xt(x_cat_cur, move_chance)
                        x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_hat.dtype)
            elif has_cat:
                with self.activate_table(table_name) as self.active_table:
                    x_cat_hat, _ = self.q_xt(x_cat_cur, move_chance)

                    # Get predictions
                    x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_hat.dtype)
            else:
                x_cat_hat_oh = torch.zeros_like(x_cat_cur).float()
                x_cat_hat = torch.zeros_like(x_cat_cur).long()
            data[table_name].x_num_hat = x_num_hat.float()
            data[table_name].x_cat_hat_oh = x_cat_hat_oh
            data[table_name].t_hat = t_hat.squeeze().repeat(b)
            data[table_name].sigma = sigma_num_hat.unsqueeze(0).repeat(b, 1)
            x_cat_hat_dict[table_name] = x_cat_hat

        denoised_dict = {
            table_name: torch.zeros_like(x_num_hat).to(device)
            for table_name, x_num_hat in data.x_num_hat_dict.items()
        }
        raw_logits_dict = {
            table_name: torch.zeros_like(x_cat_hat_oh).to(device)
            for table_name, x_cat_hat_oh in data.x_cat_hat_oh_dict.items()
        }
        dataloader = self.get_dataloader(data, batch_size=batch_size)
        pbar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Denosing (step {i})",
        )
        for batch in pbar:
            batch = batch.to(self.device)
            denoised_batch, raw_logits_batch = self._denoise_fn(
                batch.x_num_hat_dict,
                batch.x_cat_hat_oh_dict,
                batch.t_hat_dict,
                sigma=batch.sigma_dict,
                batch=batch,
            )
            for table_name in data.node_types:
                if table_name in self.dimension_tables:
                    denoised_dict[table_name] = data[table_name].x_num_hat.to(device)
                    raw_logits_dict[table_name] = data[table_name].x_cat_hat_oh.to(
                        device
                    )
                    continue
                index = batch[table_name].input_id.to(device)
                denoised_dict[table_name][index] = denoised_batch[table_name].to(device)
                raw_logits_dict[table_name][index] = raw_logits_batch[table_name].to(
                    device
                )

        d_cur_dict = dict()
        q_xs_dict = dict()
        x_cat_next_dict = dict()
        for table_name in data.node_types:
            x_num_hat = data[table_name].x_num_hat.to(device)
            denoised = denoised_dict[table_name].to(device)
            sigma_num_hat = sigma_num_hat_dict[table_name][i]
            sigma_num_next = sigma_num_next_dict[table_name][i]
            x_cat_cur = x_cat_cur_dict[table_name]
            raw_logits = raw_logits_dict[table_name]
            x_cat_hat = x_cat_hat_dict[table_name]
            sigma_cat_hat = sigma_cat_hat_dict[table_name][i]
            sigma_cat_next = sigma_cat_next_dict[table_name][i]

            b = x_num_hat.shape[0]
            if table_name in self.dimension_tables:
                d_cur = None
                x_num_next = x_num_hat
            else:
                # Euler step
                d_cur = (x_num_hat - denoised.to(torch.float64)) / sigma_num_hat.to(
                    torch.float64
                )
                x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * d_cur

            # Unmasking
            x_cat_next = x_cat_cur
            q_xs = torch.zeros_like(x_cat_cur).float()
            has_cat = len(self.num_classes_dict[table_name]) > 0
            if table_name in self.dimension_tables and has_cat:
                x_cat_hat_oh = data[table_name].x_cat_hat_oh.to(device)
            elif has_cat:
                with self.activate_table(table_name) as self.active_table:
                    logits = self._subs_parameterization(
                        raw_logits.to(device), x_cat_hat
                    )
                    alpha_t = torch.exp(-sigma_cat_hat).unsqueeze(0).repeat(b, 1)
                    alpha_s = torch.exp(-sigma_cat_next).unsqueeze(0).repeat(b, 1)
                    x_cat_next, q_xs = self._mdlm_update(
                        logits, x_cat_hat, alpha_t, alpha_s
                    )

                    x_cat_hat_oh = self.to_one_hot(x_cat_hat).float()
                data[table_name].x_cat_hat_oh = x_cat_hat_oh
            d_cur_dict[table_name] = d_cur
            data[table_name].x_num_next = x_num_next.float()
            if i > 0:
                data[table_name].t_next = t_next.squeeze().repeat(b)
            data[table_name].sigma = sigma_num_next.unsqueeze(0).repeat(b, 1)
            q_xs_dict[table_name] = q_xs
            x_cat_next_dict[table_name] = x_cat_next

        x_num_next_dict = {
            table_name: torch.zeros_like(x_num_next)
            for table_name, x_num_next in data.x_num_next_dict.items()
        }
        # Apply 2nd order correction.
        if self.sampler_params["second_order_correction"]:
            if i > 0:
                dataloader = self.get_dataloader(data, batch_size=batch_size)
                pbar = tqdm(
                    dataloader,
                    total=len(dataloader),
                    desc=f"2nd order correction (step {i})",
                )
                for batch in pbar:
                    batch = batch.to(self.device)
                    denoised_batch_dict, _ = self._denoise_fn(
                        batch.x_num_next_dict,
                        batch.x_cat_hat_oh_dict,
                        batch.t_next_dict,
                        sigma=batch.sigma_dict,
                        batch=batch,
                    )

                    for table_name, denoised_batch in denoised_batch_dict.items():
                        if table_name in self.dimension_tables:
                            x_num_next_dict[table_name] = data[table_name].x_num.to(
                                device
                            )
                            continue
                        index = batch[table_name].input_id.to(device)

                        x_num_next = (
                            data[table_name]
                            .x_num_next.to(device)[index]
                            .to(torch.float64)
                        )
                        x_num_hat = (
                            data[table_name]
                            .x_num_hat.to(device)[index]
                            .to(torch.float64)
                        )
                        sigma_num_next = sigma_num_next_dict[table_name][i].to(
                            torch.float64
                        )
                        sigma_num_hat = sigma_num_hat_dict[table_name][i].to(
                            torch.float64
                        )
                        d_cur = d_cur_dict[table_name][index].to(torch.float64)
                        d_prime = (
                            x_num_next - denoised_batch.to(device)
                        ) / sigma_num_next
                        x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * (
                            0.5 * d_cur + 0.5 * d_prime
                        )
                        x_num_next_dict[table_name][index] = x_num_next.float()
            else:
                x_num_next_dict = data.x_num_next_dict
        return x_num_next_dict, x_cat_next_dict, q_xs_dict

    def get_dataloader(
        self,
        data: HeteroData,
        batch_size: int = 20000,
    ) -> NeighborLoader:
        if self.num_neighbors > -1:
            shuffle = True  # Use different neigborhood  each step
        else:
            shuffle = False
        return get_subgraph_dataloader(
            data,
            root_table=self.root_table,
            batch_size=batch_size,
            shuffle=shuffle,
            n_hops=self.n_hops_dataloader,
            num_neighbors=self.num_neighbors,
            num_workers=0,  # Increase this for faster dataloading
            is_disjoint=self.is_disjoint,
            num_seed_nodes=self.num_seed_nodes,
            dimension_tables=self.dimension_tables,
            drop_last=False,
            two_stage=False,
        )


def antithetic_sampler(num_samples, device, sampling_eps=1e-3):
    """
    Based on Simple and Effective Masked Diffusion Language Models (Sahoo et al., 2024)
    """
    _eps_t = torch.rand(num_samples, device=device)
    offset = torch.arange(num_samples, device=device) / num_samples
    _eps_t = (_eps_t / num_samples + offset) % 1
    t = (1 - sampling_eps) * _eps_t + sampling_eps
    return t


def low_discrepancy_sampler(num_samples, device):
    """
    Inspired from the Variational Diffusion Paper (Kingma et al., 2022)
    Based on the implementation (https://github.com/muellermarkus/cdtd)
    """
    single_u = torch.rand((1,), device=device, requires_grad=False, dtype=torch.float32)
    return (
        single_u
        + torch.arange(
            0.0, 1.0, step=1.0 / num_samples, device=device, requires_grad=False
        )
    ) % 1
