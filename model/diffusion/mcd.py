# model/diffusion/mcd.py
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Diffusion-TS'))

from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
from Models.interpretable_diffusion.model_utils import default, extract


# ── 1. 可逆实例归一化（RevIN）────────────────────────────────
class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    参考 Non-stationary Transformer (NeurIPS 2022)
    对每个样本窗口单独归一化，生成后反归一化回原始尺度
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # 可学习的仿射参数（每个特征维度一个）
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        归一化并保存均值/方差供后续denorm使用
        """
        # 沿时间维度计算统计量，保持dim不变
        self._mean = x.mean(dim=1, keepdim=True).detach()        # (B, 1, F)
        self._std  = x.std(dim=1, keepdim=True, unbiased=False).detach() + self.eps  # (B, 1, F)

        x = (x - self._mean) / self._std
        x = x * self.affine_weight + self.affine_bias
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成后调用，把归一化空间的输出还原到原始尺度
        """
        x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self._std + self._mean
        return x


# ── 2. MCD主模型：继承Diffusion_TS，加regime条件 + RevIN ─────
class MarketConditionedDiffusion(Diffusion_TS):
    """
    Market-Conditioned Diffusion (MCD)

    在Diffusion_TS基础上增加：
      1. regime embedding：nn.Embedding(3, d_model)
         -> 生成label_emb传给Transformer内部的AdaLayerNorm
         -> 训练和采样时都可以指定regime_id
      2. RevIN：输入归一化 + 生成后反归一化
         -> 解决金融时序非平稳性问题

    参数
    ----
    n_regimes : int
        市场状态数量，默认3（牛/熊/震荡）
    d_model : int
        Transformer的embedding维度，regime embedding维度与之一致
    其余参数与Diffusion_TS完全一致
    """

    def __init__(
        self,
        seq_length: int,
        feature_size: int,
        n_regimes: int = 3,
        d_model: int = 64,
        **kwargs
    ):
        # 把d_model传给父类（父类参数名是n_embd）
        super().__init__(
            seq_length=seq_length,
            feature_size=feature_size,
            d_model=d_model,
            **kwargs
        )
        # 在__init__里super().__init__()之后加这一行
        print("actual n_embd:", self.model.encoder.blocks[0].ln1.linear.in_features)

        self.n_regimes = n_regimes
        self.d_model   = d_model

        # ── regime embedding，维度和Transformer内部d_model对齐
        actual_embd = self.model.encoder.blocks[0].ln1.linear.in_features
        self.regime_emb = nn.Embedding(n_regimes, actual_embd)  # 改这行

        # ── RevIN，特征维度 = feature_size
        self.revin = RevIN(num_features=feature_size)

    # ── 内部：带条件的output ──────────────────────────────────
    def output(self, x, t, regime_id=None, padding_masks=None):
        """
        覆盖父类output，加入regime条件。
        x         : (B, T, F)  noisy时序
        t         : (B,)       diffusion timestep
        regime_id : (B,)       市场状态标签，0/1/2；None时退化为无条件
        """
        # 生成label_emb：(B, 1, d_model)，AdaLayerNorm内部会broadcast
        label_emb = None
        if regime_id is not None:
            label_emb = self.regime_emb(regime_id)  # (B, 1, d_model)

        trend, season = self.model(x, t, padding_masks=padding_masks, label_emb=label_emb)
        return trend + season

    # ── 训练loss：接受regime_id ───────────────────────────────
    def _train_loss(self, x_start, t, regime_id=None, target=None, noise=None, padding_masks=None):
        noise  = default(noise, lambda: torch.randn_like(x_start))
        target = x_start if target is None else target

        x         = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.output(x, t, regime_id=regime_id, padding_masks=padding_masks)

        train_loss = self.loss_fn(model_out, target, reduction='none')

        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = (
                self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')
                + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            )
            train_loss = train_loss + self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    # ── forward：RevIN归一化 + 训练loss ──────────────────────
    def forward(self, x, regime_id=None, **kwargs):
        """
        x         : (B, T, F)
        regime_id : (B,)  长整型tensor，值为0/1/2
        """
        b, seq_len, n_feat = x.shape
        assert n_feat == self.feature_size, \
            f"feature_size mismatch: got {n_feat}, expected {self.feature_size}"

        # RevIN归一化
        x_norm = self.revin(x)

        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        return self._train_loss(x_start=x_norm, t=t, regime_id=regime_id, **kwargs)

    # ── 采样：可指定regime，生成后反归一化 ───────────────────
    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        regime_id: int | torch.Tensor,
        ref_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        生成指定market regime下的合成时序。

        参数
        ----
        batch_size : 生成样本数
        regime_id  : int或(B,)的tensor，指定生成哪种市场状态
        ref_x      : 可选，一个真实样本batch (B,T,F)，用于设置RevIN的均值/方差
                     不传则用标准正态的统计量（生成的序列是归一化空间的）

        返回
        ----
        generated : (batch_size, T, F)，已反归一化到原始尺度
        """
        device = self.betas.device

        # 处理regime_id
        if isinstance(regime_id, int):
            regime_tensor = torch.full((batch_size,), regime_id, dtype=torch.long, device=device)
        else:
            regime_tensor = regime_id.to(device)

        shape = (batch_size, self.seq_length, self.feature_size)

        # 用fast_sample还是sample
        if self.fast_sampling:
            generated_norm = self._fast_sample_with_regime(shape, regime_tensor)
        else:
            generated_norm = self._sample_with_regime(shape, regime_tensor)

        # 反归一化
        if ref_x is not None:
            # 用真实样本的统计量设置RevIN
            _ = self.revin(ref_x.to(device))
        
       

        generated = self.revin.inverse(generated_norm)
        return generated

    @torch.no_grad()
    def generate_unconditional(
        self,
        batch_size: int,
        ref_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        无条件生成，label_emb=None，退化为普通扩散模型。
        用于消融实验baseline。
        """
        device = self.betas.device
        shape  = (batch_size, self.seq_length, self.feature_size)

        if self.fast_sampling:
            generated_norm = self._fast_sample_with_regime(shape, regime_tensor=None)
        else:
            generated_norm = self._sample_with_regime(shape, regime_tensor=None)

        if ref_x is not None:
            self.revin(ref_x.to(device))  # 去掉 _ = ，让统计量正确缓存
        
        

        generated = self.revin.inverse(generated_norm)
        return generated

    @torch.no_grad()
    def _sample_with_regime(self, shape, regime_tensor) -> torch.Tensor:
        device = self.betas.device
        img    = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            batched_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            model_mean, _, model_log_variance, _ = self._p_mean_variance_cond(
                img, batched_t, regime_tensor
            )
            noise = torch.randn_like(img) if t > 0 else 0.
            img   = model_mean + (0.5 * model_log_variance).exp() * noise
        return img

    @torch.no_grad()
    def _fast_sample_with_regime(self, shape, regime_tensor: torch.Tensor) -> torch.Tensor:
        device = self.betas.device
        batch  = shape[0]

        times      = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times      = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img        = torch.randn(shape, device=device)

        for time, time_next in time_pairs:
            time_cond  = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self._model_predictions_cond(img, time_cond, regime_tensor)

            if time_next < 0:
                img = x_start
                continue

            alpha      = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma      = self.eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c          = (1 - alpha_next - sigma ** 2).sqrt()
            noise      = torch.randn_like(img)
            img        = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    # ── 带条件的内部预测函数 ──────────────────────────────────
    def _model_predictions_cond(self, x, t, regime_tensor, clip_x_start=False):
        from Models.interpretable_diffusion.model_utils import identity
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start    = self.output(x, t, regime_id=regime_tensor)
        x_start    = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def _p_mean_variance_cond(self, x, t, regime_tensor, clip_denoised=True):
        _, x_start = self._model_predictions_cond(x, t, regime_tensor)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start


# ── 3. 快速验证（直接运行此文件时执行）──────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 模拟一个batch：32个样本，窗口长20，11个特征
    B, T, F = 32, 20, 11
    x         = torch.randn(B, T, F).to(device)
    regime_id = torch.randint(0, 3, (B,)).to(device)

    model = MarketConditionedDiffusion(
        seq_length   = T,
        feature_size = F,
        n_regimes    = 3,
        d_model      = 64,
        n_layer_enc  = 2,
        n_layer_dec  = 3,
        n_heads      = 4,
        timesteps    = 100,
        sampling_timesteps = 10,
        beta_schedule = 'cosine',
    ).to(device)

    # 测试forward（训练）
    loss = model(x, regime_id=regime_id)
    print(f"训练loss: {loss.item():.4f}")

    # 测试generate（采样）
    generated = model.generate(batch_size=4, regime_id=0, ref_x=x[:4])
    print(f"生成shape: {generated.shape}")   # 期望 (4, 20, 11)
    print("✅ MCD模型验证通过")