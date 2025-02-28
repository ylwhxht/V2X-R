import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial
import time
from ...utils.MDD_utils import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config, extract_into_tensor, make_beta_schedule, noise_like, detach
import os
from opencood.models.attresnet_modules.self_attn import AttFusion
from opencood.models.attresnet_modules.auto_encoder import AutoEncoder
from opencood.models.mdd_modules.unet import DiffusionUNet
INTERPOLATE_MODE = 'bilinear'

def tolist(a):
    try:
        return [tolist(i) for i in a]
    except TypeError:
        return a

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    from DIT
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv):
        B, N, C = x.shape
        _, T, _ = kv.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (B, heads, N, C//heads)
        kv = self.kv(kv).reshape(B, T, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (2, B, heads, T, C//heads)
        k, v = kv[0], kv[1]# (B, heads, T, C//heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, heads, N, T), N is 1
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.mlp_kv = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_kv = norm_layer(dim)
        self.norm2_kv = norm_layer(dim)

    def forward(self, x, kv):
        kv = kv + self.drop_path(self.mlp_kv(self.norm2_kv(kv)))

        x = x + self.drop_path(self.attn(self.norm1(x), self.norm_kv(kv)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, kv

class Denosier(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.depth = 4
        self.crossblocks = nn.ModuleList([CrossBlock(embed_dim, num_heads=4) for _ in range(self.depth)])

        self.pre_layer = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        
        self.initialize_weights()

        self.t_embedder = TimestepEmbedder(embed_dim)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
        self.apply(_basic_init)


    def forward(self, feat, noisy_masks, upsam=False, t=None):

        out = {}
        t = self.t_embedder(t)[:,:,None,None]

        resize_scale = 0.25
        cur_feat = F.interpolate(feat, scale_factor=resize_scale, mode='bilinear', align_corners=False)
        B, C, H, W = cur_feat.shape
        kv = cur_feat.reshape(B, C, -1).transpose(1,2) # (B, N, C)

        x = noisy_masks
        x = self.pre_layer(x)
        x = x + t
        x0 = x
        cur_kv = kv

        x = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        B, C, H, W = x.shape
        x = x.reshape(B,C,-1).transpose(1,2)

        for _d in range(self.depth):
            cur_kv, x = self.crossblocks[_d](cur_kv, x)
        x = cur_kv

        x = x.reshape(B, H, W, C).permute(0,3,1,2)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        out = x + x0
        
        return out


class Config:
	def __init__(self, entries: dict={}):
		for k, v in entries.items():
			if isinstance(v, dict):
				self.__dict__[k] = Config(v)
			else:
				self.__dict__[k] = v

class Cond_Diff_Denoise(nn.Module):
    def __init__(self,  model_cfg, embed_dim,):
        super().__init__()
        ### hyper-parameters
        # self.parameterization = 'eps'
        self.parameterization = 'x0'
        beta_schedule="linear"
        config = Config(model_cfg)
        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        timesteps = config.diffusion.num_diffusion_timesteps
        linear_start=5e-3
        linear_end=5e-2
        self.v_posterior = v_posterior =0 
        self.loss_type="l2"
        self.signal_scaling_rate = 1
        ###

        new_embed_dim = embed_dim
        #self.denoiser = Denosier(new_embed_dim)
        
        self.denoiser = DiffusionUNet(config)

        # q sampling
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        # diffusion loss
        learn_logvar = False
        logvar_init = 0
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        self.l_simple_weight = 1.

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        if len(lvlb_weights) > 1:
            lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()



    def q_sample(self, x_start, t, noise=None):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, feat, upsam, noisy_masks, t, clip_denoised: bool):
        model_out = self.gen_pred(feat, noisy_masks, upsam, t)

        x = noisy_masks
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        if upsam: # last sampling step
            model_mean = x_recon
            posterior_variance, posterior_log_variance = 0, 0
        else:
            x_recon = F.interpolate(x_recon, x.shape[2:], mode=INTERPOLATE_MODE, align_corners=False)
            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t) # Why do we need this?

        return model_mean, posterior_variance, posterior_log_variance, model_out


    def p_sample(self, feat, noisy_masks, t, upsam, clip_denoised=False, repeat_noise=False):
        model_mean, _, model_log_variance, model_out = self.p_mean_variance(feat, upsam, noisy_masks, t=t, clip_denoised=clip_denoised)

        x = noisy_masks
        b, *_, device = *x.shape, x.device
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if not upsam:
            out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise #, model_out
        else:
            out = model_mean

        return out

    def gen_pred(self, feat, noisy_masks, upsam=False, t=None):
        model_out = self.denoiser(torch.cat([feat, noisy_masks], dim=1), t.float())
        return model_out

    def p_sample_loop(self, feat, noisy_masks, latent_shape):
        b = latent_shape[0]
        num_timesteps = self.num_timesteps 
        
        for t in reversed(range(0, num_timesteps)):
            noisy_masks = F.interpolate(noisy_masks, latent_shape[2:], mode=INTERPOLATE_MODE, align_corners=False)
            noisy_masks = self.p_sample(feat, noisy_masks, torch.full((b,), t, device=feat.device, dtype=torch.long), 
                                                upsam=True if t==0 else False)
        return noisy_masks

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        ra_spatial_features = data_dict['ra_spatial_features']
        
        combined_pred = ra_spatial_features

        if self.training:
            bs = spatial_features.shape[0]
            all_noisy_masks = [] 
            for b_idx in range(bs):
                x = combined_pred[b_idx:b_idx+1]
                latent_shape = x.shape
                x_start = spatial_features[b_idx:b_idx+1]
                t = torch.full((x.shape[0],), self.num_timesteps-1, device=x.device, dtype=torch.long)
                noise = default(None, lambda: torch.randn_like(x_start))
                _x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                noisy_masks = _x_noisy
                for _t in reversed(range(1, self.num_timesteps)):
                    _t = torch.full((x.shape[0],), _t, device=x.device, dtype=torch.long)
                    noisy_masks = self.p_sample(x, noisy_masks, _t, upsam=False)
                _t = 0
                _t = torch.full((x.shape[0],), _t, device=x.device, dtype=torch.long)
                noisy_masks = self.p_sample(x, noisy_masks, _t, upsam=True)
                all_noisy_masks.append(noisy_masks)
            data_dict['pred_feature'] = torch.stack(all_noisy_masks,dim=0).squeeze()
        else:
            t1 = time.time()
            noisy_masks = {}
            x = combined_pred
            latent_shape = x.shape
            x_start = spatial_features * self.signal_scaling_rate #batch['mask_'+task]
            noise = default(None, lambda: torch.randn_like(x_start))
            t = torch.ones(x.shape[0], device=x.device).long() * (self.num_timesteps - 1)
            _x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            noisy_masks= _x_noisy
            t2 = time.time()
            noisy_masks = self.p_sample_loop(x, noisy_masks, x.shape)
            all_noisy_masks = noisy_masks
            t3 = time.time()
            #print(t2-t1,t3-t2)
            data_dict['pred_feature'] = all_noisy_masks
            

        return data_dict
