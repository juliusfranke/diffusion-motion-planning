import torch


def to_tanh_space(x, lo, hi, eps=1e-6):
    # x in [lo, hi]  -> z in R
    u = (x - lo) / (hi - lo)
    v = torch.clamp(2 * u - 1, -1 + eps, 1 - eps)
    z = 0.5 * torch.log((1 + v) / (1 - v))
    return z


def from_tanh_space(z, lo, hi):
    v = torch.tanh(z)
    u = (v + 1) * 0.5
    x = lo + (hi - lo) * u
    return x


# # Special case for actions in [-0.5, 0.5]:
# def actions_to_z(a):
#     a = torch.clamp(a, -0.5 + EPS, 0.5 - EPS)
#     return 0.5 * torch.log((1 + 2 * a) / (1 - 2 * a))  # atanh(2a)


# def z_to_actions(z):
#     return 0.5 * torch.tanh(z)
