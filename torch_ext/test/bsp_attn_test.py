import torch
import bsp_attn

def main():
    n = 1
    h = 32
    s = 1024
    l = 1024
    e = 128

    q = torch.rand(n, h, l, e).cuda()
    k = torch.rand(n, h, s, e).cuda()
    v = torch.rand(n, h, s, e).cuda()
    mask_offsets = (torch.arange(8) * 8).cuda()
    mask_indices = torch.remainder(torch.arange(64), 8).cuda()
    dropout = 0
    is_causal = False
    scale = e ** -0.5

    o = bsp_attn.scaled_dot_product_attention(q, k, v, mask_offsets, mask_indices, dropout, is_causal, scale)
    print(o)


if __name__ == '__main__':
    main()
