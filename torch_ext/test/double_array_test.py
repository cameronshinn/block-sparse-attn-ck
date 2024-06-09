import torch
import bsp_attn


def main():
    a = torch.rand(4).cuda()
    print(a)
    a2 = bsp_attn.double_array(a)
    print(a2)


if __name__ == '__main__':
    main()
