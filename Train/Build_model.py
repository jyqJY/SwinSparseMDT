from SparseSwinMTD.Models.SparseSwinMTD import SparseSwin


def BuildSparseSwin(image_resolution, swin_type, num_classes,
                    ltoken_num, ltoken_dims, num_heads,
                    qkv_bias, lf, attn_drop_prob, lin_drop_prob,
                    freeze_12, device):

    dims = {'tiny': 96, 'small': 96, 'base': 128}
    dim_init = dims.get(swin_type.lower())

    if (dim_init is None) or (image_resolution % 16 != 0):
        print('Error: Check your swin type or image resolution.')
        return None

    model = SparseSwin(
        swin_type=swin_type,
        num_classes=num_classes,
        c_dim_3rd=dim_init * 4,
        hw_size_3rd=int(image_resolution / 16),
        ltoken_num=ltoken_num,
        ltoken_dims=ltoken_dims,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        lf=lf,
        attn_drop_prob=attn_drop_prob,
        lin_drop_prob=lin_drop_prob,
        freeze_12=freeze_12,
        device=device
    ).to(device)
    return model





if __name__ == '__main__':

    swin_type = 'tiny'
    device = 'cuda'
    image_resolution = 224
    num_classes = 4
    ltoken_num = 49
    ltoken_dims = 512
    num_heads = 16
    qkv_bias = True
    lf = 2

    attn_drop_prob = .0

    lin_drop_prob = .0
    freeze_12 = False




    model = BuildSparseSwin(
        image_resolution=image_resolution,
        swin_type=swin_type,
        num_classes=num_classes,
        ltoken_num=ltoken_num,
        ltoken_dims=ltoken_dims,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        lf=lf,
        attn_drop_prob=attn_drop_prob,
        lin_drop_prob=lin_drop_prob,
        freeze_12=freeze_12,
        device=device
    )



