import torch
pth_path = '/mnt/lustre/zhaozhiyu.vendor/videoMAEOUT/K400_fuxian/ct_384_lr2e_5_2D_nmix_nrep_w1e_1_E40_dpath_25_dlast_50_origin_test_b4/checkpoint-39/'
pth_out = 'large384.pth'
ck = torch.load(pth_path+'mp_rank_00_model_states.pt')
del_keys = list(ck.keys())[1:]
for Dkey in del_keys:
    _ = ck.pop(Dkey)
torch.save(ck, pth_path+pth_out)
ck = torch.load(pth_path+pth_out)
del_keys = list(ck.keys())
print('load ok!')