import os, argparse, yaml, torch, numpy as np, rasterio
from .models.edge_sr import EdgeGuidedSR

def write_tif(path, arr, ref_path):
    arr = np.asarray(arr, dtype='float32')
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        profile.update(count=1, dtype='float32')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--in_lr', required=True)
    ap.add_argument('--in_hr', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = ckpt['cfg']
    model = EdgeGuidedSR(scale=cfg['model']['scale'],
                         in_ch_thermal=cfg['model']['in_ch_thermal'],
                         in_ch_optical=cfg['model']['in_ch_optical'],
                         ch=cfg['model']['base_ch'])
    model.load_state_dict(ckpt['model']); model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in sorted([f for f in os.listdir(args.in_lr) if f.lower().endswith('.tif')]):
        lr_path = os.path.join(args.in_lr, name)
        hr_path = os.path.join(args.in_hr, name)
        if not os.path.isfile(hr_path):
            continue
        with rasterio.open(lr_path) as ds_t, rasterio.open(hr_path) as ds_o:
            t = ds_t.read().astype('float32')
            o = ds_o.read().astype('float32')
        t_t = torch.from_numpy(t[None])
        o_t = torch.from_numpy(o[None])
        with torch.no_grad():
            sr = model(t_t, o_t)[0,0].numpy()
        out_path = os.path.join(args.out_dir, name.replace('.tif', '_sr.tif'))
        write_tif(out_path, sr, hr_path)
        print('Wrote', out_path)

if __name__ == '__main__':
    main()
