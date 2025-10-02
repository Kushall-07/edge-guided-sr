import os, argparse, yaml, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data.datasets import PairedThermalOpticalDataset
from .models.edge_sr import EdgeGuidedSR
from .losses import L1Loss, edge_alignment_loss

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    return ap.parse_args()

def main():
    args = get_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_train = PairedThermalOpticalDataset(cfg['data']['thermal_dir'], cfg['data']['optical_dir'],
                                           tile_size=cfg['data']['tile_size'], stride=cfg['data']['stride'],
                                           scale=cfg['model']['scale'], split='train', split_ratio=cfg['data']['train_split'])
    ds_val   = PairedThermalOpticalDataset(cfg['data']['thermal_dir'], cfg['data']['optical_dir'],
                                           tile_size=cfg['data']['tile_size'], stride=cfg['data']['stride'],
                                           scale=cfg['model']['scale'], split='val', split_ratio=cfg['data']['train_split'])
    dl_train = DataLoader(ds_train, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    dl_val   = DataLoader(ds_val,   batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])

    model = EdgeGuidedSR(scale=cfg['model']['scale'],
                         in_ch_thermal=cfg['model']['in_ch_thermal'],
                         in_ch_optical=cfg['model']['in_ch_optical'],
                         ch=cfg['model']['base_ch']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    l1 = L1Loss()

    os.makedirs(cfg['train']['ckpt_dir'], exist_ok=True)
    best = 1e9

    for epoch in range(cfg['train']['epochs']):
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for batch in pbar:
            t_lr = batch['t_lr'].to(device)
            o_hr = batch['o_hr'].to(device)
            sr = model(t_lr, o_hr)
            target = F.interpolate(t_lr, scale_factor=cfg['model']['scale'], mode='bilinear', align_corners=False)
            loss = l1(sr, target) * cfg['loss']['w_l1'] + edge_alignment_loss(sr, o_hr) * cfg['loss']['w_edge']
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        model.eval(); val_loss=0.0
        with torch.no_grad():
            for batch in dl_val:
                t_lr = batch['t_lr'].to(device)
                o_hr = batch['o_hr'].to(device)
                sr = model(t_lr, o_hr)
                target = F.interpolate(t_lr, scale_factor=cfg['model']['scale'], mode='bilinear', align_corners=False)
                val_loss += l1(sr, target).item()
        val_loss /= max(1, len(dl_val)); print(f"Val loss: {val_loss:.4f}")
        torch.save({'model': model.state_dict(), 'cfg': cfg}, os.path.join(cfg['train']['ckpt_dir'], 'last.ckpt'))
        if val_loss < best:
            best = val_loss
            torch.save({'model': model.state_dict(), 'cfg': cfg}, os.path.join(cfg['train']['ckpt_dir'], 'best.ckpt'))
            print('Saved best checkpoint.')

if __name__ == '__main__':
    main()
