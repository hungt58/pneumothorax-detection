"""Tune segmentation threshold on the validation split only."""
import argparse, csv, random
from pathlib import Path
import numpy as np
import torch
from build_dataset import build_dataloaders, build_file_list, split_data
from models import UNet

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--images",type=Path,required=True); p.add_argument("--masks",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--num-workers",type=int,default=2); p.add_argument("--output",type=Path,default=Path("results/unet/threshold_tuning.csv"))
    return p.parse_args()

@torch.inference_mode()
def main():
    a=args(); random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:",device)
    records=build_file_list(str(a.images),str(a.masks))
    train,val,test=split_data(records)
    _,loader,_=build_dataloaders(train,val,test,batch_size=a.batch_size,num_workers=a.num_workers)
    print(f"Pairs: {len(records)} | train: {len(train)} | val: {len(val)} | test: {len(test)}")
    print("Threshold tuning: VALIDATION only; test set untouched.")
    ckpt=torch.load(a.checkpoint,map_location=device)
    features=tuple(ckpt.get("features",(32,64,128,256)))
    model=UNet(features=features).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    probs=[]; masks=[]
    for i,b in enumerate(loader,1):
        probs.append(torch.sigmoid(model(b["image"].to(device))).cpu())
        masks.append(b["mask"].cpu())
        if i==1 or i%50==0: print(f"  inference batch {i:04d}")
    probs=torch.cat(probs); masks=torch.cat(masks)
    mf=masks.flatten(1); target_pos=mf.sum(1)>0
    rows=[]
    print("\nthr   dice+   iou+    pred+rate  dice_all  iou_all")
    for t in np.arange(.30,.901,.05):
        pred=(probs>=float(t)).float(); pf=pred.flatten(1)
        inter=(pf*mf).sum(1); ps=pf.sum(1); ms=mf.sum(1); union=ps+ms-inter
        dice=(2*inter+1)/(ps+ms+1); iou=(inter+1)/(union+1)
        row={"threshold":float(t),"dice_positive":dice[target_pos].mean().item(),
             "iou_positive":iou[target_pos].mean().item(),
             "predicted_positive_rate":(ps>0).float().mean().item(),
             "dice_all":dice.mean().item(),"iou_all":iou.mean().item()}
        rows.append(row)
        print(f"{t:.2f}  {row['dice_positive']:.4f}  {row['iou_positive']:.4f}  {row['predicted_positive_rate']:.4f}     {row['dice_all']:.4f}    {row['iou_all']:.4f}")
    best=max(rows,key=lambda r:r["dice_positive"])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("\nBEST (validation dice_positive)")
    for k,v in best.items(): print(f"{k}: {v:.4f}")
    print("Saved:",a.output)
if __name__=="__main__": main()
