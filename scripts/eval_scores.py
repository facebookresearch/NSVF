import cv2, torch
import numpy as np
import os, sys, glob
import skimage.measure, skimage.transform
import fairnr.criterions.models as LPIPS

def cv2tensor(im):
    return im2tensor(im[:,:,::-1])

def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def cv_lpips(model,pred,gt):
    with torch.no_grad():
        return float(model.forward(cv2tensor(pred),cv2tensor(gt)))

targets = sorted(glob.glob(os.path.join(sys.argv[1], "*.png")))
outputs = sorted(glob.glob(os.path.join(sys.argv[2], "*.png")))
assert len(targets) == len(outputs)
print("evaluating on {} images.".format(len(targets)))

tot_ssim = 0
tot_psnr = 0
tot_lpips = 0
cnt = 0.
model = LPIPS.PerceptualLoss(model='net-lin',net='alex',use_gpu=False)
for gt, pred in zip(targets, outputs):
    gt, pred = cv2.imread(gt), cv2.imread(pred)
    ssim = skimage.measure.compare_ssim(pred, gt, multichannel=True, data_range=255)
    psnr = skimage.measure.compare_psnr(pred, gt, data_range=255)
    lpips = cv_lpips(model, pred, gt)

    tot_ssim += ssim
    tot_psnr += psnr
    tot_lpips += lpips
    cnt += 1
    print('%d avg =%f %f %f'%(cnt,tot_psnr/cnt,tot_ssim/cnt,tot_lpips/cnt))