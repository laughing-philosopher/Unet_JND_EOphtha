import cv2
import numpy as np

from model import build_UNet

# Patch dimension details
Img_Width = 48
Img_Height = 48
Img_Channels = 1

model = build_UNet((Img_Width, Img_Height, Img_Channels),(Img_Width, Img_Height, Img_Channels))
model.load_weights('UNet_JND_EOphtha.h5')

def processing(img, threshold, batch_size):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # The initial processing of the image
    img_g = img1[:,:,1]

    # Applying CLAHE as pre-processing step
    clahe = cv2.createCLAHE(clipLimit = 8, tileGridSize=(8,8))
    img_c = clahe.apply(img_g)

    [m,n] = img_c.shape

    gt = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gt = gt / 255.0
    gt = (gt > 0.5)


    iarr = []
    jarr = []
    tot = 0
    step = 8
    for i in range(0,m,step):
        for j in range(0,n,step):
            if((i+48)>(m-1) or (j+48)>(n-1)):
                pass
            else:
                iarr.append(i)
                jarr.append(j)
                tot = tot + 1
    #  # increase this threshold if more false positives are coming
    
    
    # batch_size = 10



    current_batch_size = tot // batch_size
    cnt1 = 0
    cnt2 = 0
    final_res = np.zeros((m,n))
    for batch in range(batch_size):
        
        patches_img = np.zeros((current_batch_size,48,48,1))
        patches_gt = np.zeros((current_batch_size,48,48,1))
        

        for k in range(current_batch_size):
            # getting image patches
            itr = k + batch * current_batch_size
            if ((iarr[itr] + 48) > (m - 1) or (jarr[itr] + 48) > (n - 1)):
                pass
            else:
                patch_img = img_c[iarr[itr]:(iarr[itr]+48),jarr[itr]:(jarr[itr]+48)]
                patch_img = np.expand_dims(patch_img, axis=-1)
                patch_img = np.expand_dims(patch_img, axis=0)
                patches_img[k] = patch_img
                # getting gt patches
                patch_gt = gt[iarr[itr]:(iarr[itr]+48),jarr[itr]:(jarr[itr]+48)]
                patch_gt = np.expand_dims(patch_gt, axis=-1)
                patch_gt = np.expand_dims(patch_gt, axis=0)
                patches_gt[k] = patch_gt
                    
        print(patches_img.shape, batch*current_batch_size)
        inter_res,_ = model.predict([patches_img,patches_gt],verbose=False)
        # print(inter_res)
        # inter_res = (inter_res > threshold)
        for k in range(current_batch_size):
            itr=k+batch*current_batch_size
            final_res[iarr[itr]:(iarr[itr]+48),jarr[itr]:(jarr[itr]+48)] = final_res[iarr[itr]:(iarr[itr]+48),jarr[itr]:(jarr[itr]+48)] + np.squeeze(inter_res[k])
            
        del patches_img, patches_gt, inter_res
    
    if (tot % batch_size != 0):
        current_batch_size = tot % batch_size
        patches_img = np.zeros((current_batch_size,48,48,1))
        patches_gt = np.zeros((current_batch_size,48,48,1))
        

        for k in range(current_batch_size):
            # getting image patches
            itr = k + (tot - current_batch_size)
            if ((iarr[itr] + 48) > (m - 1) or (jarr[itr] + 48) > (n - 1)):
                pass
            else:
                patch_img = img_c[iarr[itr]:(iarr[itr]+48),jarr[itr]:(jarr[itr]+48)]
                patch_img = np.expand_dims(patch_img, axis=-1)
                patch_img = np.expand_dims(patch_img, axis=0)
                patches_img[k] = patch_img
                # getting gt patches
                patch_gt = gt[iarr[itr]:(iarr[itr]+48),jarr[itr]:(jarr[itr]+48)]
                patch_gt = np.expand_dims(patch_gt, axis=-1)
                patch_gt = np.expand_dims(patch_gt, axis=0)
                patches_gt[k] = patch_gt
                    
        print(patches_img.shape, batch*current_batch_size)
        inter_res,_ = model.predict([patches_img,patches_gt],verbose=False)
        # print(inter_res)
        # inter_res = (inter_res > threshold)
        for k in range(current_batch_size):
            itr=k+(tot - current_batch_size)
            final_res[iarr[itr]:(iarr[itr]+48),jarr[itr]:(jarr[itr]+48)] = final_res[iarr[itr]:(iarr[itr]+48),jarr[itr]:(jarr[itr]+48)] + np.squeeze(inter_res[k])
            
        del patches_img, patches_gt, inter_res
    
    psm_th2 = final_res / np.max(final_res)
    psm_th2 = (psm_th2 > 0.1).astype(float)

    return psm_th2
