import numpy as np
import pandas as pd

def save_matches(outputs,batch_idx):
    keypoints_detected = outputs['matches'][0, :, :]
    print(np.nonzero(keypoints_detected))
    indices=np.nonzero(keypoints_detected)

    pd.DataFrame(outputs['landmarks_1'][0, :, :].cpu().detach().numpy()).to_csv("landmarks1.csv", header=None,  index=None)
    pd.DataFrame(outputs['landmarks_2'][0, :, :].cpu().detach().numpy()).to_csv("landmarks2.csv", header=None,  index=None)
    pd.DataFrame(outputs['matches'][0, :, :].cpu().detach().numpy()).to_csv("matches.csv", index=None)

    kpts1=[]
    kpts2=[]

    for j in indices:
        # print(type(outputs['landmarks_1'][0, j[0].item(), :].tolist()))
        kpts1.append(outputs['landmarks_1'][0, j[0].item(), :].tolist())
        kpts2.append(outputs['landmarks_2'][0, j[1].item(), :].tolist())
        # print(outputs['landmarks_1'][0, j[0].item(), :].tolist())
        # print(outputs['landmarks_2'][0, j[1].item(), :].tolist())
    # print(outputs['landmarks_1'][0, :, :])
    #
    with open('img_{}matches1.txt'.format(batch_idx), 'w') as f:
        for line in kpts1:
            f.write(f"{line}\n")

    with open('img_{}matches2.txt'.format(batch_idx), 'w') as f:
        for line in kpts2:
            f.write(f"{line}\n")