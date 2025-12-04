
import glob
from PIL import Image
import numpy as np
import os
import torch
import time
import imageio
import torchvision.transforms as transforms
from twoDomainModel import Model as net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(torch.cuda.is_available())
model = net(in_channel=2)

model_path = r"/CAFusion/models/med.pth"
use_gpu = torch.cuda.is_available()
# img1_dataset = sorted(glob.glob(r"D:\Dataset\MY_DATASET(2025-2-27)\SPECT-MRI\test_data\SPECT-MRI\MRI\*"))
#mri_pet
img1_dataset = sorted(glob.glob(r"D:\Dataset\MY_DATASET(2025-2-27)\PET-MRI\Test_Data\PET-MRI\MRI\*"))

# img1_dataset = sorted(glob.glob(r"D:\Dataset\GFP-PC\test1\GFP\*"))
# img1_dataset = sorted(glob.glob(r"D:\Dataset\LLVIP\LLVIP\visible\test\*"))[0:40]
# img1_dataset = sorted(glob.glob(r"D:\Dataset\RoadScene-master\RoadScene-master\crop_LR_visible\*"))[0:40]
# img1_dataset = sorted(glob.glob(r"D:\Dataset\MSRS-main\MSRS-main\test\ir\*"))[0:50]
# img1_dataset = sorted(glob.glob(r"D:\Dataset\TNO_my\ir\*"))
img_name = []
for i in range(len(img1_dataset)):
    img_name.append(img1_dataset[i].split("\\")[-1])
    print(img1_dataset[i].split("\\")[-1])
print(img_name)
if use_gpu:

    model = model.cuda()
    model.cuda()

    model.load_state_dict(torch.load(model_path), strict=True)

else:

    state_dict = torch.load(model_path, map_location='cpu',strict=True)

    model.load_state_dict(state_dict)


def fusion():
    for num in range(len(img_name)):
        tic = time.time()
        # path1 = r'D:\coding\pycharm_projects\paper3\9_28_Twodomainmodel\datasets\SPECT-MRI\test_data\SPECT-MRI\SPECT\{}'.format(img_name[num])
        # path2 = r'D:\coding\pycharm_projects\paper3\9_28_Twodomainmodel\datasets\SPECT-MRI\test_data\SPECT-MRI\MRI\{}'.format(img_name[num])
        path1 = r'D:\Dataset\MY_DATASET(2025-2-27)\PET-MRI\Test_Data\PET-MRI\PET\{}'.format(img_name[num])
        path2 = r'D:\Dataset\MY_DATASET(2025-2-27)\PET-MRI\Test_Data\PET-MRI\MRI\{}'.format(img_name[num])
        # path1 = r'D:\Dataset\GFP-PC\test1\GFP\{}'.format(img_name[num])
        # path2 = r'D:\Dataset\GFP-PC\test1\PC\{}'.format(img_name[num])
        # path1 = r'D:\Dataset\LLVIP\LLVIP\infrared\test\{}'.format(img_name[num])
        # path2 = r'D:\Dataset\LLVIP\LLVIP\visible\test\{}'.format(img_name[num])
        # path1 = r'D:\Dataset\RoadScene-master\RoadScene-master\cropinfrared\{}'.format(img_name[num])
        # path2 = r'D:\Dataset\RoadScene-master\RoadScene-master\crop_LR_visible\{}'.format(img_name[num])
        # path1 = r'D:\Dataset\TNO_my\ir\{}'.format(img_name[num])
        # path2 = r'D:\Dataset\TNO_my\vi\{}'.format(img_name[num])
        # path1 = r"D:\Dataset\MSRS-main\MSRS-main\test\ir\{}".format(img_name[num])
        # path2 = r"D:\Dataset\MSRS-main\MSRS-main\test\vi\{}".format(img_name[num])
        img1 = Image.open(path1).convert('L')
        img2 = Image.open(path2).convert('L')
        img1_org = img1
        img2_org = img2
        tran = transforms.ToTensor()
        img1_org = tran(img1_org)
        img2_org = tran(img2_org)
        input_img = torch.cat((img1_org, img2_org), 0).unsqueeze(0)
        if use_gpu:
            input_img = input_img.cuda()
        else:
            input_img = input_img
        model.eval()
        print(input_img.shape)
        out = model(input_img)
        d = np.squeeze(out.detach().cpu().numpy())
        result = (d* 255).astype(np.uint8)
        try:
            os.mkdir(r"fusion_out/Y/PET/")
        except FileExistsError:
            pass  # 文件夹已存在，不做处理

        imageio.imwrite(
            r'fusion_out/PET/{}'.format(img_name[num]),result)

        toc = time.time()
        print('end  {}{}'.format(num // 10, num % 10), ', time:{}'.format(toc - tic))

if __name__ == '__main__':

    fusion()
    print("融合完成！")