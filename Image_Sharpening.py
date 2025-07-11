pip install datasets==3.6.0
git clone https://github.com/swz30/MIRNetv2.git
git clone https://github.com/swz30/Restormer.git
pip install piq Augmentor torch torchvision natsort kornia

wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/enhancement_fivek.pth -P /content/MIRNetv2/pretrained_models/

import os
import cv2
import sys
import piq

import torch
import shutil
import kornia
import datasets
import argparse
import Augmentor
import torchvision
import numpy as np

from glob import glob
import torch.nn as nn
from tqdm import tqdm
from runpy import run_path
import torch.optim as optim
from natsort import natsorted
from google.colab import drive
from torch.nn import Sequential
import matplotlib.pyplot as plt
import torch.nn.functional as func
from torchvision import transforms
from PIL import ImageFilter , Image

from skimage.util import img_as_ubyte
from datasets import Dataset as dataset_cap
from torch.utils.data import DataLoader , Dataset
from torchvision.transforms.functional import to_tensor , to_pil_image

drive.mount('/content/Google_D')    # Give all permissions for successful connection
dataset = datasets.load_dataset("goodfellowliu/DIV2K" , split = "train").select(range(40))

data_folder = "/content/DIV2K/Data"
output_folder = "/content/DIV2K/Modified"
process_folder = "/content/DIV2K/Proces_Aug"
original_folder = "/content/DIV2K/Originals"

shutil.rmtree(data_folder , ignore_errors = True)
shutil.rmtree(output_folder , ignore_errors = True)
shutil.rmtree(original_folder , ignore_errors = True)

os.makedirs(data_folder , exist_ok = True)
os.makedirs(output_folder , exist_ok = True)
os.makedirs(original_folder , exist_ok = True)

filter = ImageFilter.GaussianBlur(radius = 2.2)

for num , sample in enumerate(dataset) :
  image = sample["image"]
  image.save(f"{data_folder}/Image_{num}.png")

  img = np.array(image).astype(np.float32)
  img_mod = torch.from_numpy(img).permute(2 , 0 , 1)

  c , h , w = img_mod.shape

  pad_H = ((h + 255) // 256) * 256 - h
  pad_W = ((w + 255) // 256) * 256 - w
  img_mod = func.pad(img_mod , (0 , pad_W , 0 , pad_H) , mode = "reflect")

  patches = img_mod.unfold(1 , 256 , 256).unfold(2 , 256 , 256)

  patches = patches.permute(1 , 2 , 0 , 3 , 4)
  patches = patches.reshape(-1 , c , 256 , 256)

  for number , patch in enumerate(patches) :
    to_pil_image(patch).save(f"{original_folder}/Image_{num}_{number}.png")
    to_pil_image(patch).filter(filter).save(f"{output_folder}/Image_{num}_{number}.png")

shutil.rmtree(process_folder , ignore_errors = True)

os.makedirs(process_folder , exist_ok = True)
augmentor = Augmentor.Pipeline(output_folder , process_folder)

augmentor.random_contrast(probability = 1.0 , min_factor = 0.7 , max_factor = 0.9)
augmentor.random_brightness(probability = 1.0 , min_factor = 0.7 , max_factor = 0.9)

augmentor.process()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parameters = {'inp_channels' : 3 , 'out_channels' : 3 , 'n_feat': 80 , 'chan_factor' : 1.5 , 'n_MRB' : 2 , 'n_RRG' : 4 , 'height' : 3 , 'width' : 2 , 'bias' : False , 'scale': 1}
parameters_2 = {'inp_channels' : 3 , 'out_channels' : 3 , 'dim' : 48 , 'num_blocks' : [4 , 6 , 6 , 8] , 'num_refinement_blocks' : 4 , 'heads' : [1 , 2 , 4 , 8] , 'ffn_expansion_factor' : 2.66 , 'bias' : False , 'LayerNorm_type' : 'WithBias'}

model_2 = run_path("/content/Restormer/basicsr/models/archs/restormer_arch.py")['Restormer'](**parameters_2).to(device)
model_2.load_state_dict(torch.load("" , map_location = device)["params"] , strict = True)   # Download the single_image_defocus_deblurring model from https://drive.google.com/drive/folders/1bRBG8DG_72AGA6-eRePvChlT5ZO4cwJ4 or you can copy file to your Drive and copy path here

model = run_path("/content/MIRNetv2/basicsr/models/archs/mirnet_v2_arch.py")['MIRNet_v2'](**parameters).to(device)
model.load_state_dict(torch.load("/content/MIRNetv2/pretrained_models/enhancement_fivek.pth" , map_location = device)["params"] , strict = True)

model.eval()
model_2.eval()

image_dir = process_folder
folder_path = "/content/DIV2K/Processed_Teach"
shutil.rmtree(folder_path , ignore_errors = True)

os.mkdir(folder_path)
dir_img = sorted(os.listdir(image_dir))

file_list = [os.path.join(image_dir , f) for f in dir_img]
dataset = dataset_cap.from_list([{"Image" : name} for name in file_list])

for num , sample in enumerate(dataset) :
  img = Image.open(sample["Image"])
  img = np.array(img).astype(np.float32) / 255.0

  lr_tensor = torch.from_numpy(img).permute(2 , 0 , 1).unsqueeze(0).to(device)

  with torch.no_grad() :
      sr_tensor = model(model_2(lr_tensor))

  output_img = to_pil_image(sr_tensor.squeeze().clamp(0 , 1) , mode = "RGB")
  output_img.save(f"/content/DIV2K/Processed_Teach/Image_{sample['Image'].split('/')[4].split('_' , 3)[3].split('.')[0]}.png")

plt.imshow(Image.open("/content/DIV2K/Processed_Teach/Image_0_0.png"))

class ConvBlock(nn.Module) :
    def __init__(self , in_ch , out_ch) :
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch , out_ch , kernel_size = 3 , padding = 1) , nn.LeakyReLU(inplace = True) , nn.Conv2d(out_ch , out_ch , kernel_size = 3 , padding = 1 ) , nn.LeakyReLU(inplace = True))

    def forward(self , x) :
        return self.block(x)

class StudentNet(nn.Module):
    def __init__(self, in_channels = 3 , out_channels = 3 , features = 32) :
        super().__init__()

        self.enc1 = ConvBlock(in_channels , features)
        self.enc2 = ConvBlock(features , features * 2)

        self.enc3 = ConvBlock(features * 2 , features * 4)

        self.pool = nn.MaxPool2d(kernel_size = 2 , stride = 2)
        self.bottleneck = ConvBlock(features * 4 , features * 8)

        self.up3 = nn.Upsample(scale_factor = 2 , mode = "bicubic" , align_corners = False)
        self.dec3 = ConvBlock(features * 12 , features * 4)

        self.up2 = nn.Upsample(scale_factor = 2 , mode = "bicubic" , align_corners = False)
        self.dec2 = ConvBlock(features * 6 , features * 2)

        self.up1 = nn.Upsample(scale_factor = 2 , mode = "bicubic" , align_corners = False)
        self.dec1 = ConvBlock(features * 3 , features)

        self.final = nn.Conv2d(features , out_channels , kernel_size = 1)

    def forward(self, x):
        enc1 = self.enc1(x)

        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        bottleneck = self.bottleneck(self.pool(enc3))

        dec3 = self.up3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)

        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.dec1(dec1)
        return x + self.final(dec1)

class DistillationDataset(Dataset) :
    def __init__(self , modified_dir , teacher_dir , original_dir) :
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.gt_paths = sorted([os.path.join(original_dir , f) for f in os.listdir(original_dir)])
        self.teacher_paths = sorted([os.path.join(teacher_dir , f) for f in os.listdir(teacher_dir)])
        self.modified_paths = sorted([os.path.join(modified_dir , f) for f in os.listdir(modified_dir)])

    def __len__(self) :
        return len(self.modified_paths)

    def __getitem__(self , idx) :
        gt = Image.open(self.gt_paths[idx])
        inp = Image.open(self.modified_paths[idx])
        teacher_out = Image.open(self.teacher_paths[idx])

        return {"input" : self.transform(inp) , "teacher" : self.transform(teacher_out) , "gt" : self.transform(gt)}

def distillation_loss(student_out , gt , teacher_out) :
    l1_loss = func.l1_loss(student_out , gt)
    mse_loss = func.mse_loss(student_out , teacher_out)
    gradient_loss = func.l1_loss(kornia.filters.sobel(student_out) , kornia.filters.sobel(gt))

    contrast_loss = func.l1_loss(student_out.clamp(0 , 1).std(dim = (2 , 3)) , gt.clamp(0 , 1).std(dim = (2 , 3)))
    brightness_loss = func.l1_loss(student_out.clamp(0 , 1).mean(dim = (2 , 3)) , teacher_out.clamp(0 , 1).mean(dim = (2 , 3)))

    distillation_loss = func.l1_loss(kornia.filters.sobel(student_out) , kornia.filters.sobel(teacher_out))
    structural_sim_loss = piq.ssim(student_out.clamp(0 , 1) , gt.clamp(0 , 1) , data_range = 1.0 , reduction = "mean" , downsample = False)

    return 0.05 * l1_loss + 0.05 * mse_loss + 0.05 * distillation_loss + 0.05 * brightness_loss + 0.20 * gradient_loss + 0.20 * contrast_loss + 0.20 * (1.0 - structural_sim_loss)

def train_student(student , database , optimizer , epochs = 10) :
    student.train()

    for epoch in range(epochs) :
        total_loss = 0.0

        for batch in database :
            student.zero_grad()

            gt = batch["gt"].to(device)
            inp = batch["input"].to(device)
            teacher_out = batch["teacher"].to(device)

            student_out = student(inp)
            loss = distillation_loss(student_out , gt , teacher_out)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(database)
        print(f"Epoch {epoch + 1} / {epochs} , Loss : {avg_loss : .4f}")

original_dir = "/content/DIV2K/Originals"
modified_dir = "/content/DIV2K/Proces_Aug"

processed_dir = "/content/DIV2K/Processed_Teach"

student = StudentNet().to(device)
dataset = DistillationDataset(modified_dir , processed_dir , original_dir)

loader = DataLoader(dataset , batch_size = 16 , shuffle = True)
optimizer = optim.Adam(student.parameters() , lr = 2e-4 , weight_decay = 0.0001)

train_student(student = student , database = loader , optimizer = optimizer , epochs = 20)

student.eval()
img = datasets.load_dataset("goodfellowliu/DIV2K" , split = "train").select(range(100))[90]["image"]    # You can also use any other image (size should be greater than 256 x 256)

image = img.filter(filter)
h , c = image.size[0] , image.size[1]
img.save("/content/Original_Img.png")

img_arr = np.array(image)
image = augmentor._execute_with_array(img_arr)

plt.imshow(image)
plt.title("Modified_Image" , loc = "center")
img_mod = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2 , 0 , 1)

c , h , w = img_mod.shape

pad_H = ((h + 255) // 256) * 256 - h
pad_W = ((w + 255) // 256) * 256 - w
img_mod = func.pad(img_mod , (0 , pad_W , 0 , pad_H) , mode = "reflect")

patches = img_mod.unfold(1 , 256 , 256).unfold(2 , 256 , 256)

patches = patches.permute(1 , 2 , 0 , 3 , 4)
patches = patches.reshape(-1 , c , 256 , 256).to(device)

with torch.no_grad() :
    output_tensor = student(patches)

N = output_tensor.shape[0]
output_tensor = output_tensor.reshape(N , -1).T.unsqueeze(0)

output_img = func.fold(output_tensor , output_size = (img_mod.shape[1] , img_mod.shape[2]) , kernel_size = (256 , 256) , stride = (256 , 256)).squeeze(0)

to_pil_image(output_img[ : , : h , : w].clamp(0 , 1) , mode = "RGB").save("/content/Output_Student_Img.png")

orig = torch.from_numpy(np.array(Image.open("/content/Original_Img.png"))).float().permute(2 , 0 , 1).unsqueeze(0)
modified = torch.from_numpy(np.array(Image.open("/content/Output_Student_Img.png"))).float().permute(2 , 0 , 1).unsqueeze(0)

print(f"The SOBEL score is : " + str(func.l1_loss(kornia.filters.sobel(modified) , kornia.filters.sobel(orig))))
print(f"The SSIM score is : " + str(piq.ssim(modified.clamp(0 , 1) , orig.clamp(0 , 1) , data_range = 1.0 , reduction = "mean")))
