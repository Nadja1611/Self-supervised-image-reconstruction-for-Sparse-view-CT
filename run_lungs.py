



import torch
import cv2 as cv
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import skimage.metrics as skm
import numpy as np
from tomosipo.torch_support import (
    to_autograd,
)
from model import *
from poisson_noise import *
from itertools import combinations
import LION.CTtools.ct_utils as ct
from ts_algorithms import fbp, tv_min2d
import LION.CTtools.ct_geometry as ctgeo
from skimage.transform import rescale, resize
import skimage

device = "cuda:0"
photon_count=3000 # 
attenuation_factor=2.76 # corresponds to absorption of 50%




class Noise2inverse:
    def __init__(
        self,
        device: str = 'cuda:0',
        folds: int = 4,
    ):
        self.net_denoising = UNet(in_channels=1,out_channels=1).to(device)
        self.folds = folds
        self.device = device
        
    def forward(self,reconstruction, nr_angles):
        input_x = reconstruction
        output_denoising = (self.net_denoising(input_x.float().to(self.device)))
       # with torch.no_grad():
        output_denoising_sino = self.projection_tomosipo(output_denoising, sino = nr_angles).to(self.device)
        return output_denoising, output_denoising_sino
    
    def prepare_batch(self,sinograms):
        sinograms = sinograms.squeeze()
        Reconstructions = np.zeros((sinograms.shape[0],self.folds, sinograms.shape[-2], sinograms.shape[-2]))
        number_of_angles = sinograms.shape[-1]
        projection_indices = np.array([i for i in range(0, number_of_angles)])
        indices_four_folds = [projection_indices[i::self.folds] for i in range(self.folds)]
        theta = np.linspace(0., 180., number_of_angles, endpoint=False)
        angles_four_folds = theta[indices_four_folds]
        sinograms_four_folds = sinograms[:,:,indices_four_folds]
        sinograms_four_folds = torch.movedim(sinograms_four_folds, -2,1)  
        
        for i in range(sinograms.shape[0]):
            for j in range(self.folds):  
                Reconstructions[i,j] = self.fbp_tomosipo(torch.tensor(sinograms_four_folds[i][j].unsqueeze(0).unsqueeze(0)), translate = "N2I", angle_vector = indices_four_folds[j], folds = self.folds)
        return torch.tensor(Reconstructions), indices_four_folds
   
    def projection_tomosipo(self, img, sino, translate = False):
        if sino.dtype == int:
            angles = sino
        else:
            angles = sino.shape[-1]
        # 0.1: Make geometry:
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0],336,336), number_of_angles=angles, translate = translate
        ) 
        # 0.2: create operator:
        op = to_autograd(ct.make_operator(geo))
        sino = op((img[:,0]).to(self.device))
        sino = sino.unsqueeze(1)
        sino = torch.moveaxis(sino, -1,-2)
        return sino

        
    
    def fbp_tomosipo(self, sino, angle_vector=None, translate=False, folds = None):
        angles = sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0],336,336), number_of_angles=angles, translate = translate, angle_vector = angle_vector
        ) 
        op = ct.make_operator(geo)      
        sino = torch.moveaxis(sino, -1,-2)
        result = fbp(op,sino[:,0])
        result = result.unsqueeze(1)
        return result
            

def sobolev_norm(f,g):
     l2 = torch.nn.functional.mse_loss(f,g)
     im_size = f.shape[-2]
     ### f and g have shape (batch size, 1, N_s, N_theta)
     derivatives_s1 = torch.gradient(f, axis=(2,3))[0]
     derivatives_s2 = torch.gradient(g, axis=(2,3))[0]
     l2_grad = torch.nn.functional.mse_loss(derivatives_s1, derivatives_s2)
     #print("gradient term is: " + str(l2_grad))
     #print("l2 term is: " + str(l2))     
     sobolev = (l2**2 + l2_grad**2)/im_size**2
     return sobolev


def get_images(path, amount_of_images='all', scale_number=1):
    all_images = []
    all_image_names = os.listdir(path)
    print(len(all_image_names))
    if amount_of_images == 'all':
        for name in all_image_names:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
    else:
        temp_indexing = np.random.permutation(len(all_image_names))[:amount_of_images]
        
        images_to_take = [all_image_names[i] for i in temp_indexing]
        for name in images_to_take:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
    
    return all_images


path=("C:/Users/nadja/Documents/Sparser2Sparse_tomosipo/Sparser2Sparse_tomosipo/Sparse2Sparse_new/lungs")
files = os.listdir(path)
images=[]
for file in files:
    img = plt.imread(path+"/"+file)
    img = resize(img,(320,320,1))
    img = img-np.min(img)
    img = img/np.max(img)
    images.append(img[:,:,0])

images = np.asarray(images, dtype='float16')
#images = torch.from_numpy(images).float().to(device)
Images = np.zeros((30,456,456))
Images[:,68:-68,68:-68]=images[:30]
Images = resize(Images, (30,336,336))
images = Images[:]


def add_gaussian_noise(img, sigma):
    img = np.array(img)
    noise = np.random.normal(0, sigma, img.shape)/100
    img = img + np.max(img)*noise
    return torch.tensor(img)
    

def create_noisy_sinograms(images, angles_full, sigma):
    # 0.1: Make geometry:
    geo = ctgeo.Geometry.parallel_default_parameters(
        image_shape=images.shape, number_of_angles=angles_full
    )  # parallel beam standard CT
    # 0.2: create operator:
    op = ct.make_operator(geo)
    # 0.3: forward project:
    sino = op(torch.from_numpy(images))
    sinogram_full = add_gaussian_noise(sino, sigma)
    sinogram_full = torch.moveaxis(sinogram_full, -1, -2)
    return np.asarray(sinogram_full.unsqueeze(1))

########################### N2I ###########################
number_angles = 64
sinograms = torch.tensor(create_noisy_sinograms(images, number_angles, 0))

torch.manual_seed(0)

maxi= torch.max(sinograms)
proj = sinograms/maxi
proj *= attenuation_factor

proj_noisy = apply_noise(proj, photon_count)
proj_noisy /= attenuation_factor
proj_noisy = torch.tensor(proj_noisy)
proj_noisy= proj_noisy*maxi
sinograms = proj_noisy
dataset = torch.utils.data.TensorDataset(sinograms, torch.tensor(images))


N_epochs = 2000
learning_rate = 2e-4
###### Choose from 'MSE_image', 'MSE_data', 'Sobolev_data'
loss_variant = 'MSE_data'

newpath = r'C:/Users/nadja/Documents/Sparser2Sparse_tomosipo/Sparser2Sparse_tomosipo/Sparse2Sparse_new/Results_lungs/'+loss_variant +"_" + str(learning_rate) +"_angles_" + str(number_angles) + "_noise_" + str(photon_count)
if not os.path.exists(newpath):
    os.makedirs(newpath)

N2I = Noise2inverse()
N2I_optimizer = optim.Adam(N2I.net_denoising.parameters(), lr=learning_rate)
Data_loader = DataLoader(dataset, batch_size=8, shuffle = True)
np.savez_compressed(newpath+"/data.npz",sinograms = sinograms)


########################### Now training starts ##############
l2_list = []
all_MSEs = []
all_ssim =  []
all_psnr=[]
for epoch in range(N_epochs):
    print('Epoch number ', epoch)
    running_loss = 0
    running_L2_loss = 0
    for sinos, ims in Data_loader:
        N2I_optimizer.zero_grad()
        recos, indices = N2I.prepare_batch(sinos)
        rand_ints = np.random.permutation(N2I.folds)
        loss_indices = indices[rand_ints[N2I.folds-1]]
        # input_x corresponds to \tilde{x}_{J^C} in paper,
        #mean is taken over three images in our case)
        input_x_den = torch.mean(recos[:,rand_ints[:N2I.folds-1]], axis= 1)
        input_x_den = input_x_den.unsqueeze(1)
        # target is \tilde{x}_J, as |J| = 1 , no mean required
        target = recos[:,rand_ints[N2I.folds-1]]
        target = target.unsqueeze(1).to(N2I.device)
        # divide by 5 for faster NW convergence
        input_x_den = input_x_den/5
        output_reco, output_sino = N2I.forward(input_x_den.to(N2I.device), sinos)
        target = target/5
        output_sino*=5
        if loss_variant == 'MSE_image':
            loss = torch.nn.functional.mse_loss(output_reco.float(), target.float())
            with torch.no_grad():
                l2_loss = torch.nn.functional.mse_loss(output_reco.float().squeeze(), ims.float().to(device))
        elif loss_variant == 'MSE_data':
            loss = torch.nn.functional.mse_loss(output_sino[:,:,:,loss_indices].float(), sinos[:,:,:,loss_indices].float().to(N2I.device))
            with torch.no_grad():
                l2_loss = torch.nn.functional.mse_loss(output_reco.float().squeeze(), ims.float().to(device))
        elif loss_variant == 'Sobolev_data':
            loss = sobolev_norm(output_sino[:,:,:,loss_indices].float(), sinos[:,:,:,loss_indices].float().to(N2I.device))
            with torch.no_grad():
                l2_loss = torch.nn.functional.mse_loss(output_reco.float().squeeze(), ims.float().to(device))
        loss.backward()
        N2I_optimizer.step()
        running_loss+= loss.item()
        running_L2_loss += l2_loss.item()
    l2_list.append(running_L2_loss)
    if epoch>2:
        if l2_list[-1] < np.min(l2_list[:-1]):
            np.savez_compressed(newpath+"/best_l2.npz", full_recos= full_recos.cpu().numpy())
    if epoch %15 ==0:
        plt.plot(l2_list)
        plt.show()
        with torch.no_grad():
            print('Loss for epoch:', epoch, running_loss)
            print('L2-loss for epoch:', epoch, running_L2_loss)
            plt.subplot(221)
            plt.imshow(input_x_den[0,0].detach().cpu())
            plt.subplot(222)
            plt.imshow(output_reco[0,0].detach().cpu())
            plt.subplot(223)
            plt.imshow(output_sino[0,0].detach().cpu(), aspect = 'auto')
            plt.subplot(224)
            plt.imshow(torch.abs(output_sino[0,0].detach().cpu()-sinos[0,0].detach().cpu()), aspect = 'auto')
            plt.show()
        
    if epoch %5 == 0:
        validation_dataloader =  DataLoader(dataset, batch_size=8, shuffle = False)
        full_recos =  []
        MSEs = []
        with torch.no_grad():
            for sinos, ims in validation_dataloader:
                N2I_optimizer.zero_grad()
                ims = ims.to(N2I.device)
                recos, indices = N2I.prepare_batch(sinos)
                subsets = combinations(range(4),3)
                final_recos = torch.zeros((recos.shape[0],1, recos.shape[2],recos.shape[3])).to(N2I.device)

                for rand_ints in subsets:
                    input_x_den = torch.mean(recos[:,rand_ints], axis= 1)
                    input_x_den = input_x_den.unsqueeze(1)
                    # target is \tilde{x}_J, as |J| = 1 , no mean required
                    # divide by 5 for faster NW convergence
                    input_x_den = input_x_den/5
                    output_reco, output_sino = N2I.forward(input_x_den.to(N2I.device), sinos)
                    final_recos += output_reco*5/4
                full_recos.append(final_recos)
                err = torch.mean(torch.mean((final_recos.squeeze()-ims)**2,-1),-1)
                MSEs.append(err)
        
        full_recos = torch.cat(full_recos,0)
        MSEs = torch.cat(MSEs,0)
        all_MSEs.append(torch.mean(MSEs).cpu())
        ssim = []
        psnr = []
        for i in range(4):
            plt.imshow(full_recos[i,0].cpu())
            plt.show()
        for i in range(len(full_recos)):
            data_range = images[i].max()-images[i].min()
            ssim.append(skimage.metrics.structural_similarity(full_recos[i,0].cpu().numpy(), images[i]))
            psnr.append(skimage.metrics.peak_signal_noise_ratio(full_recos[i,0].cpu().numpy().astype("float64"), images[i], data_range = data_range))
        all_ssim.append(np.mean(ssim))
        all_psnr.append(np.mean(psnr))
        plt.subplot(121)
        plt.plot(all_ssim)
        plt.subplot(122)
        plt.plot(all_psnr)
        plt.show()
        if epoch >2:
            if all_psnr[-1] > np.max(all_psnr[:-1]):
                np.savez_compressed(newpath+"/best_psnr.npz", psnr = all_psnr[-1], full_recos = full_recos.cpu().numpy())
            if all_ssim[-1] > np.max(all_ssim[:-1]):
                np.savez_compressed(newpath+"/best_ssim.npz", ssim = all_ssim[-1], full_recos = full_recos.cpu().numpy())
        print('MSE this epoch: ', all_MSEs[-1])
        print('SSIM this epoch: ', all_ssim[-1])
        print('maxSSIM:', np.max(all_ssim))
        print('PSNR this epoch: ', all_psnr[-1])
        print('maxPSNR:', np.max(all_psnr))
        print('minMSE:', np.min(all_MSEs))
np.savez_compressed(newpath+"/metrics.npz", ssim = all_ssim, psnr = all_psnr)
np.savez_compressed(newpath+"/l2.npz", l2=l2_list)
