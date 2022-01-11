import numpy as np
import torch
import torchvision


# %matplotlib inline
import matplotlib.pyplot as plt

"""
pre-defined variables (can be changed)
n_iternation can be increased to a large number and learning_rate can be decreased to involve more detail to the image production cycle. At current
values, the program is only able to produce images that has the general facial features of intended face.
"""
batch_size = 8

n_iterations = 10

dim_z = 512

learning_rate = 0.5


def get_user_rating(image_num):
    
    print("For image number ", image_num+1, ",")
    r = input("Enter your rating (enter quit to stop):")
    if r == "quit":
      print("terminated")
      exit()
    else:
      return int(r)


def plot_images(images):
    grid = torchvision.utils.make_grid(
        generated_images.clamp(min=-1, max=1),
        nrow=9,
        scale_each=True,
        normalize=True
    )
    grid = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(16, 8))
    plt.cla()
    plt.imshow(grid)
    plt.show(block=False)

def plot_single_image(images):
    plt.figure(figsize=(2, 8))
    images = images.clamp(min=0, max=1)
    images = np.transpose(images, (1, 2, 0))
    plt.cla()
    plt.imshow(images)
    plt.show(block=False)


#load pre-trained generator
#Interchange line 71-80 and 81-89 to change the GAN Model used by the program by uncommenting one and commenting the other.


"""
use_gpu = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if use_gpu else "cpu")
model = torch.hub.load(
    "facebookresearch/pytorch_GAN_zoo",
    model="StyleGAN",
    pretrained=True,
    useGPU=use_gpu
)
"""
use_gpu = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if use_gpu else "cpu")
model = torch.hub.load(
    "facebookresearch/pytorch_GAN_zoo",
    model="PGAN",
    checkpoint="celebAHQ-512",
    pretrained=True,
    useGPU=use_gpu
)



# initialize latent variable


z = torch.zeros((dim_z,))

#Finds average of all: this function finds the average of the ratings and applies them to the noise.
def method_1(z_noised, F):
  var = np.dot(F, noise)
  sum_F = np.sum(F)
  average = var/sum_F
  new_z = z + average*learning_rate
  return new_z

#Finds the maximum rating: this function acknowledges the maximum rating inputted and only uses that to alter the noise.
def method_2(z_noised, F):
  max_indx = np.argmax(F)
  update = noise[max_indx, :]

  new_z = z + update*learning_rate

  return new_z

#This function considers only a certain number of ratings based on user input.
def method_3(z_noised, F, inp):
  idx = F[np.argsort(F)[-inp:]]
  mask = (F >= idx[0])
  F = F * mask
  sum_F = np.sum(F)
  var = np.dot(F, noise)
  sum_F = np.sum(F)
  average = var/sum_F
  new_z = z + average*learning_rate
  return new_z


#This function only considers images with ratings above a provided threshold.
def method_4(z_noised, F, threshold):
  mask = (F >= threshold)
  F = F * mask
  sum_F = np.sum(F)
  var = np.dot(F, noise)
  sum_F = np.sum(F)
  average = var/sum_F
  is_all_below = np.all((F <= threshold))
  if is_all_below:
    new_z = z
  else:
    new_z = z + average*learning_rate
  return new_z


for iteration in range(1, n_iterations + 1):

    print("=" * 40)
    print(" Iteration {}/{}".format(iteration, n_iterations))
    print("=" * 40)

   
    noise = torch.randn((batch_size, dim_z))

    with torch.no_grad():

       
      z_noised = z.repeat(batch_size, 1) + noise      
      all_z = torch.cat([z.view(1, -1), z_noised])
      generated_images = model.test(all_z)
      generated_image = model.test(z_noised)
    
    print("  ====seed image====")
    plot_single_image(generated_images[0, :, :, :])
    print("  ==================")
    plot_images(generated_image)
      

    # a list of all ratings for this iteration
    F = []

    # for each image, request user feedback and add it to F
    for image_num in range(0, batch_size):
        F_n = get_user_rating(image_num)
        F.append(F_n)

    # convert F from python list to NumPy array
    F = np.array(F, dtype=np.float64)

    noise = noise.cpu().numpy().astype(np.float64)
    z_noised = z_noised.cpu().numpy()
    z = z.cpu().numpy()
    """
    Uncomment the line with the method you wish to implement and comment all other lines (amongst line 187-190)
    Description of the methods can be found as comment above their code (see line 98 and below)

    For method 3, replace 3 with n, n being the top n images you want to consider
    For method 4, replace 5 with the threshold
    """
    z = method_1(z_noised, F) 
    #z = method_2(z_noised, F)
    #z = method_3(z_noised, F,3)
    #z = method_4(z_noised, F, 5)
    z = torch.tensor(z).to(device).float()
    z_noised = torch.tensor(z_noised).to(device).float()
