# paths
dir_backgrounds = "./backgrounds/treebark_cropped15/" # path to background images
mask_name = './mask_triangle.tiff' # path to mask
dir_output = './output/' # path to output

# image variables
sample_per_im = 32 # number of background samples per image
imsize = 256 # crop size from the background image
targetsize = [int(imsize/8),int(imsize/4)] # size of the target
resize_factor = 1 # resize background image before cropping
ts = 10000 # number of training steps
bs = 32 # batch size
n_gpu = 2 # number of GPUs to use (in its current form, the script requires 2 GPUs with ~11 GB memory to run)

# discriminator variables
depth_DM = 64
lr_DM = 0.0002 # learning rate of disciminator network
drop_DM = 0.5 # dropout rate in discriminator network
depth_AM = 4
# adversarial variables
lr_AM = 0.0001 # learning rate of adversarial network
drop_AM = 0.5 # dropout rate in adversarial network
rand_input = 100 # number of random numbers for input

# output image variables
si = 500 # save interval
n_plot_samples = 16 # number of image samples to print
save_name_image = 'camo1_' # prefix for images with targets and backgrounds
save_name_target = 'target1_' # prefix for images of targets