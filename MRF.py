#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:58:39 2017

@author: yihuicui
"""

import pylab as pl
import numpy as np
import argparse


def MRF(I, J, eta, zeta, beta):
    """
    Perform Inference to determine the label of every pixel.
    1. Go through the image in random order.
    2. Evaluate the energy for every pixel being being set to either 1 or -1.
    3. Assign the label resulting in the least energy to the pixel in question.

    Inputs:
        I: The original noisy image.
        J: The denoised image from the previous iteration.
        eta: Weight of the observed-latent pairwise ptential.
        zeta: Weight of the latent-latent pairwise ptential.
        beta: Weight of unary term.
    Output:
        denoised_image: The denoised image after one MRF iteration.
    """
    denoised_image=J.copy()
    x_random=np.random.permutation(I.shape[0]-1)
    y_random=np.random.permutation(I.shape[1]-1)
    for i in range(x_random.shape[0]):
        for j in range(y_random.shape[0]):
            x=x_random[i]
            y=y_random[j]
            energy_1=energy_evaluation(I,denoised_image,x,y,1,eta,zeta,beta)
            energy_2=energy_evaluation(I, denoised_image, x, y, -1, eta, zeta, beta)
            if energy_1<energy_2:
                denoised_image[x][y]=1
            else:
                denoised_image[x][y]=-1
    return denoised_image


def energy_evaluation(I, J, pixel_x_coordinate, pixel_y_coordinate,
    pixel_value, eta, zeta, beta):
    """
    Evaluate the energy of the image of a particular pixel set to either 1or-1.
    1. Set pixel(pixel_x_coordinate,pixel_y_coordinate) to pixel_value
    2. Compute the unary, and pairwise potentials.
    3. Compute the energy

    Inputs:
        I: The original noisy image.
        J: The denoised image from the previous iteration.
        pixel_x_coordinate: the x coordinate of the pixel to be evaluated.
        pixel_y_coordinate: the y coordinate of the pixel to be evaluated.
        pixel_value: could be 1 or -1.
        eta: Weight of the observed-latent pairwise ptential.
        zeta: Weight of the latent-latent pairwise ptential.
        beta: Weight of unary term.
    Output:
        energy: Energy value.
         """
    energy=0
    def latent_latent(pixel_x_coordinate,pixel_y_coordinate):
        latent_latent_energy=0
        if pixel_x_coordinate>0:
            latent_latent_energy+=pixel_value*J[pixel_x_coordinate - 1, pixel_y_coordinate]
        if pixel_x_coordinate+1<J.shape[0]:
            latent_latent_energy+=pixel_value*J[pixel_x_coordinate + 1, pixel_y_coordinate]
        if pixel_y_coordinate>0:
            latent_latent_energy+=pixel_value * J[pixel_x_coordinate, pixel_y_coordinate - 1]
        if pixel_y_coordinate+1 < J.shape[0]:
            latent_latent_energy+= pixel_value * J[pixel_x_coordinate, pixel_y_coordinate + 1]
        return latent_latent_energy
    def observed_latent(pixel_x_coordinate,pixel_y_coordinate):
        return pixel_value* I[pixel_x_coordinate, pixel_y_coordinate]
    latent_latent_energy=latent_latent(pixel_x_coordinate,pixel_y_coordinate)
    observed=observed_latent(pixel_x_coordinate,pixel_y_coordinate)
    energy=-zeta*latent_latent_energy-eta*observed-beta*pixel_value
    return energy


def greedy_search(noisy_image, eta, zeta, beta, conv_margin):
    """
    While convergence is not achieved (this verified by calling
    the function 'not_converged'),
    1. iteratively call the MRF function to perform inference.
    2. If the number of iterations is above 10, stop and return
    the image that you have at the 10th iteration.

    Inputs:
        noisy_image: the noisy image.
        eta: Weight of the pairwise observed-latent potential.
        zeta: Weight of the pairwise latent-latent potential.
        beta: Weight of unary term.
        conv_margin: Convergence margin
    Output:
        denoised_image: The denoised image.
    """

    # Noisy Image.
    I_new = noisy_image.copy()
    I_old=noisy_image.copy()
    i=0
    while (not_converged(I_old,I_new,i,conv_margin) and i<10):
        I_old=I_new
        I_new=MRF(noisy_image,I_old,eta,zeta,beta)
        i+=1
    denoised_image=I_new
    return denoised_image
def not_converged(image_old, image_new, itr, conv_margin):
    """
    Check for convergence. Convergence is achieved if the denoised image
    does not change between two consequtive iterations by a certain
    margin 'conv_margin'.
    1. Compute the percentage of pixels that changed between two
     consecutive iterations.
    2. Convergence is achieved if the computed percentage is below
    the convergence margin.

    Inputs:
        image_old: Denoised image from the previous iteration.
        image_new: Denoised image from the current iteration.
        itr: The number of iteration.
        conv_margin: Convergence margin.
    Output:
        has_converged: a boolean being true in case of convergence
    """
    if itr==0:
        return True
    sum_value=np.sum(image_old!=image_new)
    mean=sum_value/(image_old.shape[0]*image_old.shape[1])
    if mean<conv_margin:
        return False
    else:
        return True

def load_image(input_file_path, binarization_threshold):
    """
    Load image and binarize it by:
    0. Read the image.
    1. Consider the first channel in the image
    2. Binarize the pixel values to {-1,1} by setting the values
    below the binarization_threshold to -1 and above to 1.
    Inputs:
        input_file_path.
        binarization_threshold.
    Output:
        I: binarized image.
    """
    image=pl.imread(input_file_path)
    image=image[:,:,0].astype(int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j]>binarization_threshold:
                image[i][j]=1
            else:
                image[i][j]=-1
    return image


def inject_noise(image):
    """
    Inject noise by flipping the value of some randomly chosen pixels.
    1. Generate a matrix of probabilities of every pixel
    to keep its original value .
    2. Flip the pixels if its corresponding probability in
    the matrix is below 0.1.

    Input:
        image: original image
    Output:
        noisy_image: Noisy image
    """
    noisy_image = image.copy()
    index=np.random.choice(10,image.shape)
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            if index[i][j]<1:
                noisy_image[i][j]=noisy_image[i][j]*(-1)


    return noisy_image



def f_reconstruction_error(original_image, reconstructed_image):
    """
    Compute the reconstruction error (L2 loss)
    inputs:
        original_image.
        reconstructed_image.
    output:
        reconstruction_error: MSE of reconstruction.
    """
    reconstruction_error = np.mean((original_image - reconstructed_image) ** 2)
    return reconstruction_error

def plot_image(image, title, path):
    pl.figure()
    pl.imshow(image)
    pl.title(title)
    pl.savefig(path)

def parse_arguments(parser):
    """
    Parse arguments from the command line
    Inputs:
        parser object
    Output:
        Parsed arguments
    """
    parser.add_argument('--input_file_path',
        type=str,
        default="img/seven.png",
        metavar='<input_file_path>',
        help='Path to the input file.')

    parser.add_argument('--weight_pairwise_observed_unobserved',
        type=float,
        default=2,
        metavar= '<weight_pairwise_observed_unobserved>',
        help='Weight of observed-unobserved pairwise potential.')

    parser.add_argument('--weight_pairwise_unobserved_unobserved',
        type=float,
        default=1.5,
        metavar='<weight_pairwise_unobserved_unobserved>',
        help='Weight of unobserved-unobserved pairwise potential.')

    parser.add_argument('--weight_unary',
        type=float,
        default=0.1,
        metavar='<weight_unary>',
        help='Weight of the unary term.')

    parser.add_argument('--convergence_margin',
        type=float,
        default=0.999,
        metavar='<convergence_margin>',
        help='Convergence margin.')

    parser.add_argument('--binarization_threshold',
        type=float,\
        default=0.05, \
        metavar='<convergence_margin>',
        help='Perc of different btw the images between two iter.')

    args = parser.parse_args()
    return args
def main():
    # Read the input arguments
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    # Parse the MRF hyperparameters
    eta = args.weight_pairwise_observed_unobserved
    zeta = args.weight_pairwise_unobserved_unobserved
    beta = args.weight_unary

    # Parse the convergence margin
    conv_margin = args.convergence_margin

    # Parse the input file path
    input_file_path = args.input_file_path

    # Load the image.
    I = load_image(input_file_path, args.binarization_threshold)

    # Create a noisy version of the image.
    J = inject_noise(I)
    plot_image(J,'Noisy Image','img/noisy_Image')
    # Call the greedy search function to perform MRF inference
    newJ = greedy_search(J, eta, zeta, beta, conv_margin)

    # Plot the Original Image
    plot_image(I, 'Original Image', 'img/Original_Image')

    # Plot the Denoised Image
    plot_image(newJ, 'Denoised version', 'img/Denoised_Image')

    # Compute the reconstruction error
    reconstruction_error = f_reconstruction_error(I, newJ)
    print('Reconstruction Error: ', reconstruction_error)


if __name__ == "__main__":
    main()
