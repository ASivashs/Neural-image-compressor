# Run this file to see example

from NeuralCompressor import NeuralCompressor
from ImgProcces import ImgProcces
from utils import save_compress_image


def main():
    img_procces = ImgProcces()
    # Convert image to splitted array
    img_arr = img_procces.img_to_array(img_path='images/mono.jpg')

    # Init neural class
    neural_compressor = NeuralCompressor(
        p=32,
        err=20000,
        a=0.0001,
        img_arr=img_arr
    )
    # Train compress neuron and compress image
    compress_matrix = neural_compressor.compress_img(
        pre_trained_neurons=False, 
        pre_trained_neurons_name='mono', 
        compressed_file_name='compressed/mono.npy'
    )
    save_compress_image('compressed/mono.npy', compress_matrix)
    # Decompress image
    decompress_matrix = neural_compressor.decompress_img(
        pre_trained_neurons_name='mono',
        compressed_file_name='compressed/mono.npy'
    ) 

    # Convert decompress matrix to image
    img_procces.array_to_img(decompress_matrix)


if __name__ == '__main__':
    main()
