import click
import os

from NeuralCompressor import NeuralCompressor
from ImgProcces import ImgProcces
from utils import load_pre_trined_neurons


@click.group()
def train():
    pass


@train.command()
@click.option(
    '--path',
    type=str,
    default='images/mono.jpg',
    help='Path to image.'
)
@click.option(
    '-e',
    type=int,
    default=2000,
    help='Standard error.'
)
def train_compress_neuron(path, e):
    """
    Train neuron for fast compress image.
    """
    img_name = path.split('/')[-1]
    img_name = img_name.split('.')[0]

    img_proccess = ImgProcces()
    img_arr = img_proccess.img_to_array(path)

    pre_trained_path = '/'.join(__file__.split('/')[:-2])
    for dirname, _, filename in os.walk(f'{pre_trained_path}/pre-trained'):
        pre_trained_files = filename

    if f'{img_name}1.npy' in pre_trained_files and f'{img_name}2.npy' in pre_trained_files:
        yn_load_pre_trained = load_pre_trined_neurons(img_name)
    else:
        yn_load_pre_trained = False

    compress_matrix = NeuralCompressor(
        p=32,
        err=e,
        a=0.0001,
        img_arr=img_arr
    ).compress_img(
        img_name=img_name, 
        pre_trained_neurons=yn_load_pre_trained
        )


@train.command()
@click.option(
    '--path',
    type=str,
    default='images/mono.jpg',
    help='Path to image.'
)
@click.option(
    '--result-path',
    type=str,
    default='result/out.jpg',
    help='Path to compress result.'
)
def compress_img(path, result_path):
    """
    Compress image.
    """
    path = str(path)
    result_path = str(result_path)

    img_name = path.split('/')[-1]
    img_name = img_name.split('.')[0]

    img_proccess = ImgProcces()
    img_arr = img_proccess.img_to_array(path)

    pre_trained_path = '/'.join(__file__.split('/')[:-2])
    for dirname, _, filename in os.walk(f'{pre_trained_path}/pre-trained'):
        pre_trained_files = filename

    if f'{img_name}1.npy' in pre_trained_files and f'{img_name}2.npy' in pre_trained_files:
        compress_matrix = NeuralCompressor(
            p=32,
            err=1000000,
            a=0.0001,
            img_arr=img_arr
        ).compress_img(
            img_name=img_name, 
            pre_trained_neurons=True
            )

        img_proccess.array_to_img(compress_matrix, result_path)
    
    else:
        print('Neuron is not trained. Please before compress train neuron: \
            "./compressor.py train-compress-neuron --path [Path to image] -e [Standart error]"')


@train.command()
@click.option(
    '--path',
    type=str,
    default='images/mono.jpg',
    help='Path to image.'
)
@click.option(
    '--result-path',
    type=str,
    default='result/out_mono.jpg',
    help='Path to compress result.'
)
def compress_monochrome_img(path, result_path):
    """
    Compress image to monochrome mode (take less space).
    """
    path = str(path)
    result_path = str(result_path)

    img_name = path.split('/')[-1]
    img_name = img_name.split('.')[0]

    img_proccess = ImgProcces()
    img_arr = img_proccess.img_to_array(path)

    pre_trained_path = '/'.join(__file__.split('/')[:-2])
    for dirname, _, filename in os.walk(f'{pre_trained_path}/pre-trained'):
        pre_trained_files = filename

    if f'{img_name}1.npy' in pre_trained_files and f'{img_name}2.npy' in pre_trained_files:
        compress_matrix = NeuralCompressor(
            p=32,
            err=1000000,
            a=0.0001,
            img_arr=img_arr
        ).compress_img(
            img_name=img_name, 
            pre_trained_neurons=True
            )

        result_path = f'{pre_trained_path}/{result_path}'
        img_proccess.array_to_compressed_img(compress_matrix, result_path)
    
    else:
        print('Neuron is not trained. Please before compress train neuron: \
            "./compressor.py train-compress-neuron --path [Path to image] -e [Standart error]"')


if __name__ == '__main__':
    train()
