from email.policy import default
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
def compress_img(path):
    path = str(path)
    if path.split('.')[-1] != '.jpg':
        path = path[:-4] + '.jpg'
        print(path)

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
            err=100000,
            a=0.0001,
            img_arr=img_arr
        ).compress_img(
            img_name=img_name, 
            pre_trained_neurons=True
            )


if __name__ == '__main__':
    train()
