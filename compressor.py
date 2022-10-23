import click

from NeuralCompressor import NeuralCompressor
from ImgProcces import ImgProcces
from utils import load_pre_trined_neurons, get_filename, \
    is_pre_trained, save_compress_image


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
def train_compress_neuron(path='images/mono.jpg', e=2000):
    """
    Train neuron for fast compress image.
    """
    
    img_name = get_filename(path)

    img_proccess = ImgProcces()
    img_arr = img_proccess.img_to_array(path)

    if is_pre_trained(img_name):
        yn_load_pre_trained = load_pre_trined_neurons(img_name)
    else:
        yn_load_pre_trained = False

    compress_matrix = NeuralCompressor(
        p=32,
        err=e,
        a=0.0001,
        img_arr=img_arr
    ).compress_img(
        pre_trained_neurons_name=img_name, 
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
    default='result/mono.npy',
    help='Path to compress result.'
)
def compress_img(path='images/mono.jpg', result_path='compressed/mono.npy'):
    """
    Compress image if neuron is pretrained.
    """

    img_name = get_filename(path)

    img_proccess = ImgProcces()
    img_arr = img_proccess.img_to_array(path)

    if is_pre_trained(img_name):
        compress_matrix = NeuralCompressor(
            p=32,
            err=1000000,
            a=0.0001,
            img_arr=img_arr
        ).compress_img(
            pre_trained_neurons_name=img_name, 
            pre_trained_neurons=True
            )

        save_compress_image(result_path, compress_matrix)

    else:
        print('Neuron is not trained. Please before compress train neuron: \
            "./compressor.py train-compress-neuron --path [Path to image] -e [Standart error]"')


@train.command()
@click.option(
    '--path',
    type=str,
    default='compressed/mono.npy',
    help='Path to compressed image.'
)
@click.option(
    '--result-path',
    type=str,
    default='result/mono.jpg',
    help='Path to decompressed result.'
)
def decompress_img(path='compressed/mono.npy', result_path='result/mono.jpg'):
    """
    Decompress image if neuron is pretrained.
    """

    img_name = get_filename(path)

    img_proccess = ImgProcces()
    img_arr = img_proccess.img_to_array(f'images/{img_name}.jpg')

    if is_pre_trained(img_name):
        decompress_matrix = NeuralCompressor(
            p=32,
            err=1000000,
            a=0.0001,
            img_arr=img_arr
        ).decompress_img(
            pre_trained_neurons_name=img_name, 
            compressed_file_name=path
            )

        img_proccess.array_to_img(decompress_matrix, result_path)

    else:
        print('Neuron is not trained. Please before compress train neuron: \
            "./compressor.py train-compress-neuron --path [Path to image] -e [Standart error]"')


if __name__ == '__main__':
    train()
