from NeuralCompressor import NeuralCompressor
from ImgProcces import ImgProcces


def main():
    img_procces = ImgProcces()
    img_arr = img_procces.img_to_array(img_path='images/mono.jpg')

    compress_matrix = NeuralCompressor(
        p=32,
        err=2000,
        a=0.0001,
        img_arr=img_arr
    ).compress_img(
        img_name='mono',
    )

    img_procces.array_to_img(compress_matrix)


if __name__ == '__main__':
    main()
