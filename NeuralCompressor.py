import numpy as np
from time import time

from utils import matrix_mul, matrix_dif, transpose, \
    matrix_pow, matrix_single_mul, matrix_sum, is_pre_trained


class NeuralCompressor:
    
    def __init__(self, p, a, err, img_arr):
        self.p = p
        self.a = a
        self.err = err

        if img_arr is not None:
            self.img_arr = img_arr

            self.N = len(self.img_arr[0])
            self.L = self.img_arr.shape[0] * self.img_arr.shape[1]

            # Z = (N*L)/((N+L)*p+2)
            self.Z = (self.N * self.L) / ((self.N + self.L) * self.p + 2)


    def compress_img(
            self, 
            pre_trained_neurons=False, 
            pre_trained_neurons_name='mono', 
            compressed_file_name='compressed/mono.npy'
        ):
        """
        Train neural network for compress img or load trained network from file (pre-trained).

        :param pre_trained_neurons(default=False): Load pre trained neurons from file (pre-trained folder) or train new neurons.  
        :param img_name(default='mono'): Image name for classify neurons.
        :param compressed_file_name(default='compressed/mono.npy'): Save result of compression to file.
        :return: Matrix of compressed image.
        """

        if not pre_trained_neurons: 
            w1 = np.random.rand(self.N, self.p) * 2 - 1
            w2 = np.array(w1.transpose())
        else:
            with open(f'pre-trained/{pre_trained_neurons_name}1.npy', 'rb') as w1_file:
                w1 = np.load(w1_file)
            with open(f'pre-trained/{pre_trained_neurons_name}2.npy', 'rb') as w2_file:
                w2 = np.load(w2_file)

        step = 0
        timer = 0

        while True:
            error_common = 0
            step += 1

            timer_start = time()

            for row in self.img_arr:
                row = row.reshape(1, -1)
                # Y(i) = X(i)*W
                y = np.matmul(row, w1)
                # X'(i) = Y(i)*W'
                x = np.matmul(y, w2)
                # ∆X(i) = X'(i) – X(i) 
                dx = x - row
                dx = dx.reshape(1, -1)

                # W2(t + 1) = W(t) - α*[Y(i) T * ∆X(i)]
                w2 -= self.a * np.matmul(y.transpose(), dx)
                # W1(t + 1) = W(t) - α*[X(i)] T *∆X(i)*[W'(t)]^T
                w1 -= self.a * np.matmul(np.matmul(row.transpose(), dx), w2.transpose())
                
                # Е(q) = ∑∆X(q)i *∆X(q)i
                error = (dx * dx).sum()
                error_common += error

                with open(f'pre-trained/{pre_trained_neurons_name}1.npy', 'wb') as w1_file:
                    np.save(w1_file, w1)
                with open(f'pre-trained/{pre_trained_neurons_name}2.npy', 'wb') as w2_file:
                    np.save(w2_file, w2)

            timer_finish = time()
            timer += timer_finish - timer_start
            print(f'Iteration: {step}, Time: {timer}s., Error: {error_common}')

            if error_common < self.err:
                compress_matrix = []
                for row in self.img_arr:
                    compress_matrix.append(row)

                return compress_matrix


    def decompress_img(
            self, 
            pre_trained_neurons_name='mono', 
            compressed_file_name='compressed/mono.npy'
        ):
        """
        Decompress compressed file with pre-trined neural network.

        :param pre_trained_neurons(default=False): Load pre trained neurons from file (pre-trained folder) or train new neurons.  
        :param pre_trained_neuron_name(default='mono'): Image name for find classifed neurons.
        :param compressed_file_name(default='compressed/mono.npy'): Load result of compression.
        :return: Matrix of compressed image.
        """

        if is_pre_trained(pre_trained_neurons_name):
            with open(f'pre-trained/{pre_trained_neurons_name}1.npy', 'rb') as w1_file:
                w1 = np.load(w1_file)
            with open(f'pre-trained/{pre_trained_neurons_name}2.npy', 'rb') as w2_file:
                w2 = np.load(w2_file)
        else:
            return

        with open(compressed_file_name, 'rb') as comp_file:
            try:
                compressed_img_arr = np.load(comp_file)
            except Exception as err:
                print(err)
                return

        decompressed_img_arr = []
        for row in compressed_img_arr:
            decompressed_img_arr.append(
                np.matmul(
                    np.matmul(row, w1),
                    w2
                )
            )

        decompressed_img_arr = np.array(decompressed_img_arr)

        return decompressed_img_arr


    def compress_img_no_np(self, pre_trained_neurons=False, img_name='mono'):
        """
        Train neural network for compress img or load trained network from file (pre-trained).
        :param pre_trained_neurons(default=False): Load pre trained neurons from file (pre-trained folder) or train new neurons.  
        :param img_name(default='mono'): Image name for classify neurons.
        :return: Matrix of compressed image.
        """
        if not pre_trained_neurons: 
            # W'(t + 1) = W'(t) – α'*[Y(i)] T *∆X(i)
            w1 = np.random.rand(self.N, self.p) * 2 - 1
            w2 = np.array(transpose(w1.tolist()))
        else:
            with open(f'pre-trained/{img_name}1.npy', 'rb') as w1_file:
                w1 = np.load(w1_file)
            with open(f'pre-trained/{img_name}2.npy', 'rb') as w2_file:
                w2 = np.load(w2_file)
        step = 0
        timer = 0

        while True:
            error_common = 0
            step += 1

            timer_start = time()
            
            for row in range(len(self.img_arr)):
                omg_img = self.img_arr[row].reshape(1, -1)
                w1 = list(w1)
                w2 = list(w2)

                # Y(i) = X(i)*W
                y = matrix_mul(omg_img, w1)
                # X'(i) = Y(i)*W'
                x = matrix_mul(y, w2)
                # ∆X(i) = X'(i) – X(i) 
                dx = np.array(matrix_dif(x, omg_img))
                dy = dx.reshape(1, -1)

                # W(t + 1) = W(t) –  α*[X(i)] T *∆X(i)*[W'(t)]^T
                w2 = matrix_dif(w2, matrix_single_mul(matrix_mul(transpose(y), dx.tolist()), self.a))
                w1 = matrix_dif(w1, matrix_single_mul(matrix_mul(matrix_mul(transpose(omg_img), dx.tolist()), transpose(w2)), self.a))

                # Е(q) = ∑∆X(q)i *∆X(q)i
                error = (dx * dx).sum()
                error_common += error

                print(row)

            timer_finish = time()
            timer = timer_finish - timer_start

            print(f'Iteration: {step}, Time: {timer}s., Error: {error_common}')

            if error_common < self.err:
                compressed_img = []
                new_row = []
                for row in self.img_arr:
                    new_row.append(
                        matrix_mul(
                            matrix_mul(row, w1),
                            w2
                        )
                    )

                compressed_img.append(new_row)
                compressed_img = compressed_img[0]
                compressed_img = np.array(compressed_img)

                return compressed_img
