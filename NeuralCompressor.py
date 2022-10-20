import numpy as np
from time import time


class NeuralCompressor:
    
    def __init__(self, p, a, err, img_arr):
        self.p = p
        self.a = a
        self.err = err
        self.img_arr = img_arr

        self.N = len(self.img_arr[0])
        self.L = self.img_arr.shape[0] * self.img_arr.shape[1]
        self.Z = (self.N * self.L) / ((self.N + self.L) * self.p + 2)


    @classmethod
    def __matrix_mul(cls, matrix1, matrix2):
        """
        
        """
        result = [[0 for cols in matrix2[0]] for row in matrix1]

        for num1 in range(len(matrix1)):
            for num2 in range(len(matrix2[0])):
                for num_result in range(len(matrix2)):
                    result[num1][num2] += matrix1[num1][num_result] * matrix2[num_result][num2]
        return result


    @staticmethod
    def __matrix_single_mul(matrix, n):
        """
        
        """
        result = [[number * n for number in line] for line in matrix]
        return result


    @staticmethod
    def __matrix_pow(matrix, n):
        """
        
        """
        result = [[number ** n for number in line] for line in matrix]
        return result


    @staticmethod
    def __matrix_sum(matrix1, matrix2):
        """
        
        """
        result = [[matrix1[line][row] + matrix2[line][row] for row in range(len(matrix1[line]))] for line in range(len(matrix1))]
        return result

    
    @staticmethod
    def __matrix_dif(matrix1, matrix2):
        """
        
        """
        result = [[matrix1[line][row] - matrix2[line][row] for row in range(len(matrix1[line]))] for line in range(len(matrix1))]
        return result


    @staticmethod
    def __transpose(matrix):
        """
        
        """
        result = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
        return result


    def compress_img(self, debug=False, pre_trained_neurons=False):
        """
        
        """
        if not pre_trained_neurons: 
            w1 = np.random.rand(self.N, self.p) * 2 - 1
            w2 = np.array(self.__transpose(w1.tolist()))
        else:
            with open('pre-trained/w1.npy', 'rb') as w1_file:
                w1 = np.load(w1_file)
            with open('pre-trained/w2.npy', 'rb') as w2_file:
                w2 = np.load(w2_file)
        step = 0
        timer = 0

        while True:
            error_common = 0
            step += 1

            timer_start = time()

            for num, row in enumerate(self.img_arr):
                row = row.reshape(1, -1)
                y = np.matmul(row, w1)
                x = np.matmul(y, w2)
                dx = x - row
                dx = dx.reshape(1, -1)

                w2 -= self.a * np.matmul(y.transpose(), dx)
                w1 -= self.a * np.matmul(np.matmul(row.transpose(), dx), w2.transpose())
                
                error = (dx * dx).sum()
                error_common += error

                with open('pre-trained/w1.npy', 'wb') as w1_file:
                    np.save(w1_file, w1)
                with open('pre-trained/w2.npy', 'wb') as w2_file:
                    np.save(w2_file, w2)

            timer_finish = time()
            timer += timer_finish - timer_start
            print(f'Iteration: {step}, Time: {timer}s, Error: {error_common}')

            if error_common < self.err:
                compressed_img = []
                new_row = []
                for row in self.img_arr:
                    new_row.append(
                        np.matmul(
                            np.matmul(row, w1),
                            w2
                        )
                    )
                compressed_img.append(new_row)

                compressed_img = compressed_img[0]
                compressed_img = np.array(compressed_img)
                return compressed_img


    def compress_img_no_npy(self, debug=False, pre_trained_neurons=True):
        """
        
        """
        if not pre_trained_neurons: 
            w1 = np.random.rand(self.N, self.p) * 2 - 1
            w2 = np.array(self.__transpose(w1.tolist()))
        else:
            with open('w1.npy', 'rb') as w1_file:
                w1 = np.load(w1_file)
            with open('w2.npy', 'rb') as w2_file:
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

                y = self.__matrix_mul(omg_img, w1)
                x = self.__matrix_mul(y, w2)
                dx = np.array(self.__matrix_dif(x, omg_img))
                dy = dx.reshape(1, -1)

                w2 = self.__matrix_dif(w2, self.__matrix_single_mul(self.__matrix_mul(self.__transpose(y), dx.tolist()), self.a))
                w1 = self.__matrix_dif(w1, self.__matrix_single_mul(self.__matrix_mul(self.__matrix_mul(self.__transpose(omg_img), dx.tolist()), self.__transpose(w2)), self.a))

                error = (dx * dx).sum()
                error_common += error

                print(row)

            timer_finish = time()
            timer = timer_finish - timer_start

            print(f'Iteration: {step}, Time: {timer}s, Error: {error_common}')

            if error_common < self.err:
                compressed_img = []
                new_row = []
                for row in self.img_arr:
                    new_row.append(
                        self.__matrix_mul(
                            self.__matrix_mul(row, w1),
                            w2
                        )
                    )
                compressed_img.append(new_row)

                compressed_img = compressed_img[0]
                compressed_img = np.array(compressed_img)
                return compressed_img
