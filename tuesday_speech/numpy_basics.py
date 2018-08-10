import numpy as np
import matplotlib.pyplot as plt

# Creating an array
a1 = np.empty((5, 3))        # content can be anything initially
a2 = np.zeros((5, 3))        # full of zeros
a4 = np.random.rand(5, 3)    # filled with random values between 0 and 1
a3 = np.zeros((5, 3, 4, 2))    # 4-D array

# Array size
n_rows = a2.shape[0]            # Number of rows
n_columns = a2.shape[1]         # Number of columns
n_rows, n_columns = a2.shape    # Both at the same time
n_elements = a2.size            # Number of elements (rows*columns)

# Obtaining array elements, rows, and columns:
element = a2[1, 1]          # Value from location (1, 1)
second_row = a2[1, :]
third_column = a2[:, 2]

# Creating linearly spaced sequences of numbers:
b1 = range(5)               # 0,1,2,3,4 in an iterable
b2 = list(range(1, 5))      # A list containing 1,2,3,4
b3 = np.arange(5)           # A Numpy array containing 0,1,2,3,4
b4 = np.linspace(1, 5, 7)   # Create 7 equally spaced numbers between 1 and 5

# Iterating over every row of an array
for row in range(n_rows):
    a2[row, :] = row

# Transposing
a2t = a2.T

# Reshaping
a2reshaped = np.reshape(a2, (3, 5))
a2reshaped_fortran_order = np.reshape(a2, (3, 5), order='F')
a2flat = a2.flatten()       # 1-D array

# Numpy array functions (e.g. mean)
m = np.mean(a2)
m_row = np.mean(a2, axis=0)
m_column = np.mean(a2, axis=1)
# Similarly for many other functions  (sum, fft, std, min, max, ...)

# Matrix operations
matrix_product = np.matmul(a2, a2.T)   # Matrix multiplication
elementwise_product = a2 * a2   # Element-wise multiplication
matrix_sum = a2 + a2
mean_subtracted = a2 - m                        # Subtract mean element from every element
mean_subtracted_col = a2 - m_column[:, None]    # Subtract mean of columns from every column
mean_subtracted_row = a2 - m_row

# Saving to file and loading from file
np.save('filename.npy', a2)
a2fromfile = np.load('filename.npy')

# Matplotlib
plt.plot(a2t)
plt.show()
plt.figure()
plt.plot(np.random.rand(50))
plt.show()


# EXERCISE task:
# Create two large square matrices (smaller than 3000x3000) filled with random values.
# Then find the diagonal of the matrix product of the two matrices.
# The diagonal elements of matrix product can be obtained in many ways; time different approaches.
# TODO
