# Matrices in Python
# For this exercise, we shall make use of the @ operator again. Recall from the last exercise, we used this operator to take the dot product of vectors. In general the operator will combine vectors and/or matrices in the expected linear algebra way, i.e. it will be either the vector dot product, matrix multiplication, or matrix operation on a vector, depending on it's input. For example to calculate the following expressions,
# a=s⋅ta=s⋅t 
# s=Ats=At 
# M=ABM=AB ,
# One would use the code,
# a = s @ t
# s = A @ t
# M = A @ B
# (This is in contrast to the  ∗∗  operator, which performs element-wise multiplication, or multiplication by a scalar.)
# You may need to use some of the following functions:
# inv(A)
# transpose(A)
# gsBasis(A)
# These, respectively, take the inverse of a matrix, give the transpose of a matrix, and produce a matrix of orthonormal column vectors given a general matrix of column vectors - i.e. perform the Gram-Schmidt process. This exercise will require you to combine some of these functions.

import numpy as np
from numpy.linalg import norm, inv
from numpy import transpose
from readonly.bearNecessities import *

# GRADED FUNCTION
# You should edit this cell.

# In this function, you will return the transformation matrix T,
# having built it out of an orthonormal basis set E that you create from Bear's Basis
# and a transformation matrix in the mirror's coordinates TE.
def build_reflection_matrix(bearBasis) : # The parameter bearBasis is a 2×2 matrix that is passed to the function.
    # Use the gsBasis function on bearBasis to get the mirror's orthonormal basis.
    E = gsBasis(bearBasis)
    # Recall, the mirror operates by negating the last component of a vector.
    # Replace a,b,c,d with appropriate values
    TE = np.array([[1, 0],
                   [0, -1]])
    # Combine the matrices E and TE to produce your transformation matrix.
    T = E @ TE @ transpose(E) # Could be inv(E) but E orthonormal, so it is simpler to use transpose(E)
    # Finally, we return the result. There is no need to change this line.
    return T
