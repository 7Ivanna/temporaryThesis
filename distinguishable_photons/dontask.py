import numpy as np

def random_unitary_matrix(n):
    random_matrix = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    q, r = np.linalg.qr(random_matrix)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    q = q * ph
    return q


def u_picker(matrix, state_kl, state_lj):
    elements = {}
    n = matrix.shape[0]
    for i, j in zip(state_kl, state_lj):
        elements[f"U_{i}{j}"] = matrix[i-1, j-1]  
    
    return elements

n = 4  # n x n matrix
unitary_matrix = random_unitary_matrix(n)

state_kl = (1, 1)  # input state (i indices, written in my math as k and l )
state_lj = (1, 1)  # compared state (j indices, written in my math as i and j )

# The ones we will multiply, in order from 
selected_elements = u_picker(unitary_matrix, state_kl, state_lj)

print("Unitary Matrix:")
print(unitary_matrix)
print("\nSelected Elements:")
for key, value in selected_elements.items():
    print(f"{key}: {value}")

#%%
import numpy as np

def unitaryMZI(n, theta, phi_row):
    # Define the 2x2 matrix using the given theta
    t_matrix = np.array([[np.sin(theta), np.cos(theta)], 
                         [np.cos(theta), -np.sin(theta)]], dtype="complex")
    
    # Define the column vector
    t_column = np.array([[1],[1]], dtype="complex")  # You may replace 'a' and 'b' with actual values
    print(t_matrix)
    print(t_column)
    # Perform the matrix-vector multiplication
    multiply = np.matmul(t_matrix, t_column)

    print(multiply)
    return multiply

# Example usage
unitaryMZI(1, np.pi, 0)
n = 4  # Size of the matrix n x n 

# Define custom phase shifts for each row (one theta and phi for each row)
theta_row = np.array([0.1, 0.2, 0.3,0.4])  # Phase shifts theta for rows
phi_row = np.array([0.4, 0.5, 0.6,0.5])    # Phase shifts phi for rows

# See how Jacob does this step above, you can probably do the same thing for this 

matrix = unitaryMZI(n, theta_row, phi_row)

print(matrix)
def full_u_picker(matrix):
    """
    Select all elements from a unitary matrix in the same way as in the u_picker function.
    The function returns a dictionary where each element is of the form U_kl_ij.
    
    Args:
        matrix (np.ndarray): The unitary matrix to pick elements from.
    
    Returns:
        dict: A dictionary with the selected elements, keyed by their matrix positions.
    """
    n = matrix.shape[0]
    elements = {}
    
    # Iterate over all possible (k, l) and (i, j) combinations
    for k in range(1, n+1):
        for l in range(1, n+1):
            elements[f"U_{k}{l}"] = matrix[k-1, l-1]
    
    return elements

def build_u_matrix(matrix):
    """
    Build a full matrix where each element is selected using the u_picker method, 
    for a complete n x n matrix of unitary elements.
    
    Args:
        matrix (np.ndarray): The unitary matrix.
        
    Returns:
        np.ndarray: The reconstructed matrix based on the selection method.
    """
    n = matrix.shape[0]
    
    # Create an empty n x n matrix
    result_matrix = np.zeros((n, n), dtype=complex)
    
    # Populate the new matrix with elements picked from the original matrix
    for i in range(n):
        for j in range(n):
            result_matrix[i, j] = matrix[i, j]  # Directly using the original matrix elements
    
    return result_matrix

n = 4 

# Generate a random unitary matrix for now
unitary_matrix = unitaryMZI(n, theta_row, phi_row)

# Select the elements using the full_u_picker method
picked_elements = full_u_picker(unitary_matrix)

# Rebuild the full matrix using the picked elements
result_matrix = build_u_matrix(unitary_matrix)

print("Unitary Matrix:")
print(unitary_matrix)

print("\nSelected Elements:")
for key, value in picked_elements.items():
    print(f"{key}: {value}")

print("\nReconstructed Matrix from Picked Elements:")
print(result_matrix)


# %%

t_matrix = np.array([[np.sin(0), np.cos(0)], 
                        [np.cos(0), -np.sin(0)]], dtype="complex")

# Define the column vector
t_column = np.array([[1],[1]], dtype="complex")  # You may replace 'a' and 'b' with actual values
print(t_matrix)
print(t_column)

multiply = np.matmul(t_matrix, t_column)
print(multiply)
# %%
def unitaryMZI(n, theta, phi_row):
    # Define the 2x2 matrix using the given theta
    t_matrix = np.array([[np.sin(theta), np.cos(theta)], 
                         [np.cos(theta), -np.sin(theta)]], dtype="complex")
    
    # Define the column vector
    t_column = np.array([[1],[1]], dtype="complex")  # You may replace 'a' and 'b' with actual values
    print(t_matrix)
    print(t_column)
    # Perform the matrix-vector multiplication
    multiply = np.matmul(t_matrix, t_column)

    print(multiply)
    return multiply

print(unitaryMZI(2,np.pi/2,1))

# %%
