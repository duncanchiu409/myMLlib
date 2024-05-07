import numpy as np
from myMLlib.LogisticRegression import LogisticRegression

def main():
    # Generate an array of evenly spaced values between -10 and 10
    z_tmp = np.arange(-10,11)

    # Use the function implemented above to get the sigmoid values
    y = sigmoid(z_tmp)

    # Code for pretty printing the two arrays next to each other
    np.set_printoptions(precision=3) 
    print("Input (z), Output (sigmoid(z))")
    print(np.c_[z_tmp, y])
    
main()