import numpy as np

def calculateMSE(true, pred):
    
    # creating the perfect output
    trueOutput = [0] * 10
    trueOutput[true] = 1
    
    # Convert lists to numpy arrays
    trueOutput = np.array(trueOutput)
    pred = np.array(pred)
    
    print("True output:", trueOutput)
    print("Prediction:", pred)
    
    # Check if the lists have the same length
    if len(trueOutput) != len(pred):
        raise ValueError("The two lists must have the same length.")
    
    # Calculate MSE
    mse = np.mean((trueOutput - pred) ** 2)
    
    return mse

# Example usage
print("MSE:", calculateMSE(5, [0,0,0,0,0,1,0,0,0,0]))