import torch as pt

# Check if CUDA is available
if pt.cuda.is_available():
    device = pt.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = pt.device("cpu")
    print("CUDA is not available. Using CPU.")

    