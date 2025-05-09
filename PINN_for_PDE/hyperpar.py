import numpy as np

LAYERS = [2, 60, 60 , 60, 1]
NU = 0.001
LR = 0.001
EPOCHS = 10000
N_INT, N_BC, N_IC = 30000, 5000, 5000
T_max = 0.5
SEQ_LEN = 20
NX = 400
NT = 200
DT = 0.001
X_LB, X_UB, Y_LB, Y_UB = -1.0, 1.0, -1.0, 1.0
T_LB, T_UB = 0.0, 1.0

REYNOLDS = 1000 #Reynolds number, generally 1000-2000, 1000 for simple problems
HIDDEN_SIZE = 50 #LSTM hidden size, generally 50-100, 100 for complex problems
NUM_LAYERS = 3 #LSTM layers, generally 2-4, 3 for complex problems more layers learn more temporal patterns but train slowe
N_SAMPLES = 300
N_COLLOCATION = 10000 #number of collocation points, generally 1000-10000, 10000 for complex problems
#lambda values for loss functions in LSTM
LAMBDA_DATA = 1.0 #weight for data loss
LAMBDA_PDE = 1 #weight for PDE loss
LAMBDA_BC = 1.0 #weight for BC loss
LAMBDA_IC = 1.0 #weight for IC loss
BATCH_SIZE = 64 #batch size for training, generally 32-128, 64 for complex problems

SAVE_U = False
SAVE_LOSS = True
SAVE_MODEL = True