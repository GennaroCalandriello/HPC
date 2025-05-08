import numpy as np

LAYERS = [2, 60, 60 , 60, 1]
NU = 0.001
LR = 0.001
EPOCHS = 4000
N_INT, N_BC, N_IC = 30000, 5000, 5000
T_max = 0.5
SEQ_LEN = 20
NX = 400
NT = 200
DT = 0.001
REYNOLDS = 1000 #Reynolds number, generally 1000-2000, 1000 for simple problems
HIDDEN_SIZE = 50 #LSTM hidden size, generally 50-100, 100 for complex problems
NUM_LAYERS = 3 #LSTM layers, generally 2-4, 3 for complex problems more layers learn more temporal patterns but train slowe
N_SAMPLES = 2000

#lambda values for loss functions in LSTM
LAMBDA_DATA = 1.0 #weight for data loss
LAMBDA_PDE = 0.1 #weight for PDE loss
LAMBDA_BC = 1.0 #weight for BC loss
LAMBDA_IC = 1.0 #weight for IC loss

SAVE_U = False
SAVE_LOSS = True
SAVE_MODEL = True