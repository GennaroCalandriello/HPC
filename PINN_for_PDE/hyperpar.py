import numpy as np

LAYERS = [2, 60, 60 , 60, 1]
NU = 0.01/np.pi
LR = 0.006
EPOCHS = 6000
N_INT, N_BC, N_IC = 30000, 5000, 5000
T_max = 0.5
SEQ_LEN = 20
NX = 200
NT = 200
DT = 0.1
HIDDEN_SIZE = 50 #LSTM hidden size, generally 50-100, 100 for complex problems
NUM_LAYERS = 3 #LSTM layers, generally 2-4, 3 for complex problems more layers learn more temporal patterns but train slowe
N_SAMPLES = 2000

SAVE_U = False
SAVE_LOSS = True
SAVE_MODEL = True