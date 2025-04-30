import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython import display
from SEATBELT_TRAIN_TEST.model_basic import *

model = ActionRecognitionLSTM(input_size=4, hidden_size=64, num_classes=5, num_layers=4)

batch_size = 1
sequence_length = 5
input_size = 4

from torchview import draw_graph
architecture = 'ActionRecognitionLSTM_basic'
# architecture = 'ActionRecognitionLSTM_attention'
model_graph = draw_graph(model, input_size=(batch_size,sequence_length,input_size), graph_dir ='TB' ,
                         expand_nested=True, graph_name=f'{architecture}',
                         save_graph=True, filename=f'./output/{architecture}')
model_graph.visual_graph

