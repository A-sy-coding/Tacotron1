from network import Tacotron
from data import get_dataset, DataLoader, collate_fn, get_param_size, inv_spectrogram, find_endpoint, save_wav, spectrogram
from torch import optim
import numpy as np
import argparse
import os
import time
import torch
import io
import torch.nn as nn
from text.symbols import symbols, en_symbols
import hyperparams as hp
from text import text_to_sequence

'''
해당 py파일은 data가 흘러들어가게 되는 과정을 확인하려고 한다.
'''

# load dataset
dataset = get_dataset()
device = torch.device('cuda:0')
print('------- dataset dictionary key : ',dataset[0].keys()) # dictionary type

if 'english' in hp.cleaners:
        _symbols = en_symbols
print('------ symbols : ',_symbols)


# Tacotron class의 인자값 : (voca_size, emb_dim = 256, enc_hidden_dim = 128, proj_dim = 128, num_mel = 80, dec_hidden_dim = 256, reduction_factor = 5, num_freq = 1024)
# voca_size : len(_symbols)
# emb_dim : 256 (embedding 차원)
# enc_hidden_dim : 126 (encoder hidden 차원)
# proj_dim : 128 (projection 차원)
# num_mel : 80 (mel-spectrogram의 개수)
# dec_hidden_dim : 256 (decoder hidden 차원)
# redunction_factor : 5 (decoder의 하나의 step 당 출력될 mel-spectrogram 개수)
# num_freq : 1024 (Linear-spectrogram의 주파수 설정)
model = Tacotron(len(_symbols)).to(device)


# Make optimizer
optimizer = optim.Adam(model.parameters(), lr=hp.lr)

# Training
model = model.train()

criterion = nn.L1Loss()

# Loss for frequency of human register
n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)
print('---- sample rate : ', hp.sample_rate)
print('---- num_freq : ', hp.num_freq)
print('--- n_priority_freq(사람이 등록한 frequency ? ) : ', n_priority_freq) 

## dataset이 고정된 길이가 아니면 collate_fn 함수를 작성하여 해결해 주어야 한다.
'''
collate_fn : 32개의 batch size를 가진 data들이 들어오게 되고, [batch_size, text(dict), wav(dict)]
text와 wav의 길이들이 데이터마다 각각 다르기 때문에 _prepare_data함수를 사용해 text와 wav길이를 패딩을 사용하여 통일하게 된다.

이후, spectrogram 과 mel-spectrogram의 값을 구하게 되고, timestep이 5의 배수가 되도록 패딩을 수행한다.
출력값으로 text, magnitude, mel이 나오게 된다.
'''
dataloader = DataLoader(dataset, batch_size=32,
                                shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=8)

# sp_data = next(iter(dataloader))
# print('[text, magnitude, mel] : ',len(sp_data)) # 3 -> [text, magnitude, mel]
# print('text의 임베딩한 길이 :',len(sp_data[0][0])) # text : [batch_size, text_embiddimg] -> [32, batch_size마다 padding한 고정된 text 길이]
# print('magnitude frequency 길이 :',len(sp_data[1][0])) # magnitude : [batch_size, frequency, time] -> [32, 1024, 5의 배수만큼 padding한 time 길이]
# print('mel frequency 길이 : ',len(sp_data[2][0])) # mel : [batch_size, mel_frequency, time] -> [32, 80, 5의 배수만큼 padding한 time 길이]

for i, data in enumerate(dataloader):
    current_step = i + 0 * len(dataloader) + 1
    print('----- {}번째 데이터 값 : {}'.format(current_step, len(data)))
    print('text shape : ',data[0].shape) 
    print('magnitude shape : ',data[1].shape)
    print('mel shape : ',data[2].shape)
    ## data -> [text, magnitude, mel] -> 각각마다 batch_size:32

    optimizer.zero_grad()

    # Make decoder input by concatenating [GO] Frame
    try:
        mel_input = np.concatenate((np.zeros([32, hp.num_mels, 1], dtype=np.float32),data[2][:,:,:-1]), axis=2)
    except:
        raise TypeError("not same dimension")

    print('mel에서 마지막 값을 뺀 나머지 shape : ',data[2][:,:,:-1].shape)
    print('mel에서 맨 앞의 <GO> frame을 추가한 shape : ',mel_input.shape)
    
    characters = torch.from_numpy(data[0]).type(torch.cuda.LongTensor).to(device)
    mel_input = torch.from_numpy(mel_input).type(torch.cuda.FloatTensor).to(device)
    mel_spectrogram = torch.from_numpy(data[2]).type(torch.cuda.FloatTensor).to(device)
    linear_spectrogram = torch.from_numpy(data[1]).type(torch.cuda.FloatTensor).to(device)

    print('text shape : ', characters.shape)
    print('<GO> frame 생성한 mel shape : ',mel_input.shape)
    print('원본 mel shape : ',mel_spectrogram.shape)
    print('spectrogram shape (원본으로 사용) : ',linear_spectrogram.shape)
    
    mel_input = torch.transpose(mel_input, 1, 2)
    mel_spectrogram = torch.transpose(mel_spectrogram, 1, 2)
    linear_spectrogram = torch.transpose(linear_spectrogram, 1, 2)
    
    # Forward --> 인자값으로 text와 <go> frame을 추가한 mel 값이 들어가게 된다.
    print('\n ---- Tacotron forward 시작 --------')
    mel_output, linear_output = model.forward(characters, mel_input)