import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np

import random
import math
from torch.autograd import Variable
import argparse
import os
import dataloader_emb as dataloader
import pickle

class Encoder(nn.Module):
    def __init__(self, input_dim,hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.rnn = nn.GRU(input_dim, hid_dim)
    def forward(self, embedding):
        b_len = embedding.size(0)
        outputs = Variable(torch.cuda.DoubleTensor(b_len, embedding.size(1), self.hid_dim))
        for i in range(b_len):
            cur_emb = embedding[i:i+1, :]
            if i==0:
                o, hidden = self.rnn(cur_emb)
            else:
                o, hidden = self.rnn(cur_emb, hidden)
            outputs[i, :, :] = hidden
        outputs = outputs.permute(1,0,2)
        return outputs,hidden
class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.rnn = nn.GRU(output_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim,output_dim)
    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        output, hidden = self.rnn(input, hidden)
        output = self.fc_out(output)
        return output,hidden
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
    def forward(self, src, teacher_forcing_ratio = 0.75):
        src = src.permute(1,0,2)
        trg = src
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).cuda().double()#.to(self.device)
        embs,hidden = self.encoder(src)
        input = trg[0,:]
        for t in range(trg_len):
            #insert input token embedding, previous hidden state and the context state
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden)
            #place predictions in a tensor holding predictions for each token
            outputs[t,:,:] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # print ('teacher_force',teacher_force)
            input = trg[t,:] if teacher_force and t+1<trg_len else output.squeeze(0)
        outputs = outputs.permute(1,0,2)
        return outputs,embs
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i,  src  in enumerate(iterator):
        src = src.cuda()
        optimizer.zero_grad()
        output,_ = model(src)
        loss = criterion(output, src)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, src in enumerate(iterator):
            src = src.cuda()
            output,_ = model(src)
            loss = criterion(output, src)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def getEmb(m,N,hid_dim,fN):
    with open('./data_dict.pkl','rb') as f:
        data_dict,_,img_size = pickle.load(f)
    daily_out_file = './tmp_weight/gru_'+fN+'.txt'
    fout_daily = open(daily_out_file, 'w')
    idl = list(data_dict.keys())
    print ('total ids:',len(idl))
    emb_dict = {}
    for idd in idl:
        date_l = list(data_dict[idd].keys())
        emb_dict[idd] = {}
        for date_ in date_l:
            date_0 = date_
            X_d = data_dict[idd][date_0]
            print (X_d[0][:6])
            X_d = torch.from_numpy(np.array([X_d])).cuda()
            _, avg_vec = m(X_d)
            avg_vec = avg_vec.detach().cpu().numpy()
            emb_dict[idd][date_0] = avg_vec[0]
            avg_vec = avg_vec.reshape([N*hid_dim,])
            print('id',str(idd), 'date', date_0,'id_embed', avg_vec[:5])
            row = str(idd) + ',' + str(date_0) + ',' + (',').join([str(round(_, 3)) for _ in avg_vec]) + '\n'
            fout_daily.write(row)
    fout_daily.close()
    with open('./tmp_weight/gru_'+fN+'.pkl','wb') as f:
        pickle.dump(emb_dict,f)

parser = argparse.ArgumentParser(description='PyTorch GRU Training')
parser.add_argument('--num_epochs', default=15, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=0)
parser.add_argument('--hid_dim', default=64, type=int)
parser.add_argument('--window', default=3, type=int)
parser.add_argument('--input_dim',default=100, type=int) # the number of input features
parser.add_argument('--data_file',default='./data.csv',type=int)
args = parser.parse_args()



SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

INPUT_DIM = args.input_dim 
OUTPUT_DIM = INPUT_DIM
HID_DIM = args.hid_dim
BATCH_SIZE = args.batch_size
DATA_FILE = args.data_file
N_EPOCHS = args.num_epochs
CLIP = 1

if not os.path.exists('./tmp_weight'):
    os.mkdir('./tmp_weight')


loader = dataloader.self_dataloader(batch_size=BATCH_SIZE,N=args.window,data_file=DATA_FILE)
train_iterator = loader.run()

enc = Encoder(INPUT_DIM, HID_DIM)
dec = Decoder(OUTPUT_DIM, HID_DIM)

device = torch.device('cuda') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2Seq(enc, dec, device).to(device).double()



# initialize model
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# train model
# by default, it trained for epoches, but easy to extend it to support early stop for validation set.
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    
    print(f'Epoch: {epoch+1:02} ')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    

getEmb(m=model,N=args.window, hid_dim=args.hid_dim,fN='all_epoch_N'+str(args.window)+'_hid'+str(args.hid_dim))



