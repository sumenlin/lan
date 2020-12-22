import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np

import random
import math
import time
from torch.autograd import Variable
import argparse
import os
# import pandas
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime, timedelta
import dataloader_emb as dataloader
import pickle
def preprocessDataFrame(X_train):
    df = X_train.values
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(df)
    X_train = pd.DataFrame(x_scaled,index=X_train.index, columns=X_train.columns)
    
    return X_train

class Encoder(nn.Module):
    def __init__(self, input_dim,hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.rnn = nn.GRU(input_dim, hid_dim)
    def forward(self, embedding):
        #src = [src len, batch size]#X = X.permute(1, 0, 2) 
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #outputs are the hidden states for each step t
        b_len = embedding.size(0)
        outputs = Variable(torch.cuda.DoubleTensor(b_len, embedding.size(1), self.hid_dim))
        # print ('encoder input',type(embedding),embedding.shape)#3,batch,210
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
        #input = [batch size]
        input = input.unsqueeze(0)
        output, hidden = self.rnn(input, hidden)
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #seq len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
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
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        src = src.permute(1,0,2)
        trg = src
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).cuda().double()#.to(self.device)
        #last hidden state of the encoder is the context
        embs,hidden = self.encoder(src)
        input = trg[0,:]
        for t in range(trg_len):
            #insert input token embedding, previous hidden state and the context state
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden)
            #place predictions in a tensor holding predictions for each token
            outputs[t,:,:] = output
            # print ('trg',trg[t,:].type(),trg[t,:].shape)
            # print ('out',output.type(),output.shape)
            # #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # #if teacher forcing, use actual next token as next input
            # #if not, use predicted token
            # # top1 = output
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
        # print (output.type(),output.shape,'output')
        # print (src.type(),src.shape,'src')
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
    if os.path.exists('../../data/data_dict.pkl'):
        with open('../../data/data_dict.pkl','rb') as f:
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
    else:
        week_fea = ['day_of_week', 'weekend','season']
        news_fea = ['news_low', 'news_high', 'news_range', 'news_mean', 'news_pos', 'news_neg', 'news_low_nomal', 'news_high_nomal', 'news_range_nomal', 'news_mean_nomal', 'news_pos_nomal', 'news_neg_nomal']
        weather_fea = [ 'day_maxtempf', 'day_mintempf', 'day_totalsnowcm', 'tp_tempf', 'tp_weathercode', 'tp_precipmm', 'tp_humidity', 'tp_windchillf', 'tp_feelslikef', 'tp_windspeedmph', 'tp_windgustmph', 'tp_visibility', 'tp_pressure', 'tp_cloudcover', 'tp_heatidxf', 'severe_weather_Avalanche', 'severe_weather_Blizzard', 'severe_weather_Coastal Flood', 'severe_weather_Cold/Wind Chill', 'severe_weather_Debris Flow', 'severe_weather_Dense Fog', 'severe_weather_Dense Smoke', 'severe_weather_Drought', 'severe_weather_Dust Devil', 'severe_weather_Dust Storm', 'severe_weather_Excessive Heat', 'severe_weather_Extreme Cold/Wind Chill', 'severe_weather_Flash Flood', 'severe_weather_Flood', 'severe_weather_Freezing Fog', 'severe_weather_Frost/Freeze', 'severe_weather_Funnel Cloud', 'severe_weather_Hail', 'severe_weather_Heat', 'severe_weather_Heavy Rain', 'severe_weather_Heavy Snow', 'severe_weather_High Surf', 'severe_weather_High Wind', 'severe_weather_Hurricane', 'severe_weather_Ice Storm', 'severe_weather_Lake-Effect Snow', 'severe_weather_Lakeshore Flood', 'severe_weather_Lightning', 'severe_weather_Rip Current', 'severe_weather_Sleet', 'severe_weather_Storm Surge/Tide', 'severe_weather_Strong Wind', 'severe_weather_Thunderstorm Wind', 'severe_weather_Tornado', 'severe_weather_Tropical Depression', 'severe_weather_Tropical Storm', 'severe_weather_Wildfire', 'severe_weather_Winter Storm', 'severe_weather_Winter Weather', 'severe_weather']
        ga_fea = ['active_kilocalories', 'distance_in_meters', 'high_stress_duration_seconds', 'max_stress_level', 'average_hr', 'rest_stress_duration_seconds', 'resting_hr', 'medium_stress_duration_seconds', 'low_stress_duration_seconds', 'intensity_duration_goal_seconds', 'stress_duration_seconds', 'active_time_seconds', 'activity_stress_seconds', 'min_heart_rate', 'steps', 'duration_in_seconds', 'average_stress_level', 'max_hr', 'vigorous_intensity_duration_seconds', 'moderate_intensity_duration_seconds']
        pa_fea = ['act_still_ep_2','act_still_ep_3', 'act_still_ep_0', 'act_still_ep_1', 'act_still_ep_4','unlock_duration_ep_4', 'unlock_duration_ep_0',  'unlock_duration_ep_1','unlock_duration_ep_2',  'unlock_duration_ep_3', 'unlock_num_ep_1', 'unlock_num_ep_0',  'unlock_num_ep_3','unlock_num_ep_2',  'unlock_num_ep_4']
        ba_fea = ['saw_work_beacon', 'saw_home_beacon', 'saw_home_beacon_am', 'saw_work_beacon_am', 'saw_home_beacon_pm', 'saw_work_beacon_pm', 'time_at_work', 'minutes_at_desk', 'number_desk_sessions', 'mean_desk_session_duration', 'median_desk_session_duration', 'percent_at_desk', 'percent_at_work', 'num_5min_breaks', 'num_15min_breaks', 'num_30min_breaks']
        hrv_fea = ['ave_hrv', 'min_hrv', 'max_hrv', 'median_hrv', 'sd_hrv', 'sdann', 'ave_hrv_8_to_6', 'min_hrv_8_to_6', 'max_hrv_8_to_6', 'sdann_8_to_6', 'ave_hrv_not_8_to_6', 'min_hrv_not_8_to_6', 'max_hrv_not_8_to_6', 'sdann_not_8_to_6', 'diff_sdann_8_to_6_to_not', 'ratio_sdann_8_to_6_to_not']
        sm_fea = ['anger', 'anxiety_x', 'negative_affect', 'positive_affect', 'sadness', 'swear', 'causation', 'certainty', 'cognitive_mech', 'inhibition', 'discrepancies', 'negation', 'tentativeness', 'feel', 'hear', 'insight', 'percept', 'see', 'first_person_singular', 'first_person_plural', 'second_person', 'third_person', 'indefinite_pronoun', 'future_tense', 'past_tense', 'present_tense', 'adverbs', 'article', 'verbs', 'auxiliary_verbs', 'conjunction', 'exclusive', 'inclusive', 'preposition', 'quantifier', 'relative', 'achievement', 'bio', 'body', 'death', 'family', 'friends', 'health', 'home', 'humans', 'money', 'religion', 'sexual', 'social', 'work', 'NumTokens', 'Positive', 'Negative', 'Neutral', 'story_x', 'likes_count_x', 'comments_count_x', 'story_y', 'likes_count_y', 'comments_count_y', 'story_x.1', 'likes_count_x.1', 'comments_count_x.1', 'story_y.1', 'likes_count_y.1', 'comments_count_y.1', 'story_x.2', 'likes_count_x.2', 'comments_count_x.2', 'story_y.2', 'likes_count_y.2', 'comments_count_y.2', 'frequency']
        fea =  ga_fea+pa_fea+ba_fea+hrv_fea+sm_fea+week_fea+news_fea+weather_fea
        img_size = len(fea)
        df = pd.read_csv('../../data/daily_gpbhsc_completeid.csv')
        df = df.drop_duplicates().reset_index(drop=True)
        df = df.rename(index=str,columns={'anxiety_y':'anxiety'})
        Xheads = fea
        df_ = df
        df_ = df_.fillna(0)
        df_ = df_.sort_values(by=['snapshot_id', 'date']).reset_index(drop=True)
        data_train = df_
        #normalization
        data_train_x = data_train.loc[:,Xheads]
        data_train_y = data_train.loc[:,['snapshot_id','date']]
        data_train_x = preprocessDataFrame(data_train_x)
        data_train = pd.concat([data_train_y, data_train_x], axis=1) 
        
        idl = data_train.snapshot_id.drop_duplicates().tolist()
        daily_out_file = './tmp_weight/gru_'+fN+'.txt'
        fout_daily = open(daily_out_file, 'w')
        
        for idd in idl:
            date_l = data_train[data_train.snapshot_id.isin([idd])].date.tolist()
            for date_ in date_l:
                date_0 = date_
                d_ = data_train[data_train.snapshot_id.isin([idd])]
                X_d = []
                for j in range(N):
                    date_ = datetime.strptime(date_0, "%Y-%m-%d")
                    date_ = date_ + timedelta(days=j)
                    date_ = datetime.strftime(date_, "%Y-%m-%d")
                    d__ = d_[d_.date.isin(date_)].reset_index(drop=True)
                    if d__.empty:
                        d__ = np.zeros([1,img_size])
                    else:
                        d__ = d_.loc[:,Xheads].values[0]
                        print (d_.loc[:,Xheads].values.shape())
                    X_d.append(d__)
                X_d = np.array([X_d])
                _, avg_vec = m(X_d)
                avg_vec = avg_vec.reshape([1,N*hid_dim])
                print('id',idd, 'date', date_0,'id_embed', avg_vec[:5])
                row = id + ',' + str(date_0) + ',' + (',').join([str(round(_, 3)) for _ in avg_vec]) + '\n'
                fout_daily.write(row)
        fout_daily.close()

parser = argparse.ArgumentParser(description='PyTorch GRU Training')
parser.add_argument('--num_epochs', default=15, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=0)
parser.add_argument('--hid_dim', default=64, type=int)
parser.add_argument('--window', default=3, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

INPUT_DIM = 210
OUTPUT_DIM = INPUT_DIM
HID_DIM = args.hid_dim
BATCH_SIZE = args.batch_size

N_EPOCHS = args.num_epochs
CLIP = 1

if not os.path.exists('./tmp_weight'):
    os.mkdir('./tmp_weight')


loader = dataloader.self_dataloader(batch_size=BATCH_SIZE,N=args.window)
train_iterator,valid_iterator = loader.run()

enc = Encoder(INPUT_DIM, HID_DIM)
dec = Decoder(OUTPUT_DIM, HID_DIM)

device = torch.device('cuda')

model = Seq2Seq(enc, dec, device).to(device).double()



#initialize model
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './tmp_weight/gru_model_N'+str(args.window)+'_hid'+str(args.hid_dim)+'.pt')
    
    print(f'Epoch: {epoch+1:02} ')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
getEmb(m=model,N=args.window, hid_dim=args.hid_dim,fN='all_epoch_N'+str(args.window)+'_hid'+str(args.hid_dim))


