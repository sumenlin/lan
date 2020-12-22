from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime, timedelta
import pickle
import os
class self_dataset(Dataset): 
    def __init__(self, N): 
        self.win = N
        X, img_size = self.setData()
        self.imgs = X
        self.img_size = img_size
        
    def setData(self):
        if os.path.exists('../../data/data_dict.pkl'):
            with open('../../data/data_dict.pkl','rb') as f:
                _,X_train,img_size = pickle.load(f)
            print ('X_train shape:', X_train.shape)
            return X_train,img_size


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
        
        
        data_dict = {}
        X_train = []
        idl = data_train.snapshot_id.drop_duplicates().tolist()
        print ('total',len(data_train))
        empty_cnt1 = 0
        empty_cnt2 = 0
        for iddx,idd in enumerate(idl):
            print ('idd',iddx,idd)
            data_dict[idd] = {}
            date_l = data_train[data_train.snapshot_id.isin([idd])].date.tolist()
            d_ = data_train[data_train.snapshot_id.isin([idd])]
            for date_ in date_l:
                date_0 = date_
                X_d = []
                flag = 0
                for j in range(self.win):
                    date_ = datetime.strptime(date_0, "%Y-%m-%d")
                    date_ = date_ + timedelta(days=j)
                    date_ = datetime.strftime(date_, "%Y-%m-%d")
                    # print (date_0,j,date_)
                    d__ = d_[d_.date.isin([date_])].reset_index(drop=True)
                    if d__.empty:
                        flag+=1
                        d__ = np.zeros([img_size,])
                    else:
                        d__ = d__.loc[:,Xheads].values[0]
                    X_d.append(d__)
                if flag==1:
                    empty_cnt1+=1
                elif flag==2:
                    empty_cnt2+=1
                data_dict[idd][date_0]=np.array(X_d)
                # print ('feature', idd, date_0,X_d[0][:5])
                X_train.append(X_d)
        X_train = np.array(X_train)
        print('X_train shape:', X_train.shape)
        print ('empty x:', empty_cnt1,empty_cnt2)
        with open('../../data/data_dict.pkl','wb') as f:
             pickle.dump([data_dict,X_train,img_size], f)
        return X_train,img_size

    def __getitem__(self, index):  
        return self.imgs[index]
    
    def __len__(self):
        return len(self.imgs)
        
class self_dataloader():  
    def __init__(self, batch_size, N, num_workers=1, shuffle=True):
    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.N = N
        self.img_size = 0
    def run(self):
        train_dataset = self_dataset(N=self.N)
        self.img_size = train_dataset.img_size
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers)
        val_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader


def preprocessDataFrame(X_train):

    df = X_train.values
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(df)
    X_train = pd.DataFrame(x_scaled,index=X_train.index, columns=X_train.columns)
    
    return X_train












