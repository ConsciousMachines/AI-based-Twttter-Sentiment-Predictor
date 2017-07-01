import numpy as np
import datetime
from datetime import datetime
import pandas as pd
import pandas_datareader.data as web
import warnings
warnings.filterwarnings("ignore")

import tweepy
import textblob
#from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import TextBlob
import re

import matplotlib.pyplot as plt
from tweepy import Stream
from tweepy.streaming import StreamListener
import json

import urllib2
from bs4 import BeautifulSoup
import time


page = 'https://www.google.ca/finance?q=INDEXSP%3A.INX&ei=9aFSWeHRBIO0jAHxy6egDg'
graf_dir = ############## make a path for a folder to store graph pictures in

# FINANCE DATA MINER
'''
out1 = # file output for saving csv

idx = ['SP500', 'VIXCLS', 'VXVCLS']

data = pd.DataFrame()
start = datetime.datetime(2016, 5, 26)
end = datetime.datetime(2017, 6, 26) # today

def creator(fred_codes, start, end):

    yo = web.DataReader(fred_codes[0], "fred", start, end)
    data = pd.DataFrame(yo)

    cdates = data.index

    for i in range(1,len(fred_codes)): # FIRST RUN: CREATE TIME DICTIONARY
        current_code = fred_codes[i]
        
        yo = web.DataReader(current_code, "fred", start, end)
        data = pd.concat([data,yo],axis=1)
        print(data.shape)
        

    # DO LATER AFTER TWITTER
    #data = data.fillna(method='ffill')
    #data = data.dropna()
    #data.to_csv(out1)
    print(data.head(20))
    return data

pre_twitter = creator(idx, start, end) # finished financial data
data_len = pre_twitter.shape[0]
datez = pre_twitter.index # date index to be used in next DataFrame

'''







# TWITTER MINER


# Step 1 - Authenticate
consumer_key= ########### write your twitter API info here
consumer_secret= ############
access_token= ###########
access_token_secret= ###########
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)







# HISTORICAL TWITTER MINING
'''

full = pd.DataFrame(index = datez, columns = ['polarity','subjectivity','tweet'] )
print full.shape


for tweet in topic1:
    try:
        analysis = TextBlob(tweet.text)
        #analysis.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
        time = tweet.created_at
        time2 = # time converted to pandas index format?
        full[time2
        
        polars.append( analysis.sentiment.polarity )
        subject.append( analysis.sentiment.subjectivity )
    except:
        pass
    print('\n')

'''

np.random.seed(1)


# sigmoid function
def nonlin2(x,deriv=False):
    if(deriv==True):
        return np.sum(x)*(1-np.sum(x))
    return 1/(1+np.exp(-np.sum(x)))

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


# TWITTER STREAMING


tweets = []
times = []
polarities = []
subjectivities = []
opinions = []
opxsub = []
prices = []
logprices = []

example_preds = []
nn_preds = []
nn_preds2 = []
nn_preds3 = []
nn_mix = [] 

nn_full = [1]
nn_full2 = [1]

plt.figure(figsize=(10,8))

# TEST DATA
start = datetime(2016, 5, 26)
end = datetime(2017, 6, 26) # today

sp500 = web.DataReader('SP500', "fred", start, end)
sp500= sp500.fillna(method='ffill')
sp500 = sp500.dropna()
sp = np.asarray(sp500['SP500'],dtype=np.float32)
sp_delta = []
for i in range(len(sp)-1):
    sp_delta.append( (sp[i+1]-sp[i])/sp[i])
sp_delta = np.array(sp_delta,dtype=np.float32)*100



global n1
n1 = 6

global syn0, syn1, syn2, syn3, syn4
syn0 = 2*np.random.random([6,n1]) - 1
syn1 = 2*np.random.random([10,10]) - 1
syn2 = 2*np.random.random([5,n1]) - 1
# 2 layer NN
syn3 = 2*np.random.random([10,10]) - 1
syn4 = 2*np.random.random([10,1]) - 1


global b0, b1, b2, b3, b4
b0 = 2*np.random.random(6) - 1
b1 = 2*np.random.random(10) - 1
b2 = 2*np.random.random(6) - 1

b3 = 2*np.random.random(10) - 1
b4 = 2*np.random.random(1) - 1


class StdOutListener(tweepy.StreamListener):
    
    def __init__(self):
        
        self.counter = 1
        
    def on_data(self, data):

        decoded = json.loads(data)
        one_tweet = decoded['text'].encode('ascii', 'ignore')
        tweets.append(one_tweet)
        

        blob = TextBlob(one_tweet)
        blob2 = TextBlob(one_tweet,analyzer=NaiveBayesAnalyzer())
        
        sent = blob.sentiment
        polarities.append(sent[0])
        subjectivities.append(sent[1])



        sent2 = blob2.sentiment

        if sent2[0] == 'pos':
            opxsub.append( sent[1] )
            opinions.append(1)
        if sent2[0] == 'neg':
            opxsub.append( -1*sent[1])
            opinions.append(-1)


        

        print( 'Text: ' , one_tweet)
        print( 'Polarity->',np.round([sent[0],sent[1]],3),'<-Subjectivity')
        

        btime = decoded['created_at'].split()

        timez = btime[3] # append this to compare to SP500 price
        print( timez )

        # GOOGLE FINANCE S&P500 INFO
       
        '''
        page2 = urllib2.urlopen(page)
        soup = BeautifulSoup(page2,"html.parser")
        a = soup.find("span", {"id": "ref_626307_l"})
        a = str(a)
        a = a[24:-7]
        
        a = a[0]+a[2:]
        a = float(a)
        prices.append(float(a))       
        print a, 'S&P500 price'
        '''


        # HAND CRAFTED 2 LAYER NEURAL NETWORK

        '''


        lx = np.array([1,sent[0],sent[1],sent2[1],sent2[2],sp_delta[ self.counter -1],sp_delta[ self.counter -2],sp_delta[ self.counter -3],sp_delta[ self.counter -4],sp_delta[ self.counter -5]])

        for ello in range(10):
            global syn3, syn4, b3, b4
            ll1 = nonlin(np.dot(lx,syn3))
            ll2 = nonlin(np.dot(ll1,syn4))

            ll2_error = np.tile(a1,10) - ll2

            ll2_delta = ll2_error*nonlin(ll2,deriv=True)

            ll1_error = ll2_delta.dot(np.transpose(syn3))

            ll1_delta = ll1_error * nonlin(ll1,deriv=True)
            

            syn4 += np.dot(np.transpose(ll1),ll2_delta)
            syn3 += np.dot(np.transpose(ll0),ll1_delta)
        '''
        
        # HAND CRAFTED 1 LAYER NEURAL NETWORK


        a1 = np.array(sp_delta[ self.counter ])
        a2 = np.array(sp[ self.counter ])
        
        l0 = np.array([1,sent[0],sent[1],sent2[1],sent2[2],sp_delta[ self.counter -1]])
        l00 = np.array([1,sent[0],sent[1],sent2[1],sent2[2],sp_delta[ self.counter -1],sp_delta[ self.counter -2],sp_delta[ self.counter -3],sp_delta[ self.counter -4],sp_delta[ self.counter -5]])
        #l000 = np.array([1,sent[0],sent[1],sent2[1],sent2[2],a2[self.counter-1]])

        b0 = 2*np.random.random(6) - 1
        b1 = 2*np.random.random(10) - 1
        for ello in range(20):
            global syn0
            
            l1 = nonlin(np.dot(l0,syn0)+b0)
            l1_error = np.tile(a1,6) - l1
            l1_delta = l1_error * nonlin(l1,True)
            syn0 += np.dot(np.transpose(l0),l1_delta)
            b0 += l1_delta

            global syn1
            l11 = nonlin(np.dot(l00,syn1)+b1)
            l11_error = np.tile(a1,10) - l11
            l11_delta = l11_error * nonlin(l11,True)
            syn1 += np.dot(np.transpose(l00),l11_delta)
            b1 += l11_delta
            
            '''
            global syn2
            l111 = nonlin(np.dot(l000,syn2))
            l111_error = [a1,a2] - l111
            l111_delta = l111_error * nonlin(l111,True)
            syn2 += np.dot(np.transpose(l000),l111_delta)
            '''

            # 2 layer NN
            global syn3, syn4, b3, b4
            ll1 = nonlin(np.dot(l00,syn3))
            ll2 = nonlin(np.dot(ll1,syn4))

            ll2_error = np.tile(a1,10) - ll2

            ll2_delta = ll2_error*nonlin(ll2,deriv=True)

            ll1_error = ll2_delta.dot(np.transpose(syn3))

            ll1_delta = ll1_error * nonlin(ll1,deriv=True)
            

            syn4 += np.dot(np.transpose(ll1),ll2_delta)
            syn3 += np.dot(np.transpose(l00),ll1_delta)
        #nn_preds.append(l1) # think of a better solution
        lol1 = np.mean(l1)
        lol2 = np.mean(l11)
        lol3 = np.mean(ll2)
        lol4 = np.mean( [lol1,lol2,lol3])
        nn_preds.append( lol1 )
        nn_preds2.append( lol2)
        nn_preds3.append( np.mean(ll2))
        nn_mix.append( lol4 )
        
        #nn_preds3.append(l111)

        #nn_full.append( nn_full[-1] + lol1*nn_full[-1] )
        #nn_full2.append( nn_full2[-1] + lol2*nn_full2[-1] )





        plt.cla()
        plt.subplot(3, 1, 1)
        #plt.ylim([min(prices)+0.01, max(prices)-0.01])
        plt.plot(sp_delta[:self.counter-1],color='black',label='S&P500 %change x 100: '+str(a1))
        plt.plot(nn_preds,color='red',alpha=0.4,label='Twitter Impact Neural Net')
        plt.plot(nn_preds2,color='pink',alpha=0.8,label='Impact + 5 steps back NN')
        plt.plot(nn_preds3,color='green',alpha=0.4,label='Impact + 5 steps 2 layer')
        plt.plot(nn_preds3,color='purple',alpha=0.9,label='Mixture Model')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        x = polarities
        y = opxsub
        plt.hist2d(x, y, cmax = 1, bins=25,cmap='spring')
        plt.ylim([-1,1])
        plt.xlim([-1,1])
        plt.xlabel('Text Polarity')
        plt.ylabel('Pos/Neg Opinion * Subjectivity')
        plt.text(0,0.5, 'Twitter Sentiment',fontsize=15,color='cyan')
        plt.colorbar()

        plt.subplot(3, 1, 3)

        plt.legend()
        if len(tweets) > 4:
            plt.text(0,0, tweets[-1],fontsize=12)
            plt.text(0,0.3,tweets[-2] ,fontsize=12)
            plt.text(0,0.6, tweets[-3],fontsize=12)
            plt.text(0,0.9,tweets[-4] ,fontsize=12)
        plt.axis('off')
        plt.draw()
        plt.pause(0.0001)
        plt.savefig(graf_dir+str(self.counter))
        plt.clf()

        self.counter += 1


class StdOutListener2(tweepy.StreamListener):
    def __init__(self):     
        self.counter = 0
        
    def on_data(self, data):

        decoded = json.loads(data)
        one_tweet = decoded['text'].encode('ascii', 'ignore')
        tweets.append(one_tweet)
        

        blob = TextBlob(one_tweet)
        blob2 = TextBlob(one_tweet,analyzer=NaiveBayesAnalyzer())
        
        sent = blob.sentiment
        polarities.append(sent[0])
        subjectivities.append(sent[1])



        sent2 = blob2.sentiment

        if sent2[0] == 'pos':
            opxsub.append( sent[1] )
            opinions.append(1)
        if sent2[0] == 'neg':
            opxsub.append( -1*sent[1])
            opinions.append(-1)
        self.counter +=1
        if self.counter >= 5:
            stream.disconnect()
            


# most important part
l = StdOutListener()
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)


l2 = StdOutListener()
stream = tweepy.Stream(auth, l2)
stream.filter(track=['trump','wages','hillary','democracy', 'clinton', 'federal reserve', 'economy', 'economics', 'bank rate', 'unemployment', 'usd','income','wage' ])

while True:
    stream = tweepy.Stream(auth, l)
    stream.filter(track=['trump','wages','hillary','democracy', 'clinton', 'federal reserve', 'economy', 'economics', 'bank rate', 'unemployment', 'usd','income','wage' ])










