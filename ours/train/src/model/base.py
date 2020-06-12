"""Class for text data."""
import os
import math
import string
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
import src.model.resnet as resnet

from src.spellchecker import SpellChecker
from torch.autograd import Variable

__VERBOSE__ = False


class SimpleVocab(object):
    def __init__(self):
        super(SimpleVocab, self).__init__()
        self.spell = SpellChecker()
        self.word2id = {}
        self.wordcount = {}
        self.add_special_token('[PAD]')
        self.add_special_token('[CLS]')
        self.add_special_token('[SEP]')
        
    def add_special_token(self, token):
        self.word2id[token] = len(self.word2id)
        self.wordcount[token] = 9e9

    def tokenize_text(self, text):
        # fast
        text = text.encode('ascii', 'ignore').decode('ascii')
        table = str.maketrans(dict.fromkeys(string.punctuation))
        tokens = str(text).lower().translate(table).strip().split()
        return tokens

    def add_text_to_vocab(self, text):
        tokens = self.tokenize_text(text)
        if __VERBOSE__:
            print(f'[Tokenizer] Text: {text} / Tokens: {tokens}')
        for token in tokens:
            token = SpellChecker.correct_token(token)
            if not token in self.word2id:
                self.word2id[token] = len(self.word2id)
                self.wordcount[token] = 0
            self.wordcount[token] += 1

    def encode_text(self, text):
        tokens = self.tokenize_text(text)
        if len(tokens) > 0:
            x = [ self.word2id.get(SpellChecker.correct_token(token), 0) for token in tokens ]
        else:
            x = [0]
        return x

    def get_size(self):
        return len(self.word2id)


class Word2Vec(nn.Module):
    def __init__(self, vocab, embed_size, init_with_glove=False):
        super(Word2Vec, self).__init__()
        vocab_size = vocab.get_size() 

        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        if init_with_glove:
            import pickle
            import urllib.request
            from tqdm import tqdm
            print(f'Initialize Word2Vec with GloVe vectors...')
            word2vec_path = 'assets/word2vec'
            with open(os.path.join(word2vec_path, 'glove_840B_300d/embeddings.pkl'), 'rb') as f:
                self.word2vec_a = pickle.load(f)
            with open(os.path.join(word2vec_path, 'glove_42B_300d/embeddings.pkl'), 'rb') as f:
                self.word2vec_b = pickle.load(f)
            with open(os.path.join(word2vec_path, 'glove_6B_300d/embeddings.pkl'), 'rb') as f:
                self.word2vec_c = pickle.load(f)
            print('Done.')
        
            word2id = vocab.word2id
            id2word = [0] * len(word2id)
            for k, v in word2id.items():
                id2word[v] = k
            
            weights = []
            for idx, word in enumerate(tqdm(id2word)):
                if (word in self.word2vec_a) and (word in self.word2vec_b) and (word in self.word2vec_c):
                    w_a = torch.FloatTensor(self.word2vec_a[word])
                    w_b = torch.FloatTensor(self.word2vec_b[word])
                    w_c = torch.FloatTensor(self.word2vec_c[word])
                    w = torch.cat([w_a, w_b, w_c])
                    weights.append(w)
                else:
                    if __VERBOSE__:
                        print(f'[Warn] token "{word}" does not exist in GloVe.')
                    weights.append(self.embedding.weight[idx])
            weights = torch.stack(weights)
            self.embedding.weight = nn.Parameter(weights)

    def forward(self, x):
        return self.embedding(x)


class TextSWEMModel(nn.Module):                     
    def __init__(self,                              
                 fc_arch,                           
                 in_dim,                            
                 out_dim):                          
        super(TextSWEMModel, self).__init__()       
                                                    
        if fc_arch == 'A':                          
            self.fc_output = torch.nn.Sequential(   
                torch.nn.BatchNorm1d(in_dim),       
                torch.nn.Linear(in_dim, out_dim),   
            )                                       
        elif fc_arch == 'B':                        
            self.fc_output = nn.Linear(in_dim, out_dim)
                                                   
    def forward(self, x):                          
        '''x: sentence embedding                   
        '''                                        
        return self.fc_output(x)                   
                                                   

class TextLSTMModel(nn.Module):
    def __init__(self,
                 fc_arch,
                 texts_to_build_vocab,
                 word_embed_dim,
                 lstm_hidden_dim,
                 init_with_glove):
        super(TextLSTMModel, self).__init__()

        self.vocab = SimpleVocab()
        for text in tqdm(texts_to_build_vocab):
            self.vocab.add_text_to_vocab(text) 
        vocab_size = self.vocab.get_size()     
                                               
        self.word_embed_dim = word_embed_dim   
        self.lstm_hidden_dim = lstm_hidden_dim 
        self.embedding_layer = Word2Vec(self.vocab, word_embed_dim, init_with_glove=init_with_glove)
                                               
        self.num_layers = 2                    
        self.lstm = torch.nn.LSTM(word_embed_dim, lstm_hidden_dim,
                                  num_layers=self.num_layers, dropout=0.1)
                                               
        if fc_arch == 'A':                     
            self.fc_output = torch.nn.Sequential(
                torch.nn.BatchNorm1d(lstm_hidden_dim),
                torch.nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
            )                                  
        elif fc_arch == 'B':                   
            self.fc_output = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
                                               
    def forward(self, x):                      
        """ input x: list of strings"""        
        if isinstance(x, list) or isinstance(x, tuple):
            if isinstance(x[0], str) or isinstance(x[0], unicode):
                x = [self.vocab.encode_text(text) for text in x]
        assert isinstance(x, list) or isinstance(x, tuple)
        assert isinstance(x[0], list) or isinstance(x[0], tuple)
        assert isinstance(x[0][0], int)            
        return self.forward_encoded_texts(x)       
                                                   
    def forward_encoded_texts(self, texts):        
        # to tensor                                
        lengths = [len(t) for t in texts]          
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        for i in range(len(texts)):                
            itexts[:lengths[i], i] = torch.LongTensor(texts[i])
                                                   
        # embed words                              
        itexts = Variable(itexts).cuda()           
        etexts = self.embedding_layer(itexts)      
                                                   
        # lstm                                     
        lstm_output, _ = self.forward_lstm_(etexts)
                                                   
        # get last output (using length)           
        text_features = []                         
        for i in range(len(texts)):                
            text_features.append(lstm_output[lengths[i] - 1, i, :])
                                                   
        # output                                   
        text_features = torch.stack(text_features)
        text_features = self.fc_output(text_features)
        return text_features                       
                                                   
    def forward_lstm_(self, etexts):               
        """https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        """
        batch_size = etexts.shape[1]
        first_hidden = (
            torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim).cuda(),
            torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim).cuda(),
        )
        lstm_output, last_hidden = self.lstm(etexts, first_hidden)
        return lstm_output, last_hidden


class TextLSTMGRUModel(nn.Module):
    def __init__(self,
                 fc_arch,
                 texts_to_build_vocab,
                 word_embed_dim,
                 hidden_dim,
                 init_with_glove):
        super(TextLSTMGRUModel, self).__init__()

        self.vocab = SimpleVocab()
        for text in tqdm(texts_to_build_vocab):
            self.vocab.add_text_to_vocab(text) 
        vocab_size = self.vocab.get_size()     
                                               
        self.word_embed_dim = word_embed_dim   
        self.hidden_dim = hidden_dim 
        self.embedding_layer = Word2Vec(self.vocab, word_embed_dim, init_with_glove=init_with_glove)
                                    
        # 2-layer LSTM.
        self.num_layers = 2                    
        self.lstm = torch.nn.LSTM(word_embed_dim, hidden_dim,
                                  num_layers=self.num_layers,
                                  dropout=0.1,
                                  bidirectional=False)

        # 2-layer GRU
        self.num_layers = 2                    
        self.gru = torch.nn.GRU(word_embed_dim, hidden_dim,
                                num_layers=self.num_layers,
                                dropout=0.1,
                                bidirectional=False)

        if fc_arch == 'A':   
            self.fc_output = torch.nn.Sequential(
                nn.BatchNorm1d(2 * hidden_dim),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )                                  
        elif fc_arch == 'B':                   
            self.fc_output = nn.Linear(2 * hidden_dim, hidden_dim)
                                               
    def forward(self, x):                      
        """ input x: list of strings"""        
        if isinstance(x, list) or isinstance(x, tuple):
            if isinstance(x[0], str) or isinstance(x[0], unicode):
                x = [ self.vocab.encode_text(text) for text in x ]
        assert isinstance(x, list) or isinstance(x, tuple)
        assert isinstance(x[0], list) or isinstance(x[0], tuple)
        assert isinstance(x[0][0], int)            
        return self.forward_encoded_texts(x)       
                                                   
    def forward_encoded_texts(self, texts):        
        # to tensor                                
        lengths = [len(t) for t in texts]          
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        for i in range(len(texts)):                
            itexts[:lengths[i], i] = torch.LongTensor(texts[i])
                                                   
        # embed words                              
        itexts = Variable(itexts).cuda()           
        etexts = self.embedding_layer(itexts)      
                                                   
        # lstm                                     
        lstm_output, _ = self.forward_lstm_(etexts)
        gru_output, _ = self.forward_gru_(etexts)
                                                   
        # get last output (using length)           
        text_features = []                         
        for i in range(len(texts)):
            _lstm_feat = lstm_output[lengths[i] - 1, i, :]      # batch, num_directions * hidden_size
            _gru_feat = gru_output[lengths[i] - 1, i, :]        # batch, num_directions * hidden_size
            text_features.append(torch.cat([_lstm_feat, _gru_feat]))
                                                   
        # output                                   
        text_features = torch.stack(text_features)
        text_features = self.fc_output(text_features)
        return text_features                       
                                                   
    def forward_lstm_(self, etexts):               
        """https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        """
        lstm_output, last_hidden = self.lstm(etexts)
        return lstm_output, last_hidden
    
    def forward_gru_(self, etexts):               
        """https://pytorch.org/docs/stable/nn.html#torch.nn.GRU
        """
        gru_output, last_hidden = self.gru(etexts)
        return gru_output, last_hidden


class ImageEncoderTextEncoderBase(nn.Module):
    """Base class for image and text encoder.
    """

    def __init__(self, backbone, texts, text_method, fdims, init_with_glove, fc_arch):
        super(ImageEncoderTextEncoderBase, self).__init__()

        self.in_feature_image = None
        self.out_feature_image = fdims
        self.in_feature_text = 3 * 300
        self.out_feature_text = fdims

        # load pretrained weights.
        pretrained = True
        if backbone in [
            'resnet18', 'resnet34', 'resnet50', 'resnet101',
            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
            'wide_resnet50_2', 'wide_resnet101_2'
        ]:
            self.backbone = resnet.__dict__[backbone](pretrained=pretrained, num_classes=1000)

        print(f"Backbone: {backbone} is loaded with pretrained={pretrained}")

        if backbone in ['resnet18']:
            self.in_feature_image = 512
        else:
            self.in_feature_image = 2048

        if init_with_glove:
            self.in_feature_text = 3 * 300

        # define image/text models.
        self.model = dict()
        self.model['backbone'] = self.backbone
        self.model['image_encoder'] = nn.Sequential(
            nn.Linear(self.in_feature_image, self.out_feature_image)
        )

        if text_method == 'lstm':
            self.model['text_encoder'] = TextLSTMModel(
                fc_arch=fc_arch,
                texts_to_build_vocab=texts,
                word_embed_dim=self.in_feature_text,
                lstm_hidden_dim=self.out_feature_text,
                init_with_glove=init_with_glove,
            )
        elif text_method == 'lstm-gru':
            self.model['text_encoder'] = TextLSTMGRUModel(
                fc_arch=fc_arch,
                texts_to_build_vocab=texts,
                word_embed_dim=self.in_feature_text,
                hidden_dim=self.out_feature_text,
                init_with_glove=init_with_glove,
            )
        elif text_method == 'swem':             # swem: concatenation of mean & max pool of glove vector (300-d + 300-d = 600-d)
            self.in_feature_text = 600
            self.out_feature_text = 600
            self.model['text_encoder'] = TextSWEMModel(
                fc_arch=fc_arch,
                in_dim=self.in_feature_text,
                out_dim=self.out_feature_text,
            )

    def extract_image_feature(self, x):
        x = self.model['backbone'](x)
        x = self.model['image_encoder'](x)
        return x

    def extract_text_feature(self, texts):
        x = self.model['text_encoder'](texts)
        return x
