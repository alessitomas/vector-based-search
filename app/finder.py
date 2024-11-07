# app/finder.py

import pandas as pd
import torch
import json
import os
from torch import nn
from torch.nn.functional import cosine_similarity


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoisingAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded



class FinderModel:
    def __init__(self):
        
        self.data = pd.read_csv("classes/merged_publications.csv")  
        self.abstracts = self.data['abstract'].fillna("").tolist()
        
        
        self.sentence_embeddings = torch.load('classes/reduced_embeddings.pt')  
        self.vocab = self.load_vocab()  
        self.embedding_layer = self.load_embedding_layer()  
        
        # Load the autoencoder
        self.autoencoder = DenoisingAutoEncoder(input_dim=50, hidden_dim=32)  
        self.autoencoder.load_state_dict(torch.load('classes/glove_embeddings.pth'))  


    def load_vocab(self):
        
        glove_vectors = self.load_glove_vectors("classes/glove.6B/glove.6B.50d.txt")  
        vocab = {"<PAD>": 0, "<UNK>": 1}
        inverse_vocab = ["<PAD>", "<UNK>"]

        for word in glove_vectors.keys():
            vocab[word] = len(inverse_vocab)
            inverse_vocab.append(word)

        return vocab

    def load_glove_vectors(self, glove_file):
        glove_vectors = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
                glove_vectors[word] = vector
        return glove_vectors

    def load_embedding_layer(self):
        
        embedding_dim = 50  
        vocab_size = len(self.vocab) + 2  
        embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        
        
        glove_vectors = self.load_glove_vectors("classes/glove.6B/glove.6B.50d.txt")  
        for idx, word in enumerate(self.vocab.keys()):
            if word in glove_vectors:
                embedding_layer.weight[idx].data = glove_vectors[word]

        return embedding_layer

    def retrieve_publications(self, query, top_k=10):
        
        query_embedding = self.get_query_embedding(query)

        
        similarities = cosine_similarity(query_embedding.unsqueeze(0), self.sentence_embeddings)

        
        top_k_indices = torch.topk(similarities, k=top_k).indices.flatten()  

        
        results = self.data.iloc[top_k_indices.numpy()] 

        
        results_list = []
        for index, row in zip(top_k_indices.numpy(), results.iterrows()):
            if similarities[0, index].item() > 0.4:
                results_list.append({
                    "index": int(index),
                    "title": row[1]['title'],  
                    "abstract": row[1]['abstract'],
                    "similarity": similarities[0, index].item()  
                })

        return results_list


    def get_query_embedding(self, query):
        
        tokens = self.tokenize(query)
        embedding = self.get_sentence_embedding(tokens)  
        with torch.no_grad():  
            reduced_embedding = self.autoencoder.encoder(embedding.unsqueeze(0))  
        return reduced_embedding

    def tokenize(self, text):
        
        tokens = []
        for word in text.split():
            tokens.append(self.vocab.get(word, self.vocab["<UNK>"]))
        return tokens

    def get_sentence_embedding(self, tokens):
        
        embeddings = self.embedding_layer(torch.tensor(tokens))
        return embeddings.mean(dim=0)
