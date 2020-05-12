import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # Set the hidden size for init_hidden
        self.hidden_size = hidden_size
        
        # Embedded layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        
        # Fully Connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)   
    
    def forward(self, features, captions):        
        # Hidden state
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden = (torch.zeros(1, features.shape[0], self.hidden_size, device = device),
                  torch.zeros(1, features.shape[0], self.hidden_size, device = device))
        
        # Embedding the captions
        embedded = self.embed(captions[:,:-1])        
        embedded = torch.cat((features.unsqueeze(1), embedded), dim = 1)
                
        # LSTM
        lstm_out, self.hidden = self.lstm(embedded, hidden)
        
        # Functional component
        out = self.fc(lstm_out)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # Hidden state
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        hidden = (torch.zeros(1, inputs.shape[0], self.hidden_size, device = device),
                  torch.zeros(1, inputs.shape[0], self.hidden_size, device = device))

        out_list = list()
        word_len = 0
        
        with torch.no_grad():
            while word_len < max_len:
                lstm_out, hidden = self.lstm(inputs, hidden)
                result = self.fc(lstm_out)
                result = result.squeeze(1)
                result = result.argmax(dim = 1)               
                out_list.append(result.item())                
                inputs = self.embed(result.unsqueeze(0))                
                word_len += 1
                
                if result == 1:
                    break
        
        return out_list