import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
import vocab
from sklearn.utils import shuffle
import pandas as pd

class ProductionPhonClassifier(nn.Module):
    def __init__(self, input_vocab_size, n_embedding_dims, n_hidden_dims, n_lstm_layers, output_class_size, pretrained_embedding):
      
        super(ProductionPhonClassifier, self).__init__()
        self.lstm_dims = n_hidden_dims
        self.lstm_layers = n_lstm_layers

        self.input_lookup = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=n_embedding_dims)
        self.lstm = nn.LSTM(input_size=n_embedding_dims, hidden_size=n_hidden_dims, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(in_features=n_hidden_dims*2, out_features=n_hidden_dims)
        self.output = nn.Linear(in_features=n_hidden_dims, out_features=output_class_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.input_lookup.from_pretrained(pretrained_embedding, freeze=False)

    def forward(self, history_tensor_p,history_tensor_t, prev_hidden_state):
        """
        Given a history, and a previous timepoint's hidden state, predict 

        """     
        embed_p = self.input_lookup(history_tensor_p) 
        embed_t = self.input_lookup(history_tensor_t)

        lstm_p, h_p = self.lstm(embed_p)
        lstm_t, h_t = self.lstm(embed_t)

        last_out_p = lstm_p[:,-1,:] 
        last_out_t = lstm_t[:,-1,:] 

        linear_in= torch.cat((last_out_p,last_out_t), 1)

        linear_out=self.linear(linear_in)
        out = self.output(linear_out)
        out = self.softmax(out)

        return out, (h_p, h_t)
        
    def init_hidden(self):
        """
        Generate a blank initial history value, for use when we start predicting over a fresh sequence.
        """
        h_0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        c_0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        return(h_0,c_0)

class ProductionClassifier(nn.Module):
    def __init__(self, input_vocab_size, n_embedding_dims, n_hidden_dims, n_lstm_layers, output_class_size, pretrained_embedding):
      
        super(ProductionClassifier, self).__init__()
        self.lstm_dims = n_hidden_dims
        self.lstm_layers = n_lstm_layers

        self.input_lookup = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=n_embedding_dims)
        self.lstm = nn.LSTM(input_size=n_embedding_dims, hidden_size=n_hidden_dims, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        self.output = nn.Linear(in_features=n_hidden_dims, out_features=output_class_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.input_lookup.from_pretrained(pretrained_embedding, freeze=False)

    def forward(self, history_tensor, prev_hidden_state):
        """
        Given a history, and a previous timepoint's hidden state, predict 

        """     
        out = self.input_lookup(history_tensor)
        out, hidden = self.lstm(out)
        last_out = out[:,-1,:] 
        out = self.output(last_out)
        out = self.softmax(out)

        return out, hidden
        
    def init_hidden(self):
        """
        Generate a blank initial history value, for use when we start predicting over a fresh sequence.
        """
        h_0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        c_0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        return(h_0,c_0)

class ProductionClassifier2(nn.Module):
    def __init__(self, input_vocab_size, n_embedding_dims, n_hidden_dims, n_lstm_layers, output_class_size, pretrained_embedding):
      
        super(ProductionClassifier2, self).__init__()
        self.lstm_dims = n_hidden_dims
        self.lstm_layers = n_lstm_layers

        self.input_lookup = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=n_embedding_dims)
        self.lstm = nn.LSTM(input_size=n_embedding_dims, hidden_size=n_hidden_dims, num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(in_features=n_hidden_dims*2, out_features=n_hidden_dims)
        self.output = nn.Linear(in_features=n_hidden_dims, out_features=output_class_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.input_lookup.from_pretrained(pretrained_embedding, freeze=False)

    def forward(self, history_tensor_p,history_tensor_t, prev_hidden_state):
        """
        Given a history, and a previous timepoint's hidden state, predict 

        """     
        embed_p = self.input_lookup(history_tensor_p) 
        embed_t = self.input_lookup(history_tensor_t)

        lstm_p, h_p = self.lstm(embed_p)
        lstm_t, h_t = self.lstm(embed_t)

        last_out_p = lstm_p[:,-1,:] 
        last_out_t = lstm_t[:,-1,:] 

        linear_in= torch.cat((last_out_p,last_out_t), 1)

        linear_out=self.linear(linear_in)
        out = self.output(linear_out)
        out = self.softmax(out)

        return out, (h_p, h_t)
        
    def init_hidden(self):
        """
        Generate a blank initial history value, for use when we start predicting over a fresh sequence.
        """
        h_0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        c_0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        return(h_0,c_0)
       


class NameGenerator(nn.Module):
    def __init__(self, input_vocab_size, n_embedding_dims, n_hidden_dims, n_lstm_layers, output_vocab_size):
        """
        Initialize our name generator, following the equations laid out in the assignment. In other words,
        we'll need an Embedding layer, an LSTM layer, a Linear layer, and LogSoftmax layer. 
        
        Note: Remember to set batch_first=True when initializing your LSTM layer!

        Also note: When you build your LogSoftmax layer, pay attention to the dimension that you're 
        telling it to run over!
        """
        super(NameGenerator, self).__init__()
        self.lstm_dims = n_hidden_dims
        self.lstm_layers = n_lstm_layers

        self.input_lookup = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=n_embedding_dims)
        self.lstm = nn.LSTM(input_size=n_embedding_dims, hidden_size=n_hidden_dims, num_layers=n_lstm_layers, batch_first=True, bidirectional=True)
        self.output = nn.Linear(in_features=n_hidden_dims*2, out_features=output_vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)


    def forward(self, history_tensor, prev_hidden_state):
        """
        Given a history, and a previous timepoint's hidden state, predict the next character. 
        
        Note: Make sure to return the LSTM hidden state, so that we can use this for
        sampling/generation in a one-character-at-a-time pattern, as in Goldberg 9.5!
        """     
        out = self.input_lookup(history_tensor)

        out, hidden = self.lstm(out)
        out = self.output(out)
        out = self.softmax(out)
        #last_out = out[:,-1,:].squeeze() 
        return out, hidden
        
    def init_hidden(self):
        """
        Generate a blank initial history value, for use when we start predicting over a fresh sequence.
        """
        h_0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        c_0 = torch.randn(self.lstm_layers, 1, self.lstm_dims)
        return(h_0,c_0)
       
### Utility functions

def train_output_self(model, epochs, training_data, c2i):
    """
    Train model for the specified number of epochs, over the provided training data.
    
    Make sure to shuffle the training data at the beginning of each epoch!
    """
    opt = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.NLLLoss() # since our model gives negative log probs on the output side
        
    for i in range(epochs):
        random.shuffle(training_data)
        loss = 0
        
        for idx, word in enumerate(training_data):
            
            opt.zero_grad()
            word_tens=vocab.sentence_to_tensor(word, c2i, True)
            x_tens = word_tens[:,:-1]
            y_tens = word_tens[:,1:]
            
            y_hat,_ = model(x_tens, model.init_hidden())
                        
            loss += loss_func(y_hat.squeeze(), y_tens.squeeze())
            
            if idx % 5000 == 0:
                print(f"{idx}/{len(training_data)}, loss: {loss}")
                
            # send back gradients:
            loss.backward()
            # now, tell the optimizer to update our weights:
            opt.step()
            loss = 0
                
    return model

def train_output_y(model, epochs, training_inputs_1,training_inputs_2, c2i, training_outputs, o2i):
    """
    Train model for the specified number of epochs, over the provided training data.
    
    Make sure to shuffle the training data at the beginning of each epoch!
    """
    opt = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.NLLLoss() # since our model gives negative log probs on the output side
        
    for i in range(epochs):
        training_inputs_1,training_inputs_2, training_outputs = shuffle(training_inputs_1, training_inputs_2, training_outputs)
        loss = 0
        
        data=zip(training_inputs_1,training_inputs_2,training_outputs)

        for idx, datum in enumerate(data):
            t1=datum[0]
            t2=datum[1]
            output=datum[2]
            #output=training_outputs.iloc[idx]

            opt.zero_grad()
            t1_tens=vocab.sentence_to_tensor(t1, c2i, True)
            t2_tens=vocab.sentence_to_tensor(t2, c2i, False)
            
            x_tens = torch.cat((t2_tens,t1_tens),1)
            
            y_tens=vocab.sentence_to_tensor([output], vocab=o2i)
            
            y_hat,_ = model(x_tens, model.init_hidden())
            loss += loss_func(y_hat, y_tens[0])
            
            if idx % 1000 == 0:
                print(f"{idx}/{len(training_inputs_1)}, loss: {loss}")
                
            # send back gradients:
            loss.backward()
            # now, tell the optimizer to update our weights:
            opt.step()
            loss = 0
                
    return model

def train_output_y2(model, epochs, training_inputs_1,training_inputs_2, c2i, training_outputs, o2i):
    """
    Train model for the specified number of epochs, over the provided training data.
    
    Make sure to shuffle the training data at the beginning of each epoch!
    """
    opt = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.NLLLoss() # since our model gives negative log probs on the output side
        
    for i in range(epochs):
        training_inputs_1,training_inputs_2, training_outputs = shuffle(training_inputs_1, training_inputs_2, training_outputs)
        loss = 0
        
        data=zip(training_inputs_1,training_inputs_2,training_outputs)

        for idx, datum in enumerate(data):
            t1=datum[0]
            t2=datum[1]
            output=datum[2]

            opt.zero_grad()

            t1_tens=vocab.sentence_to_tensor(t1, c2i, True)
            t2_tens=vocab.sentence_to_tensor(t2, c2i, False)
            
            #x_tens = torch.cat((t2_tens,t1_tens),1)
            
            y_tens=vocab.sentence_to_tensor([output], vocab=o2i)
            
            y_hat,_ = model(t1_tens, t2_tens, model.init_hidden())
            loss += loss_func(y_hat, y_tens[0])
            
            if idx % 1000 == 0:
                print(f"{idx}/{len(training_inputs_1)}, loss: {loss}")
                
            # send back gradients:
            loss.backward()
            # now, tell the optimizer to update our weights:
            opt.step()
            loss = 0
                
    return model

def train_output_binary(model, epochs, training_inputs_1,training_inputs_2, c2i, training_outputs):
    """
    Train model for the specified number of epochs, over the provided training data.
    
    Make sure to shuffle the training data at the beginning of each epoch!
    """
    opt = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.NLLLoss() # since our model gives negative log probs on the output side
        
    for i in range(epochs):
        training_inputs_1,training_inputs_2, training_outputs = shuffle(training_inputs_1, training_inputs_2, training_outputs)
        loss = 0
        
        data=zip(training_inputs_1,training_inputs_2,training_outputs)

        for idx, datum in enumerate(data):
            t1=datum[0]
            t2=datum[1]
            output=datum[2]

            opt.zero_grad()

            t1_tens=vocab.sentence_to_tensor(t1, c2i, True)
            t2_tens=vocab.sentence_to_tensor(t2, c2i, False)
            
            #x_tens = torch.cat((t2_tens,t1_tens),1)
            y_tens=torch.tensor([output])
            
            y_hat,_ = model(t1_tens, t2_tens, model.init_hidden())
            loss += loss_func(y_hat, y_tens)
            
            if idx % 1000 == 0:
                print(f"{idx}/{len(training_inputs_1)}, loss: {loss}")
                
            # send back gradients:
            loss.backward()
            # now, tell the optimizer to update our weights:
            opt.step()
            loss = 0
                
    return model

def eval_acc_binary(model, test_data, c2i, i2c):
    """
    Compute classification accuracy for the test_data against the model.
    
    :param model: The trained model to use
    :param test_data: A list of (x,y) test pairs.
    :returns: The classification accuracy (n_correct / n_total), as well as the predictions
    :rtype: tuple(float, list(str))
    """
    in_col_1='X1'
    in_col_2='X2'
    out_col='y'

    X1_tensor_seq=test_data.loc[:,in_col_1].apply(lambda x: vocab.sentence_to_tensor(x, c2i,True))
    X2_tensor_seq=test_data.loc[:,in_col_2].apply(lambda x: vocab.sentence_to_tensor(x, c2i,False))

    y_int_seq=test_data.loc[:,out_col]

    correct=[]
    labs=[]
    
    data=zip(X1_tensor_seq,X2_tensor_seq,y_int_seq)
    with torch.no_grad():
        for X1, X2, y in data:
            
            out= model.forward(X1, X2, model.init_hidden())[0]
            i=np.argmax(out)

            labs.append(i.item())
            correct.append(i==y)

    yes=0.0
    no=0.0
    total=0.0
    for entry in correct:
        total+=1
        if entry:
            yes+=1
        else:
            no+=1

    return(yes/total,list(labs))

def eval_acc2(model, test_data, c2i, i2c, o2i, i2o):
    """
    Compute classification accuracy for the test_data against the model.
    
    :param model: The trained model to use
    :param test_data: A list of (x,y) test pairs.
    :returns: The classification accuracy (n_correct / n_total), as well as the predictions
    :rtype: tuple(float, list(str))
    """
    in_col_1='X1'
    in_col_2='X2'
    out_col='y'

    X1_tensor_seq=test_data.loc[:,in_col_1].apply(lambda x: vocab.sentence_to_tensor(x, c2i,True))
    X2_tensor_seq=test_data.loc[:,in_col_2].apply(lambda x: vocab.sentence_to_tensor(x, c2i,False))

    y_int_seq=test_data.loc[:,out_col].apply(lambda x: o2i[x])

    correct=[]
    labs=[]
    
    data=zip(X1_tensor_seq,X2_tensor_seq,y_int_seq)
    with torch.no_grad():
        for X1, X2, y in data:
            
            out= model.forward(X1, X2, model.init_hidden())[0]
            i=np.argmax(out)

            labs.append(i2o[i.item()])
            correct.append(i==y)

    yes=0.0
    no=0.0
    total=0.0
    for entry in correct:
        total+=1
        if entry:
            yes+=1
        else:
            no+=1

    return(yes/total,list(labs))

def eval_acc(model, test_data, c2i, i2c, o2i, i2o):
    """
    Compute classification accuracy for the test_data against the model.
    
    :param model: The trained model to use
    :param test_data: A list of (x,y) test pairs.
    :returns: The classification accuracy (n_correct / n_total), as well as the predictions
    :rtype: tuple(float, list(str))
    """
    in_col_1='X1'
    in_col_2='X2'
    out_col='y'

    X1_tensor_seq=test_data.loc[:,in_col_1].apply(lambda x: vocab.sentence_to_tensor(x, c2i,True))
    X2_tensor_seq=test_data.loc[:,in_col_2].apply(lambda x: vocab.sentence_to_tensor(x, c2i,False))

    y_int_seq=test_data.loc[:,out_col].apply(lambda x: o2i[x])

    correct=[]
    labs=[]
    indexes=X1_tensor_seq.index
    data=zip(X1_tensor_seq,X2_tensor_seq,y_int_seq)
    with torch.no_grad():
        for X1, X2, y in data:
            X=torch.cat((X2,X1),1)

            out= model.forward(X, model.init_hidden())[0]
            i=np.argmax(out)

            labs.append(i2o[i.item()])
            correct.append(i==y)

    yes=0.0
    no=0.0
    total=0.0
    for entry in correct:
        total+=1
        if entry:
            yes+=1
        else:
            no+=1

    return(yes/total,list(labs))

def compute_prob(model, sentence, c2i):
    """
    Compute the negative log probability of p(sentence)
    
    Equivalent to equation 3.3 in Jurafsky & Martin.
    """
    
    nll = nn.NLLLoss(reduction='sum')
    
    with torch.no_grad():
        s_tens = vocab.sentence_to_tensor(sentence, c2i, True)
        x = s_tens[:,:-1]
        y = s_tens[:,1:]
        y_hat, _ = model(x, model.init_hidden())
        return nll(y_hat.squeeze(), y.squeeze().long()).item() # get rid of first dimension of each

def pretty_conf_matrix(conf_matrix, classes):
    """
    Make a nice matplotlib figure representing a confusion matrix, as per the Scikit-Learn Confusion Matrix demo
    """
    
    # for color mapping:
    norm_cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(norm_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # now label each square with the counts:
    color_thresh = norm_cm.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        
        plt.text(j, i, str(conf_matrix[i,j]), horizontalalignment="center", 
        color="white" if norm_cm[i,j] > color_thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

