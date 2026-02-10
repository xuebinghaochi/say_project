from numpy.random import randn
import random
from rnns.data import train_data,test_data
import numpy as np

vocab=list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d unique words found' % vocab_size)
word_to_idx={w:i for i,w in enumerate(vocab)}
idx_to_idx={i:w for i,w in enumerate(vocab)}
# print(word_to_idx['good'])
# print(idx_to_idx[0])

def createInputs(text):
    Input=[]
    for word in text.split(' '):
        v=np.zeros((vocab_size,1))
        v[word_to_idx[word]]=1
        Input.append(v)
    return Input

class RNN:
    def __init__(self,input_size,output_size,hidden_size=64):
        self.Wxh=randn(hidden_size,input_size)/1000
        self.Whh=randn(hidden_size,hidden_size)/1000
        self.Why=randn(output_size,hidden_size)/1000

        self.bh= np.zeros([hidden_size,1])
        self.by= np.zeros([output_size,1])

    def forward(self,Input):
        h=np.zeros((self.Whh.shape[0],1))
        self.last_inputs=Input
        self.last_hs={0:h}
        for i in range(len(Input)):
            h=np.tanh(self.Wxh @ Input[i] + self.Whh @ h + self.bh)
            self.last_hs[i+1]=h
        y=self.Why @ h +self.by
        return y,h

    def backdrop(self, d_y, learn_rate=2e-2):
        n = len(self.last_inputs)
        d_Why = d_y @ self.last_hs[n].T  # dy为n*1，self.last_hs[n]也是n*1,需要转置；dL/dyi=dL/dy[i],yi=Why@h+by的第i个元素（从0开始）yi仅由Why的第i行元素决定，即yi=Whyi0*h0+Whyi1*h1+...，所以dyi/dWhyij=hj，因此dy/dWhy[i][j]可以无歧义地表示dyi/dWhyij=hj,因此dy/dWhy=（n*1）@h转置；dL/dWhyij=dL/dyi*dyi/dWhyij即dL/dy[i]*dy/dWhy[i][j];
        d_by = d_y

        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        d_h = self.Why.T @ d_y

        for i in reversed(range(n)):
            temp = (1 - self.last_hs[i + 1] ** 2) * d_h
            d_bh += temp
            d_Whh += temp @ self.last_hs[i].T
            d_Wxh += temp @ self.last_inputs[i].T
            d_h = self.Whh @ temp

        for d in [d_Whh, d_Why, d_bh, d_h]:
            np.clip(d, -1, 1, out=d)

        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by

def softmax(x):
    return np.exp(x) /np.sum(np.exp(x))

rnn = RNN(vocab_size,2)

# try:
# inputs = createInputs('i am very good')
# out,h =rnn.forward(inputs)
# probs =softmax(out)
# print(probs)

def processData(data,backdrop=True):
    items = list(data.items())
    random.shuffle(items)
    loss=0
    num_correct=0
    for x,y in items:
        Input=createInputs(x)
        target=int(y)
        out,_ = rnn.forward(Input)
        probs = softmax(out)
        loss-=np.log(probs[target])
        num_correct += int(np.argmax(probs)==target)

        if backdrop:
            d_L_d_y=probs
            d_L_d_y[target]-=1
            rnn.backdrop(d_L_d_y)

    return loss/len(data),num_correct/len(data)

for epoch in range(1000):
    train_loss,train_acc=processData(train_data)
    if epoch % 100 == 99:
        print('---Epoch %d---'% (epoch+1))
        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss.item(),train_acc))
        test_loss,test_acc=processData(test_data,backdrop=False)
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss.item(),test_acc))

# for x,y in train_data.items():
#     inputs=createInputs(x)
#     target=int(y)
#     out, _=rnn.forward(inputs)
#     probs=softmax(out)
#     d_L_d_y=probs
#     d_L_d_y[target]-=1
#
#     rnn.backdrop(d_L_d_y)