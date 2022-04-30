import os
import torch
import pandas as pd
import torch.nn as nn
from nets.utils import Vocabulary


class GRU(nn.Module):
  def __init__(self, hidden_dim=128, output_dim=3, n_layers=2, num_words=29275, drop_prob = 0.2):
    super().__init__()
    self.embedding = nn.Embedding(num_words, hidden_dim)
    self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first = True, dropout = drop_prob)
    self.fc = nn.Linear(hidden_dim, output_dim)
    self.ReLU = nn.ReLU()

  def forward(self, x):
    x = self.embedding(x)
    out, h = self.gru(x)
    out = self.fc(self.ReLU(out[:, -1]))
    return out

  def load(self, ondevice=True):
    if torch.cuda.is_available() and ondevice:
      self.device = 'cuda'
      self.load_state_dict(torch.load(os.getcwd()[:-len('telegram-bot')] + os.sep + r'nets\Pretrained Models\SentimentAnalysis_gpu.pth'))
    else:
      self.device = 'cpu'
      self.load_state_dict(torch.load(os.getcwd()[:-len('telegram-bot')] + os.sep + r'nets\Pretrained Models\SentimentAnalysis_cpu.pth'))

    self.to(self.device)


class SentimentPipeline():
  def __init__(self):
    self.model = GRU()
    self.model.load()
    self.model.eval()
    self.vocabulary = Vocabulary()
    self.vocabulary.fromJson()
    self.label2sentiment = {0: 'Positive',  1:'Neutral', 2:'Negative'}

  def __call__(self, df:pd.DataFrame):
    df['text_indices'] = df['text'].apply(self.vocabulary.sentence2indices)
    results, values = [], []

    for index in range(df.shape[0]):
      sentence = df['text_indices'].iloc[index]
      sentence = torch.tensor(sentence).to(self.model.device).view(1,-1)
      value, predicted = torch.max(self.model(sentence), dim=1)
      predicted = predicted.item()
      results.append(
        self.label2sentiment[predicted] if predicted in self.label2sentiment.keys() else 'Neutral'
      )
      values.append(value.item())

    df.drop(columns=['text_indices'], inplace=True)
    df['sentiment'] = results
    df['sentiment_value'] = values
    return df

if __name__ == '__main__':
  pipeline = SentimentPipeline()
  df = pd.read_csv('../example/twitter.csv')
  pipeline(df)
  print(df)