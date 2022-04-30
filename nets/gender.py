import os
import torch
import pandas as pd
import torch.nn as nn
from nets.utils import NameVocabulary

class GRU(nn.Module):
  def __init__(self, hidden_dim=128, output_dim=2, n_layers=2, num_words=50, drop_prob = 0.2):
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
      self.load_state_dict(torch.load(os.getcwd()[:-len('telegram-bot')] + os.sep + r'\nets\Pretrained Models\Gender_gpu.pth'))
    else:
      self.device = 'cpu'
      self.load_state_dict(torch.load(os.getcwd()[:-len('telegram-bot')] + os.sep + r'nets\Pretrained Models\Gender_cpu.pth'))

    self.to(self.device)


class GenderPipeline():
  def __init__(self):
    self.model = GRU()
    self.model.load()
    self.model.eval()
    self.vocabulary = NameVocabulary()
    self.label2gender = {0:'Female', 1:'Male'}

  def __call__(self, df:pd.DataFrame):
    df['name_indices'] = df['name'].apply(self.vocabulary.name2indexes)
    results = []

    for index in range(df.shape[0]):
      name = df['name_indices'].iloc[index]
      name = torch.tensor(name).to(self.model.device).view(1,-1)

      _, predicted = torch.max(self.model(name), dim=1)
      predicted = predicted.item()

      results.append(
        self.label2gender[predicted] if predicted in self.label2gender.keys() else 'unknown'
      )

    df.drop(columns = ['name_indices'], inplace=True)
    df['gender'] = results
    return df

if __name__ == '__main__':
  pipeline = GenderPipeline()
  df = pd.read_csv('../example/twitter.csv')
  df = pipeline(df)
  print(df[['name', 'gender']])

