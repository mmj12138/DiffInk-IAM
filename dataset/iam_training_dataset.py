"""Per-instance IAM vocabulary and worker-local HDF5 handles for VAE training."""
import json
from pathlib import Path
import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class IAMTrainingDataset(Dataset):
    def __init__(self, h5_path, text_file, writer_file, training):
        self.path=str(h5_path)
        self.training=training
        self.vocab={c:i for i,c in enumerate(json.loads(Path(text_file).read_text()))}
        writers=json.loads(Path(writer_file).read_text())
        self.writers={w:i for i,w in enumerate(writers['train'])}
        self.keys=[]
        self._hf=None
        with h5py.File(self.path,'r') as hf:
            for key,g in hf.items():
                text=g['line_text'].asstr()[()];n=len(g['point_seq'])
                # Original 200-point minimum intentionally disabled.
                if len(text)<2 or (training and n>2000):continue
                if n==0 or set(text)-self.vocab.keys():raise ValueError(f'Invalid text/points: {key}')
                writer=g['writer_id'].asstr()[()]
                if training and writer not in self.writers:raise ValueError(f'Unknown training writer: {writer}')
                self.keys.append(key)
        if not self.keys:raise ValueError(f'Empty dataset: {self.path}')

    def __len__(self):return len(self.keys)

    def __getitem__(self,index):
        if self._hf is None:self._hf=h5py.File(self.path,'r')
        key=self.keys[index];g=self._hf[key]
        text=g['line_text'].asstr()[()];writer=g['writer_id'].asstr()[()]
        return {'key':key,'text':text,'writer_id':writer,
                'writer_label':self.writers[writer] if self.training else -1,
                'points':torch.from_numpy(g['point_seq'][:]),
                'labels':torch.tensor([self.vocab[c] for c in text],dtype=torch.long)}

    def __getstate__(self):
        state=self.__dict__.copy();state['_hf']=None;return state

    def close(self):
        if self._hf is not None:self._hf.close();self._hf=None


def collate_iam(batch):
    lengths=torch.tensor([len(b['points']) for b in batch],dtype=torch.long)
    length=((int(lengths.max())+7)//8)*8
    data=torch.tensor([0,0,0,0,1],dtype=torch.float32).repeat(len(batch),length,1)
    for i,b in enumerate(batch):data[i,:len(b['points'])]=b['points']
    mask=torch.arange(length).unsqueeze(0)<lengths.unsqueeze(1)
    return {'data':data,'mask':mask,'labels':pad_sequence([b['labels'] for b in batch],batch_first=True,padding_value=-1),
            'writers':torch.tensor([b['writer_label'] for b in batch]),
            'keys':[b['key'] for b in batch],'texts':[b['text'] for b in batch]}
