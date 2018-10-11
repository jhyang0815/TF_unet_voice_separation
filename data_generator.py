import numpy as np
import os
import re
import librosa

from random import shuffle
dir='musdb18'

#원하는 루틴: mix -> vocal, accompaniment 나눠 학습.
def get_mix_fname(dir,fname):
    fname=fname+'_mix.wav'
    fpath=os.path.join(dir,fname)
    return fpath

def get_vocal_fname(dir,fname):
    fname = fname + '_vocal.wav'
    fpath = os.path.join(dir, fname)
    return fpath

def get_inst_fname(dir,fname):

    fname = fname + '_accompaniment.wav'
    fpath = os.path.join(dir, fname)
    return fpath

# 이름 보고 파일 정리.
def file_sort(flist):
    outlist=[]
    for i in range(len(flist)):
        name=flist[i].split('_')[0]
        if name in list:
            pass
        else:
            outlist.extend(name)
    return outlist

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_tr_val_list(train):
    train_list=[]
    val_list=[]
    val_idx = np.random.choice(len(train), size=25, replace=False)
    train_idx = [i for i in range(len(train)) if i not in val_idx]
    print("Validation with MUSDB training songs no. " + str(train_idx))
    for i in range(len(val_idx)):
        val_list.extend(train(val_idx[i]))
    for i in range(len(train_idx)):
        train_list.extend(train(train_idx[i]))

    return train_list,val_list

def get_stft(fpath,sr):
    y=librosa.load(fpath,sr=sr,mono=True)
    stft=librosa.core.stft(y,n_fft=2048)
    max=np.max(stft)
    min=np.min(stft)
    stft=minmaxnorm(stft)
    return stft

def minmaxnorm(x):
    max=np.max(x)
    min=np.min(x)
    norm=(x-min)/(max-min)
    return norm

def data_generator(list_files,dir,batch_size):
    framesize=512
    sr=44100
    flist=file_sort(list_files)
    j=0
    while True:
        shuffle(flist)

        for fname in flist:
            inst_fpath=get_inst_fname(dir, fname)
            mix_fpath = get_mix_fname(dir, fname)
            vocal_fpath = get_vocal_fname(dir, fname)
            inst_spec=get_stft(inst_fpath,sr)
            mix_spec = get_stft(mix_fpath, sr)
            vocal_spec = get_stft(vocal_fpath, sr)

            vo_batch=[]
            inst_batch=[]
            mix_batch=[]

            for i in range(batch_size):

                inst_patch=inst_spec[j*i*framesize:(j*i+1)*framesize,:]
                vocal_patch = vocal_spec[j*i * framesize:(j*i + 1) * framesize,:]
                mix_patch = mix_spec[j*i * framesize:(j*i + 1) * framesize,:]

                vo_batch.append(vocal_patch)
                mix_batch.append(mix_patch)
                inst_batch.append(inst_patch)

            j=j+1

            vo_batch = np.array(vo_batch)[:, :, :, np.newaxis]
            mix_batch = np.array(mix_batch)[:, :, :, np.newaxis]
            inst_batch=np.array(inst_batch)[:,:,:,np.newaxis]

            yield vo_batch,inst_batch,mix_batch


if __name__ == "__main__":

    print('test')