# ECAPA-TDNN-for-Depression

Code for paper in "ECAPA-TDNN Based Depression Detection from Clinical Speech"

## Environments

The experimental environment is listed here. Using different versions of packages may cause some problems.
```
python==3.7
torch==1.9.0
torchaudio=0.9.0
pandas==1.3.0
numpy==1.20.3
matplotlib==3.4.2
librosa==0.8.1
sklearn==1.0.2
```
## DataSets

our corpus is recored between HAMD interview, only in audio modality. The corresponding corpus is described in detail in our atricle.

The corpus used in this paper is not publicly available at the moment, and this situation may improve in the future as we continue to collect and expand the existing dataset.

## Features

MFCC is used as the feature that input to the neural network. Before feature extractoin, the raw speech data needs to be processed in the following steps:

* The first thing to do is to separate the sound channels. As we collected the data, we save different speakers in different channels, i.e. the doctor's voice in the left channel and the subject's voice in the right channel.
* The next steps is separating the noise from speech, this is implemented by Voice Activity Detection (VAD). The simplest double threshold method is used.
* After processing the above steps, we obtained the subjects' speech of varying lengths. The speech is cut to three seconds with 50% overlap, and then MFCC features are extracted.

## Models

we use ECAPA-TDNN for depression classification in this paper, for its excellent performance in various tasks.

ECAPA-TDNN was adjusted from speaker recognition to speech classification, in a senese, a simplified version. 

for more detials about the models, please refer to [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/abs/2005.07143).

And in the subsequence experiments, the model was found to be significantly better than the models of DepAudioNet, ResNet, X-Vector, etc., not to mention some traditional manual feature methods.

ECAPA-TDNN has been validated to perform well on classification tasks, and we just use it for classification. However, this is also some interesting findings. In our experiemnts, we also extracted the embedding form model, and used T-SNE for visiualized the relationship between embeddings and speakers. we found speech embeddings spoken by one speakers always forming a cluster, and those from depressed speech were more dispersed. This remains to further verification.

## Cited
```
@inproceedings{inproceedings,
  author = {Wang, Dong and Ding, Yanhui and Zhao, Qing and Yang, Peilin and Tan, Shuping and Li, Ya},
  year = {2022},
  month = {09},
  pages = {3333-3337},
  title = {ECAPA-TDNN Based Depression Detection from Clinical Speech},
  doi = {10.21437/Interspeech.2022-10051}
}
```
