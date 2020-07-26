import torch
import torch.nn as nn
from models.IMRAM import frame_by_word

class scdm(nn.Module):
    def __init__(self,opt):
        super().__init__()

        self.opt = opt

        self.ga = nn.Linear(opt.joint_dim, opt.joint_dim)
        nn.init.xavier_uniform_(self.ga.weight)
        nn.init.zeros_(self.ga.bias)

        self.de = nn.Linear(opt.joint_dim, opt.joint_dim)
        nn.init.xavier_uniform_(self.de.weight)
        nn.init.zeros_(self.de.bias)

    def forward(self,video,mask,sentence,sentence_mask):

        b,c,d,t = video.size()
        video = video.permute(0,2,3,1).view(b,-1,c) # ->[b,d*t,c]
        sentence = frame_by_word(video,mask.view(1,-1),sentence,sentence_mask,self.opt)

        gama = torch.tanh(self.ga(sentence))
        deta = torch.tanh(self.de(sentence))

        mean = torch.mean(video,dim=(1,2),keepdim=True)
        var = torch.var(video,dim=(1,2),keepdim=True) + 1e-6
        video = gama*(video-mean)/var+deta

        video = video.permute(0,2,1).view(b,c,d,t)

        return video