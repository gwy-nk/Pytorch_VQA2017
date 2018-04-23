import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, v_size, K, f_dim, h_dim,o_dim,pretrained_we):
        super(Model, self).__init__()
        self.v_size = v_size
        # feature map Dimensions
        self.K = K
        self.f_dim = f_dim
        # GRU cell  h_dim =512, o_dim=3129(according to paper, not sure)
        self.h_dim = h_dim
        self.o_dim=o_dim
        self.gru=nn.GRU(300,self.h_dim)

        #parameters to learn
        #glove WE, Freeze during training
        self.embed = nn.Embedding(v_size, 300)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_we))
        # self.embed.weight.requires_grad = False

        #top-down attention
        self.td_W = nn.Linear(self.f_dim + self.h_dim, self.h_dim)
        self.td_W2 = nn.Linear(self.f_dim + self.h_dim, self.h_dim)
        self.att_w = nn.Linear(h_dim, 1)

        # question embedding, image(2048-512) multimodal representation
        self.q_W = nn.Linear(self.h_dim, self.h_dim)
        self.q_W2 = nn.Linear(self.h_dim, self.h_dim)
        self.i_W = nn.Linear(self.f_dim, self.h_dim)
        self.i_W2 = nn.Linear(self.f_dim, self.h_dim)

        #classification no pretrain
        self.c_W = nn.Linear(self.h_dim, self.h_dim)
        self.c_W2 = nn.Linear(self.h_dim, self.h_dim)
        self.c_Wo =nn.Linear(self.h_dim,self.o_dim)


    def non_linear(self, x, W, W2):
        y_t = F.tanh(W(x))
        g = F.sigmoid(W2(x))
        y = torch.mul(y_t, g)
        return y


    def forward(self, q_batch, i_batch):

        # batch size = 512
        # q -> 512(batch)x14(length)
        # i -> 512(batch)x36(K)x2048(f_dim)
        # one-hot -> glove
        emb = self.embed(q_batch)
        output, hn = self.gru(emb.permute(1, 0, 2))  # (seqlen, batch, hid_dim)
        q_enc = hn.view(-1,self.h_dim)

        # image encoding with l2 norm
        i_enc = F.normalize(input =i_batch, p=2)  # (batch, K, feat_dim)

        # top-down attention
        q_enc_copy = q_enc.repeat(1, self.K).view(-1, self.K, self.h_dim)

        q_i_concat = torch.cat((i_enc, q_enc_copy), -1)
        q_i_concat = self.non_linear(q_i_concat, self.td_W, self.td_W2 )#512 x 36 x 512
        i_attention = self.att_w(q_i_concat)  #512x36x1
        i_attention = F.softmax(i_attention.squeeze(),1)
        #weighted sum
        i_enc = torch.bmm(i_attention.unsqueeze(1), i_enc).squeeze()  # (batch, feat_dim)

        # element-wise (question + image) multiplication
        q = self.non_linear(q_enc, self.q_W, self.q_W2)
        i = self.non_linear(i_enc, self.i_W, self.i_W2)
        h = torch.mul(q, i)  # (batch, hid_dim)

        # output classifier
        #no sigmoid ->BCE with logitsloss
        score = self.c_Wo(self.non_linear(h, self.c_W, self.c_W2))

        return score




