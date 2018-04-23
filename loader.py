import json
import _pickle as pickle
import numpy as np




class data_loader:
    def __init__(self, b_size = 512, image_path='data/coco/coco_features.npy', train = True):

        q_dict=pickle.load(open('data/coco/coco_train_q_dict.p','rb'))
        a_dict = pickle.load(open('data/coco/coco_train_a_dict.p', 'rb'))

        self.train = train
        if self.train :
            self.data = json.load(open('data/coco/coco_train_combined.json'),encoding='UTF-8')
        else :
            self.data = json.load(open('data/coco/coco_validation_combined.json'),encoding='UTF-8')



        self.q_wtoi=q_dict['wtoi']
        self.q_itow=q_dict['itow']
        self.a_wtoi=a_dict['wtoi']
        self.a_itow=a_dict['itow']

        self.q_num = len(self.data)
        self.b_size = b_size
        self.n_batch = (int)(self.q_num/b_size)
        self.seq_len = 14
        self.o_dim =len(self.a_itow)+1

        self.i_feat = np.load(image_path, encoding='bytes').item()
        for key in self.i_feat.keys():
            i=self.i_feat.get(key)
            #default K = 36, f_dim = 2048
            self.K=i.shape[0]
            self.f_dim = i.shape[1]
            break

        self.pretrained_we()
        self.data_index=0
        # self.vq_num = len(self.v_data)
        # self.vo_dim =len(self.a_itow)+1
        # self.vn_batch = (int)(self.vq_num / b_size)

        print('Loading done')


    def pretrained_we(self):
        f = open('data/glove.6B.300d.txt', encoding='utf-8')
        # vocab size +1 for <unk>
        self.v_size = len(self.q_wtoi)+1
        we_matrix = np.zeros((self.v_size,300),dtype=np.float32)

        t_dict ={}
        for line in f.readlines():
            s_line= line.split()
            word = s_line[0]
            w_weights = np.asarray(s_line[1:], dtype=np.float32)
            t_dict[word] = w_weights


        for word, index in self.q_wtoi.items():
            try:
                w_weights= t_dict[word]
                we_matrix[index] = w_weights

            except:
                pass
        self.we_matrix = we_matrix


    def get_question(self, index):
        q=[0]*self.seq_len
        for i, w in enumerate(self.data[self.data_index + index]['question_tok']):
            try:
                q[i] = self.q_wtoi[w]
            except:
                q[i] = 0
        return q


    def get_answer(self,b):
        a = np.zeros((len(self.a_itow) + 1), dtype=np.float32)
        for word, score in self.data[self.data_index + b]['answer_score']:
            a[self.a_wtoi[word]] = score


        return a


    def next_batch(self):
        if self.data_index + self.b_size >= self.q_num and self.train:
            self.data_index = 0
            np.random.shuffle(self.data)

        q_batch=[]
        i_batch=[]
        a_batch=[]

        for b in range(self.b_size):
            try:
                q_batch.append(self.get_question(b))
                if self.train:
                    a_batch.append(self.get_answer(b))
                else:
                    a_batch.append(self.data[self.data_index + b]['question_id'])
                i_batch.append(self.i_feat[self.data[self.data_index + b]['image_id']])
            except:
                pass
        self.data_index += self.b_size
        q_batch = np.asarray(q_batch)  # (batch, seqlen)
        a_batch = np.asarray(a_batch)  # (batch, n_answers) or (batch, )
        i_batch = np.asarray(i_batch)  # (batch, feat_dim)
        return q_batch, a_batch, i_batch

    # def vget_question(self, index):
    #     q=[0]*self.seq_len
    #     for i, w in enumerate(self.v_data[self.data_index + index]['question_tok']):
    #         try:
    #             q[i] = self.q_wtoi[w]
    #         except:
    #             q[i] = 0
    #     return q
    #
    # def vget_answer(self,b):
    #     #dic
    #     dic = {}
    #     for word in self.v_data[self.data_index + b]['accuracy'].keys():
    #         try:
    #             dic[self.a_wtoi[word]] = self.v_data[self.data_index + b]['accuracy'].get(word)
    #         except:
    #             pass
    #     return dic
    #
    # def v_next_batch(self):
    #     if self.data_index + self.b_size >= self.vq_num:
    #         self.data_index = 0
    #     vq_batch=[]
    #     vi_batch=[]
    #     va_batch=[]
    #
    #     for b in range(self.b_size):
    #         vq_batch.append(self.vget_question(b))
    #         va_batch.append(self.vget_answer(b))
    #         vi_batch.append(self.i_feat[self.v_data[self.data_index + b]['image_id']])
    #     self.data_index += self.b_size
    #     vq_batch = np.asarray(vq_batch)  # (batch, seqlen)
    #     va_batch = np.asarray(va_batch)  # (batch, n_answers) or (batch, )
    #     vi_batch = np.asarray(vi_batch)  # (batch, feat_dim)
    #     return vq_batch, va_batch, vi_batch

