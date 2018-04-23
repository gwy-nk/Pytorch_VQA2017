import os
import json
from collections import OrderedDict
from nltk.tokenize import TweetTokenizer
import operator
import base64
import numpy as np
import csv
import sys
from tempfile import TemporaryFile
import _pickle as pickle
import argparse





def image_preprocess():
    csv.field_size_limit(sys.maxsize)

    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    infile = 'trainval_resnet101_faster_rcnn_genome_36.tsv'

    in_data = {}
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]),
                                            dtype=np.float32).reshape((item['num_boxes'], -1))
            in_data[item['image_id']] = item['features']
            break
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]),
                                            dtype=np.float32).reshape((item['num_boxes'], -1))
            in_data[item['image_id']] = item['features']
            break
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]),
                                            dtype=np.float32).reshape((item['num_boxes'], -1))
            in_data[item['image_id']] = item['features']
            break

    print(in_data)
    outfile = TemporaryFile()
    np.save('bottomup_coco.npy', in_data)




def down_vqa_v2():
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip -P zip/')
    os.system('wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P zip/')

    os.system('wget http://visualqa.org/data/abstract_v002/vqa/Questions_Binary_Train2017_abstract_v002.zip -P zip/')
    os.system('wget http://visualqa.org/data/abstract_v002/vqa/Questions_Binary_Val2017_abstract_v002.zip -P zip/')
    os.system('wget http://visualqa.org/data/abstract_v002/vqa/Annotations_Binary_Train2017_abstract_v002.zip -P zip/')
    os.system('wget http://visualqa.org/data/abstract_v002/vqa/Annotations_Binary_Val2017_abstract_v002.zip -P zip/')


    os.system('mkdir coco')
    os.system('mkdir coco/raw')
    os.system('unzip zip/v2_Questions_Train_mscoco.zip -d coco/raw/')
    os.system('unzip zip/v2_Questions_Val_mscoco.zip -d coco/raw/')
    os.system('unzip zip/v2_Annotations_Train_mscoco.zip -d coco/raw/')
    os.system('unzip zip/v2_Annotations_Val_mscoco.zip -d coco/raw/')

    os.system('mkdir abstract')
    os.system('mkdir abstract/raw')
    os.system('unzip zip/Annotations_Binary_Train2017_abstract_v002.zip -d abstract/raw')
    os.system('unzip zip/Annotations_Binary_Val2017_abstract_v002.zip -d abstract/raw')
    os.system('unzip zip/Questions_Binary_Train2017_abstract_v002.zip -d abstract/raw')
    os.system('unzip zip/Questions_Binary_Val2017_abstract_v002.zip -d abstract/raw')

    os.system('rm -rf zip')



def preprocess_q(path,filename,dictname):

    data = json.load(open(path+filename))
    data = data.get('questions')

    wtoi= {}
    itow= {}
    c=1
    tokenizer = TweetTokenizer()
    for i in data:
        tok = tokenizer.tokenize(i.get('question').lower())[:14]
        i['question_tok']= tok
        for n,w in enumerate(tok):
            if w not in wtoi.keys():
                temp=c
                wtoi[w]= c
                itow[(str)(temp)] = w
                c=c+1
    q_dict ={}
    q_dict['wtoi'] = wtoi
    q_dict['itow'] = itow



    with open(dictname, 'wb') as f:
        pickle.dump(q_dict, f)

    return data



def preprocess_a(path, filename):

    data = json.load(open(path+filename))
    data = data.get('annotations')
    a_list=[]
    for i in data:
        dic = {}
        answers = {}
        for j in i.get('answers'):
            if j.get('answer') not in answers.keys():
                answers[j.get('answer')] = 1
            else:
                answers[j.get('answer')] = answers[j.get('answer')]+1

        dic['question_id'] = i.get('question_id')
        l=[]
        newdic = dict(OrderedDict(sorted(answers.items(), key=lambda x: x[1], reverse=True)))
        for a,b in newdic.items():
            temp=[]
            temp.append(a)
            temp.append(b)
            l.append(temp)
        answers =l
        dic['answers'] = answers
        dic['answer_type']=i.get('answer_type')
        dic['answer'] = i.get('multiple_choice_answer')
        a_list.append(dic)
    return a_list





def combine_qa(path,q_list,a_list,savename,dictname):
    sorting_key = operator.itemgetter("question_id")
    for i, j in zip(sorted(q_list, key=sorting_key), sorted(a_list, key=sorting_key)):
        i.update(j)
    count ={}
    for i in q_list:
        if i.get('answer') not in count.keys():
            count[i.get('answer')]= 1
        else:
            count[i.get('answer')] += 1
    index = 1
    itow = {}
    wtoi = {}
    for key in count.keys():
        if count[key] >= 8:
            wtoi[key] = index
            itow[index] = key
            index+=1

    pickle.dump({'itow': itow, 'wtoi': wtoi}, open(dictname, 'wb'))


    for i in q_list:
        sum = 0
        for j in i.get('answers'):
            if j[0] in count.keys() and count[j[0]]>8:
                sum +=j[1]
        answer_weight=[]
        answer_weight2=[]
        accuracy_dict = {}

        for j in i.get('answers'):
            if j[0] in count.keys() and count[j[0]]>8:
                l=[]
                l.append(j[0])
                l.append(j[1]/sum)
                answer_weight.append(l)

            if j[0] in count.keys() and count[j[0]]>8:
                l=[]
                if j[1] ==1:
                    accuracy_dict[j[0]]=1/3
                    l.append(j[0])
                    l.append(0.3)
                elif j[1] == 2:
                    accuracy_dict[j[0]] = 2 / 3
                    l.append(j[0])
                    l.append(0.6)
                elif j[1] >= 3:
                    accuracy_dict[j[0]] = 1
                    l.append(j[0])
                    l.append(1)
                answer_weight2.append(l)

        i['answer_score']= answer_weight
        i['answer_score2']= answer_weight2
        i['accuracy'] =accuracy_dict
    json.dump(q_list, open(savename, 'w'))




if __name__ == '__main__':

    parser =argparse.ArgumentParser(description='Manual data_file generation')
    parser.add_argument('question')




    if not  os.path.exists('coco'):
        print('Downloading VQA-MSCOCO Training,Validation datasets')
        down_vqa_v2()

    if not os.path.exists('coco/coco_train_combined.json'):
        print('Processing coco Taining Data...')
        tq_list = preprocess_q('./coco/raw/','v2_OpenEnded_mscoco_train2014_questions.json','./coco/coco_train_q_dict.p')
        ta_list = preprocess_a('./coco/raw/','v2_mscoco_train2014_annotations.json')
        combine_qa('./coco/raw/',tq_list,ta_list,'./coco/coco_train_combined.json','./coco/coco_train_a_dict.p')

    if not os.path.exists('coco/coco_validation_combined.json'):
        print('Processing coco Validation Data...')
        vq_list = preprocess_q('./coco/raw/','v2_OpenEnded_mscoco_val2014_questions.json','./coco/coco_val_q_dict.p')
        va_list = preprocess_a('./coco/raw/','v2_mscoco_val2014_annotations.json')
        combine_qa('./coco/raw/',vq_list, va_list, './coco/coco_validation_combined.json','./coco/coco_val_a_dict.p')
        os.system('rm -rf ./coco/coco_val_a_dict.p')
        os.system('rm -rf ./coco/coco_val_q_dict.p')

    # if not os.path.exists('abstract_train_combined.json'):
    #     print('Processing abstract Taining Data...')
    #     tq_list = preprocess_q('OpenEnded_abstract_v002_train2017_questions.json','abstract_train_q_dict.p')
    #     ta_list = preprocess_a('abstract_v002_train2017_annotations.json')
    #     combine_qa(tq_list,ta_list,'abstract_train_combined.json','abstract_train_a_dict.p')
    #
    # if not os.path.exists('abstract_validation_combined.json'):
    #     print('Processing abstract Validation Data...')
    #     vq_list = preprocess_q('OpenEnded_abstract_v002_val2017_questions.json','abstract_val_q_dict.p')
    #     va_list = preprocess_a('abstract_v002_val2017_annotations.json')
    #
    #     combine_qa(vq_list, va_list, 'abstract_validation_combined.json','abstract_val_a_dict.p')


    if not os.path.exists('glove.6B.300d.txt'):
        print('Downloading GloVe 300d Pretrained...')
        # os.system('wget http://nlp.stanford.edu/data/glove.840B.300d.zip')
        os.system('unzip glove.840B.300d.zip')


    if not os.path.exists('coco_features.npy'):
        print('Downloading bottom-up Attention Pretrained...')
        #os.system('wget https://storage.googleapis.com/bottom-up-attention/trainval_36.zip')
        # image_preprocess()