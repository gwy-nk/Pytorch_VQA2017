from Eval.vqa import VQA
from Eval.vqaEval import VQAEval
import json
import torch
from torch.autograd.variable import Variable
from loader import data_loader
from model import Model
import os
import argparse

def json_dump(args):
    # Some preparation
    torch.cuda.manual_seed(1000)
    print ('Loading data')
    if args.imgf_path ==None:
        #default bottom-up top-down
        loader = data_loader(b_size=512, train=False)
    else:
        loader = data_loader(b_size=512,image_path=args.imgf_path, train=False)
    model = Model(v_size=loader.v_size,
                  K=loader.K,
                  f_dim=loader.f_dim,
                  h_dim=512,
                  o_dim=loader.o_dim,
                  pretrained_we=loader.we_matrix)

    model = model.cuda()
    if args.savedmodel and os.path.isfile(args.savedmodel):
        print('Reading Saved model {}'.format(args.savedmodel))
        ckpt = torch.load(args.savedmodel)
        model.load_state_dict(ckpt['state_dict'])
    else:
        print('Wrong Modelpath')

    result = []
    for step in range(loader.n_batch+1):
        # Batch preparation
        q_batch, a_batch, i_batch = loader.next_batch()
        q_batch = Variable(torch.from_numpy(q_batch))
        i_batch = Variable(torch.from_numpy(i_batch))
        if step == loader.n_batch+1:
            q_batch = Variable(torch.from_numpy(q_batch))[:loader.q_num-loader.n_batch*loader.b_size]
            i_batch = Variable(torch.from_numpy(i_batch))[:loader.q_num-loader.n_batch*loader.b_size]
            a_batch = Variable(torch.from_numpy(a_batch))[:loader.q_num-loader.n_batch*loader.b_size]

        q_batch, i_batch = q_batch.cuda(), i_batch.cuda()
        output = model(q_batch, i_batch)
        _, ix = output.data.max(1)
        for i, qid in enumerate(a_batch):
            result.append({
                'question_id': (int)(qid),
                'answer': loader.a_itow[ix[i]]
            })

    outfile = open(args.savedmodel+'result.json','w')
    json.dump(result,outfile)
    print ('Validation done')

def vqaeval(args):

    if args.abstract:
        pass
    else:
        print('Validating MScoco model')
        quesFile = './data/coco/raw/v2_OpenEnded_mscoco_val2014_questions.json'
        annFile = './data/coco/raw/v2_mscoco_val2014_annotations.json'
    if args.jsoneval:
        resFile = args.savedmodel
    else:
        resFile = args.savedmodel + 'result.json'
    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)
    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
    vqaEval.evaluate()
    json.dump(vqaEval.accuracy,     open(resFile[:-4]+'_eval.json',     'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQA Result Evaluation ')
    parser.add_argument('--savedmodel', type=str, help='saved model path')
    parser.add_argument('--jsoneval',default=False,action='store_true')
    parser.add_argument('--imgf_path', type=str, help='saved image feature path', default=None)
    parser.add_argument('--abstract',default=False, action='store_true')
    args, unparsed = parser.parse_known_args()

    if args.jsoneval:
        vqaeval(args)

    else:
        json_dump(args)
        vqaeval(args)
        # manual_eval(args)
