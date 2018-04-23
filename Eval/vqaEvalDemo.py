# coding: utf-8

import sys
import argparse
from Eval.vqa import VQA
from Eval.vqaEval import VQAEval
import json


def vqaeval(args):

    quesFile = args.question
    annFile = args.annotation
    resFile = args.result

    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
    vqaEval.evaluate()
    json.dump(vqaEval.accuracy,     open(resFile[:-4]+'eval.json',     'w'))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='VQA Evaluation')
    parser.add_argument('--result',type=str,help='result json file')
    parser.add_argument('--question',type=str,help='question file to evaluate',default= './data/coco/raw/v2_OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--annotation',type=str,help='annotation file to evaluate',default= './data/coco/raw/v2_mscoco_val2014_annotations.json')
    args, unparsed = parser.parse_known_args()
    vqaeval(args)
