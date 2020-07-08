import argparse
import json
import cPickle
from collections import defaultdict, Counter
from os.path import dirname, join

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

# from new_dataset import Dictionary, VQAFeatureDataset
from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train
import utils

from vqa_debias_loss_functions import *
from tqdm import tqdm
from torch.autograd import Variable


def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument(
        '--cache_features', default=True,
        help="Cache image features in RAM. Makes things much faster, "
             "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument(
        '--dataset', default='cpv2', help="Run on VQA-2.0 instead of VQA-CP 2.0")
    parser.add_argument(
        '-p', "--entropy_penalty", default=0.36, type=float,
        help="Entropy regularizer weight for the learned_mixin model")
    parser.add_argument(
        '--debias', default="learned_mixin",
        choices=["learned_mixin", "reweight", "bias_product", "none"],
        help="Kind of ensemble loss to use")
    # Arguments from the original model, we leave this default, except we
    # set --epochs to 15 since the model maxes out its performance on VQA 2.0 well before then
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--model_state', type=str, default='logs/exp0/model.pth')
    args = parser.parse_args()
    return args

def compute_score_with_logits(logits, labels):
    # logits = torch.max(logits, 1)[1].data # argmax
    logits = torch.argmax(logits,1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def evaluate(model,dataloader,qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0
    model.train(False)
    # import pdb;pdb.set_trace()


    for v, q, a, b,qids,hintscore in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        pred, _ ,_= model(v, q, None, None,None)
        batch_score= compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid=qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number
    print('\teval overall score: %.2f' % (100 * score))
    print('\teval up_bound score: %.2f' % (100 * upper_bound))
    print('\teval y/n score: %.2f' % (100 * score_yesno))
    print('\teval other score: %.2f' % (100 * score_other))
    print('\teval number score: %.2f' % (100 * score_number))

def evaluate_ai(model,dataloader,qid2type,label2ans):
    score=0
    upper_bound=0

    ai_top1=0
    ai_top2=0
    ai_top3=0

    for v, q, a, b, qids, hintscore in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda().float().requires_grad_()
        q = Variable(q, requires_grad=False).cuda()
        a=a.cuda()
        hintscore=hintscore.cuda().float()
        pred, _, _ = model(v, q, None, None, None)
        vqa_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), v, create_graph=True)[0]  # [b , 36, 2048]

        vqa_grad_cam=vqa_grad.sum(2)
        sv_ind=torch.argmax(vqa_grad_cam,1)

        x_ind_top1=torch.topk(vqa_grad_cam,k=1)[1]
        x_ind_top2=torch.topk(vqa_grad_cam,k=2)[1]
        x_ind_top3=torch.topk(vqa_grad_cam,k=3)[1]

        y_score_top1 = hintscore.gather(1,x_ind_top1).sum(1)/1
        y_score_top2 = hintscore.gather(1,x_ind_top2).sum(1)/2
        y_score_top3 = hintscore.gather(1,x_ind_top3).sum(1)/3


        batch_score=compute_score_with_logits(pred,a.cuda()).cpu().numpy().sum(1)
        score+=batch_score.sum()
        upper_bound+=(a.max(1)[0]).sum()
        qids=qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            if batch_score[j]>0:
                ai_top1 += y_score_top1[j]
                ai_top2 += y_score_top2[j]
                ai_top3 += y_score_top3[j]



    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    ai_top1=(ai_top1.item() * 1.0) / len(dataloader.dataset)
    ai_top2=(ai_top2.item() * 1.0) / len(dataloader.dataset)
    ai_top3=(ai_top3.item() * 1.0) / len(dataloader.dataset)

    print('\teval overall score: %.2f' % (100 * score))
    print('\teval up_bound score: %.2f' % (100 * upper_bound))
    print('\ttop1_ai_score: %.2f' % (100 * ai_top1))
    print('\ttop2_ai_score: %.2f' % (100 * ai_top2))
    print('\ttop3_ai_score: %.2f' % (100 * ai_top3))
    
def main():
    args = parse_args()
    dataset = args.dataset


    with open('util/qid2type_%s.json'%args.dataset,'r') as f:
        qid2type=json.load(f)

    if dataset=='cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    elif dataset=='cpv2' or dataset=='v2':
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                  cache_image_features=args.cache_features)

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()

    if args.debias == "bias_product":
        model.debias_loss_fn = BiasProduct()
    elif args.debias == "none":
        model.debias_loss_fn = Plain()
    elif args.debias == "reweight":
        model.debias_loss_fn = ReweightByInvBias()
    elif args.debias == "learned_mixin":
        model.debias_loss_fn = LearnedMixin(args.entropy_penalty)
    else:
        raise RuntimeError(args.mode)


    model_state = torch.load(args.model_state)
    model.load_state_dict(model_state)


    model = model.cuda()
    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # The original version uses multiple workers, but that just seems slower on my setup
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)



    print("Starting eval...")

    evaluate(model,eval_loader,qid2type)



if __name__ == '__main__':
    main()
