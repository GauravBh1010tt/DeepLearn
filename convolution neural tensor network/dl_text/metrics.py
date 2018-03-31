"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""

from __future__ import division

from operator import itemgetter
from collections import defaultdict
import scipy.stats as measures
import numpy as np

###################### CALCULATING MRR [RETURNS MRR VALUE] ######################

def mrr(out, th = 10):
  n = len(out)
  MRR = 0.0
  for qid in out:
    candidates = out[qid]
    for i in xrange(min(th, len(candidates))):
      if candidates[i] == "true":
        MRR += 1.0 / (i + 1)
        break
  return MRR * 100.0 / n

###################### CALCULATING MAP [RETURNS MAP VALUE] ######################

def map(out, th):
  num_queries = len(out)
  MAP = 0.0
  for qid in out:
    candidates = out[qid]
    avg_prec = 0
    precisions = []
    num_correct = 0
    for i in xrange(min(th, len(candidates))):
      if candidates[i] == "true":
        num_correct += 1
        precisions.append(num_correct/(i+1))
    
    if precisions:
      avg_prec = sum(precisions)/len(precisions)
    
    MAP += avg_prec
  return MAP / num_queries

###################### QUESTION LIST TO DICTIONARY [RETURNS DICTIONARY OF LISTS] ######################

def list2dict(lst):
    interm = defaultdict(list)
    new_pred = defaultdict(list)

    for i in range(len(lst)):
        interm[i+1] = lst[i]
    
    for qid in interm:
        interm[qid] = sorted(interm[qid], key = itemgetter(0), reverse = True)
        val = [rel for score, rel in interm[qid]]
        if 'true' in val:
            new_pred[qid] = val
        else:
            continue
    return new_pred

###################### READING FILE [RETURNS LIST OF LISTS] ######################

def readfile(pred_fname):
    pred = open(pred_fname).readlines()
    pred = [line.split('\t') for line in pred]
    
    ques = []
    ans = []
    i = 1
    
    while i != len(pred):
        if pred[i][0] == pred[i-1][0]:
            ans.append([float(pred[i-1][-2]), pred[i-1][-1][0:-1]])
        else:
            ans.append([float(pred[i-1][-2]), pred[i-1][-1][0:-1]])
            ques.append(ans)
            ans = []
        i += 1
    ans.append([float(pred[i-1][-2]), pred[i-1][-1][0:-1]])
    ques.append(ans)
    
    return ques

###################### MAP AND MRR FUNCTION [RETURNS MAP AND MRR VALUES] ######################

def map_mrr(pred_fname, th = 10):
    ques = readfile(pred_fname)
    dic = list2dict(ques)
    return map(dic,th), mrr(dic,th)

def eval_metric(lrmodel, X_test_l, X_test_r, res_fname,pred_fname,use_softmax=True, feat_test=None):
    
    if feat_test!=None:
        pred = lrmodel.predict([X_test_l, X_test_r, feat_test])[:,1]
    else:
        pred = lrmodel.predict([X_test_l, X_test_r])[:,1]

    ################### SAVING PREDICTIONS ###################
    
    f1 = open(res_fname, 'r').readlines()
    f2 = open(pred_fname,'w')
    
    for j,line in enumerate(f1):
        line = line.split('\t')
        #val = [line[0]+'\t',line[1]+'\t',line[2]+'\t',str(pred[j][0])+'\t',line[-1]]
        if use_softmax:
            val = [line[0]+'\t',line[1]+'\t',line[2]+'\t',str(pred[j])+'\t',line[-1]]
        else:
            val = [line[0]+'\t',line[1]+'\t',line[2]+'\t',str(pred[j][0])+'\t',line[-1]]
        f2.writelines(val)
    
    f2.close()
    
    ################### PRINTING AND SAVING MAP-MRR VALUES ###################
    
    map_val, mrr_val = map_mrr(pred_fname)
    
    #print 'MAP:', map_val
    #print 'MRR:', mrr_val
    return map_val, mrr_val

def eval_sick(model,X_test_l,X_test_r,test_score):
    #r = np.arange(1,6)
    pred = model.predict([X_test_l,X_test_r])*4+1
    pred = [i[0] for i in pred]
    pred = np.array(pred)
    test_score = np.array(test_score)*4+1
    sp_coef = measures.spearmanr(pred,test_score)[0]
    per_coef = measures.pearsonr(pred,test_score)[0]
    mse_coef = np.mean(np.square(pred-test_score))
    
    return sp_coef, per_coef, mse_coef