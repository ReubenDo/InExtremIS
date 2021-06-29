#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

score_dice = []
path_model = 'models/full_annotations/fully_{}/results_full.csv'
score_hd = []
for i in [1,3,5,7,9,11,13]:
    score_dice.append(100*pd.read_csv(path_model.format(i))['dice'].mean())
    score_hd.append(pd.read_csv(path_model.format(i))['hd95'].mean())

dice_InExtremIS = 100*pd.read_csv('models/extreme_points/manual_gradient_eucl_deep_crf/results_full.csv')['dice'].mean()
hd_InExtremIS = pd.read_csv('models/extreme_points/manual_gradient_eucl_deep_crf/results_full.csv')['hd95'].mean()
times = [2,6,10,14,18,22,26]

scores = {
    'Dice Score':{'unit':'%', 'scores':[score_dice, dice_InExtremIS]}, 
    '95th-percentile Hausdorff Distance':{'unit':'mm', 'scores':[score_hd, hd_InExtremIS]}
    }

for metric, score in scores.items():
    unit = score['unit']
    supervised_scores, InExtreMIS_score = score['scores']
    fig = plt.figure()
    plt.title('Full supervision accuracy given an annotation time budget')
    plt.ylabel(f'{metric} ({unit})')
    plt.xlabel('Annotation time budget in hours')

    plt.plot(times, supervised_scores, '-ok', label='Fully Supervised')
    plt.scatter([2], [InExtreMIS_score], c='blue', label='Extreme points Supervision')
    plt.xticks(np.arange(0, 26+1, 2.0))
    
    if metric == 'Dice Score':
        shift = 0.1
        sign = 1
    else:
        shift = 0.01
        sign = -1

    diff_same_budget = round(InExtreMIS_score-supervised_scores[0],1)
    diff_same_score = times[[n for n,i in enumerate(supervised_scores) if sign*i>sign*InExtreMIS_score][0]] - 2

    plt.annotate(s='', xy=(diff_same_score+2.1,InExtreMIS_score), xytext=(2.1,InExtreMIS_score), arrowprops=dict(arrowstyle='<->', linestyle="--",linewidth=2, color='purple'))
    plt.annotate(s='', xy=(2,supervised_scores[0]), xytext=(2,InExtreMIS_score), arrowprops=dict(arrowstyle='<->', linestyle="--",linewidth=2, color='purple'))
    
    plt.text(diff_same_score/2, InExtreMIS_score+5*shift, f'{diff_same_score} hours', ha='left', va='center',color='purple')
    plt.text(1.2, InExtreMIS_score-diff_same_budget/2, f'{abs(diff_same_budget)}{unit} {metric}', ha='left',rotation=90, va='center', color='purple')
    plt.legend()
    plt.grid()
    plt.show()
    fig.savefig(f"figs/{metric.replace(' ','')}_comparision.pdf",bbox_inches='tight')