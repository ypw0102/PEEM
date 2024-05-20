import copy
import glob
import json
import math
import re
from matplotlib import colors
import numpy
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import operator
import random
from scipy import stats
cmap = colors.LinearSegmentedColormap.from_list(
    'custom_blue_white_red',
    ['blue', 'white', 'red']
)
random.seed(0)

def softmax(S,t):
    sums = 0
    for x in S:
        sums+=math.exp(x/t)
    for i in range(len(S)):
        S[i]=math.exp(S[i]/t)/sums
    return S
def cali(S):
    while True:
        cal_s = S.mean(-1).reshape(1,-1)
        cal_s+=(1-cal_s.mean())
        S = S*cal_s
        pe, p = stats.pearsonr(S.mean(-1), cal_s[0])
        if 1-pe<1e-4:
            return S

def cal_in_loop(list1,list2):
    cnt = 0
    if len(list1)==0 or len(list2)==0:return 0
    for t1 in list1:
        if t1 == '': continue
        for t2 in list2:
            if t2 == '': continue
            if t1 == t2:
                cnt += 1
    return cnt/len(list1)/len(list2)
def calculate_overlap(list1, list2):
    if not isinstance(list1[0], list):
        list1=[[tem] for tem in list1]
    if not isinstance(list2[0], list):
        list2=[[tem] for tem in list2]
    cnt = 0
    for idx in range(len(list1)):
        t1=list1[idx]
        t2=list2[idx]
        cnt+=cal_in_loop(t1,t2)
    return cnt / len(list1)

def integrate_rate_(s):
    keys=list(s.keys())[:-1]
    new_s = []
    for i in range(len(s[keys[0]])):
        cur=[s[key][i] for key in keys]
        pre_dict = {}
        for j in cur:
            if j != '':
                pre_dict[j] = pre_dict.get(j, 0) + 1
        if len(pre_dict) == 0: new_s.append('')
        else:
            res = sorted(pre_dict.items(), key=operator.itemgetter(1), reverse=True)
            new_s.append(res[0][0])
    return new_s
def get_corrs(a,b):
    pe, p = stats.pearsonr(a,b)
    sp, p = stats.spearmanr(a,b)
    ke, p = stats.kendalltau(a,b)
    return {"pe":pe,"sp":sp,"ke":ke}
def vote(s,strategy):
    if 'base' not in strategy:
        new_s = s[:5]
        return new_s
    else:
        new_s = [t for t in s if t!='']
        if len(new_s)==0:return ''
        return random.choice(new_s)

def scale(S,weights):
    weights=weights.reshape(1,-1)/weights.sum()
    S=S*weights*len(S)
    weights_new = S.sum(-1)/len(S)#(len(S)-1)
    weights_new /= weights_new.sum()
    weights = weights.reshape(-1)
    T=abs(weights_new-weights).sum()
    return S,weights_new,T


def strength_filter(overlap_matrix_,drop_list):
    new_dim = len(overlap_matrix_) - len(drop_list)
    S = np.ones((new_dim,new_dim))
    maps = {}
    idx_i = 0
    for i in range(len(overlap_matrix_)):
        if i in drop_list: continue
        maps[idx_i] = i
        idx_i+=1

    for i in range(new_dim):
        for j in range(new_dim):
            S[i,j] = overlap_matrix_[maps[i],maps[j]]

    weights = S.sum(-1) / len(S)
    T = 1
    while T > 1e-2:
        S, weights, T = scale(S, weights)
    weights = softmax(weights, 1 / 2)
    to_drop = np.argmin(weights)
    to_drop = maps[to_drop]
    return to_drop

def cal_consistency(overlap_matrix_,drop_list):
    S,cnt=0,1e-10
    to_drop = drop_list[-1]
    drop_list = drop_list[:max(0,len(drop_list)-1)]
    for i in range(len(overlap_matrix_)):
        if i in drop_list:continue
        for j in range(len(overlap_matrix_)):
            if j in drop_list: continue
            S+=overlap_matrix_[i,j]
            cnt+=1
    old = S/cnt
    for i in range(len(overlap_matrix_)):
        if i in drop_list: continue
        S-=overlap_matrix_[to_drop,i]
        cnt-=1
    new = S/cnt
    return new>old



model_bili=0.7
sample_bili=1
hards = [1,2,3,4,5]
sub_name="test"

modes = ['random',"ensemble","calibration","filtering_0.90","peem"]
for mode in modes:
    to_store_dir = "exp_result/MATH_model-bili-{}_sample-bili-{}_hards-{}_mode-{}_subname-{}.json".format(model_bili,sample_bili,"".join([str(t) for t in hards]),mode,sub_name)
    print("************************")
    print(to_store_dir)
    a_pe_avg, a_sp_avg, a_ke_avg, a_cnt_avg,a_n,a_c = 0,0,0,0,0,0
    # all_models = ['Baichuan-7B_t0','vicuna-7b-v1.5_t0','vicuna-13b-v1.5_t0','Llama2-7b_t0.5','Llama2-13b_t0','MetaMath-7B-V1.0t0.0-n1','Mistral-7B-Instruct-v0.2_t0','Llama2-70b_t0','openchat-3.5-0106_t0','gemini_pro_t0','WizardMath-7B-V1.1_t0','GPT3.5-turbo_t0.5','GPT3.5-turbo_t0','GPT4_t0.5','GPT4_t0','GPT4-turbo_t0']
    all_models_ = ['Baichuan-7B','vicuna-7b-v1.5','vicuna-13b-v1.5','llama-2-7b-chat','Llama-2-13b-chat-hf','Mistral-7B-Instruct-v0.2','Llama-2-70b-chat-hf','MetaMath-7B-V1.0','MetaMath-13B-V1.0','MetaMath-70B-V1.0','openchat-3.5-0106','WizardMath-7B-V1.1','gemini_pro','GPT3.5-turbo','GPT4','gpt4-turbo']
    # all_models_ = [t+"_t0" for t in all_models_]
    # all_models_ = all_models_[:len(all_models_)//2]
    overall_results = {}
    ranks = {all_models_[i]:i for i in range(len(all_models_))}
    dataset=['MATH','coin','common','last_letters','strategy','gsm8k'][0]
    cates=['Intermediate Algebra', 'Prealgebra', 'Geometry', 'Counting & Probability', 'Algebra', 'Number Theory', 'Precalculus']
    # cates = ['Intermediate Algebra']
    # strategy = ["single_cross","single_integration","multiple_cross","multiple_integration","fine"][0]

    raw_datas = {key: [] for key in all_models_}
    for model in all_models_:
        sub_dirs = glob.glob("Result/{}/{}/*".format(dataset, model))
        cur_datas=[]
        for sub_dir in sub_dirs:
            with open(sub_dir, 'r') as f:
                cur = json.load(f)
                cur_datas+=cur
                f.close()
        raw_datas[model]+=cur_datas

    for cate in cates:
        acc,cnt,acc_max,cnt_max=0,0,0,0
        aff=0
        pe_avg, sp_avg, ke_avg, cnt_avg = 0, 0, 0,0
        datas = {key: [] for key in all_models_}
        for model in all_models_:
            for t in raw_datas[model]:
                if t['type']==cate:
                    datas[model].append(t)

        all_rate = {key: {} for key in all_models_}
        all_rate['Accuracy'] = {}
        for model in all_models_:
            # print(model)
            if 'question' in datas[model][0].keys():
                ques = 'question'
            elif 'problem' in datas[model][0].keys():
                ques = 'problem'
            elif 'input' in datas[model][0].keys():
                ques = 'input'
            for i in range(len(datas[model])):
                flag = 0
                for hard_t in hards:
                    if str(hard_t) in datas[model][i]['level']:
                        flag = 1
                        break
                if flag == 0: continue
                rate = vote(datas[model][i]['ans_pre']['predicts'], mode)
                all_rate[model][datas[model][i][ques].strip("Problem:\n").strip("Q:").strip(" ")] = rate
                all_rate['Accuracy'][datas[model][i][ques]] = datas[model][i]['ans_pre']['answer']
        keys = None
        for key in all_rate.keys():
            if keys is None:
                keys = set(all_rate[key].keys())
            else:
                keys = keys & set(all_rate[key].keys())
        cur_keys = list(all_rate.keys())
        all_rate_ = {key: [] for key in cur_keys}
        for key in keys:
            for kk in cur_keys:
                all_rate_[kk].append(all_rate[kk][key])


        sample_idx = random.sample(list(range(len(keys))),int(len(keys)*sample_bili))
        # print(" ")
        for key in all_rate_.keys():
                all_rate_[key] = [all_rate_[key][j] for j in sample_idx]


        np.set_printoptions(precision=10)
        attributes = cur_keys[:-1]
        overlap_matrix_raw = {}
        for i, attr1 in enumerate(attributes):
            for j, attr2 in enumerate(attributes):
                if attr1 not in overlap_matrix_raw.keys():overlap_matrix_raw[attr1] = {}
                overlap_matrix_raw[attr1][attr2] = calculate_overlap(all_rate_[attr1], all_rate_[attr2])

        for ites in trange(500):
            all_models=random.sample(all_models_,int(model_bili*len(all_models_)))
            # all_models=all_models_
            all_rate={key:all_rate_[key] for key in all_models}
            all_rate['Accuracy'] = all_rate_['Accuracy']
            cur_keys = list(all_rate.keys())
            attributes = cur_keys[:-1]
            overlap_matrix = np.ones((len(attributes)+2, len(attributes)))

            for i, attr1 in enumerate(attributes):
                for j, attr2 in enumerate(attributes):
                    overlap_matrix[i, j] = overlap_matrix_raw[attr1][attr2]
            for i, attr in enumerate(attributes):
                overlap_matrix[-2, i] = (overlap_matrix[i].sum()-overlap_matrix[i][i])/(len(overlap_matrix[i])-1)
            for i, attr in enumerate(attributes):
                overlap_matrix[-1, i] = calculate_overlap(all_rate['Accuracy'], all_rate[attr])
                acc+=overlap_matrix[-1, i]
                cnt+=1
            acc_max+=overlap_matrix[-1].max()
            cnt_max+=1

            # strategy
            if mode == "ensemble" or mode == "base":
                pass
            elif mode == 'random':
                idx_sel = random.randint(0,overlap_matrix.shape[1]-1)
                overlap_matrix[-2] = overlap_matrix[idx_sel]
            elif mode == "integration":
                integrate_rate = integrate_rate_(all_rate)
                integrate_list = [calculate_overlap(integrate_rate, all_rate[key]) for key in cur_keys[:-1]]
                for i, attr in enumerate(attributes):
                    overlap_matrix[-2, i] = integrate_list[i]
            elif mode == "calibration":
                overlap_matrix_=copy.deepcopy(overlap_matrix)
                S = copy.deepcopy(overlap_matrix_[:len(attributes)])
                for i in range(len(S)):
                    S[i,i]=0
                weights = S.sum(-1)/(len(S)-1)
                T=1
                while T>1e-2:
                    S, weights, T = scale(S,weights)
                for i, attr1 in enumerate(attributes):
                    for j, attr2 in enumerate(attributes):
                        overlap_matrix_[i, j] = S[i,j]
                weights = softmax(weights, 1/2)
                overlap_matrix[-2]=weights
            elif mode == "most":
                s_index = np.argsort(overlap_matrix[-2])
                overlap_matrix[-2] = overlap_matrix[s_index[-1]]
            elif 'filtering' in mode:
                top_rate = float(mode.split("_")[-1])
                s_index = np.argsort(overlap_matrix[-2])
                max_cons = overlap_matrix[-2][s_index[-1]]
                # max_idx = [s_index[-1],s_index[-2]]
                # for i in range(3,len(s_index)):
                max_idx = [s_index[-1]]
                for i in range(2,len(s_index)):
                    if overlap_matrix[-2][s_index[-i]] < top_rate*max_cons:
                        break
                    max_idx.append(s_index[-i])
                a_n+=len(max_idx)
                a_c+=1
                for i in range(len(attributes)):
                    cur_tem,cur_ctx=0,0
                    for j in max_idx:
                        # if i==j:continue
                        cur_ctx+=1
                        cur_tem+=overlap_matrix[j][i]
                    cur_tem = cur_tem/cur_ctx if cur_ctx!=0 else 1
                    overlap_matrix[-2,i]=cur_tem
            elif 'weem' in mode:
                drop_list = []
                overlap_matrix_ = copy.deepcopy(overlap_matrix)[:-2]
                weights = copy.deepcopy(overlap_matrix[-2])
                while True:
                    to_drop = strength_filter(overlap_matrix_,drop_list)
                    to_continue = cal_consistency(overlap_matrix_,drop_list+[to_drop])
                    if to_continue:
                        drop_list+=[to_drop]
                    else:
                        break

                weights = np.zeros_like(overlap_matrix[-2])
                for i in range(len(attributes)):
                    if i not in drop_list:
                        weights+=overlap_matrix[i]
                weights /= (len(attributes)-len(drop_list))
                overlap_matrix[-2] = weights
                a_n+=len(attributes)-len(drop_list)
                a_c+=1

            cur_res = get_corrs(overlap_matrix[-1], overlap_matrix[-2])
            pe_avg += cur_res['pe']
            sp_avg += cur_res['sp']
            ke_avg += cur_res['ke']
            cnt_avg += 1
            a_pe_avg += cur_res['pe']
            a_sp_avg += cur_res['sp']
            a_ke_avg += cur_res['ke']
            a_cnt_avg += 1


            for i, attr1 in enumerate(attributes):
                overlap_matrix[i] = overlap_matrix[i] / overlap_matrix[i].sum()
            bias_matrix = np.zeros((len(attributes), len(attributes)))
            all_sum_over = overlap_matrix[-1].sum()
            for i, attr1 in enumerate(attributes):
                for j, attr2 in enumerate(attributes):
                    bias_matrix[i, j] = overlap_matrix[i, j] - overlap_matrix[-1, j] / all_sum_over
            aff+=abs(bias_matrix).mean()

        print("Avg Acc:{}   Max Acc:{}    Aff:{}".format(acc/cnt,acc_max/cnt_max,aff/acc_max))
        print("{}  AVG:\nPe/Sp/Ke: {}/{}/{}\n".format(cate,int(1000*pe_avg/cnt_avg),int(1000*sp_avg/cnt_avg),int(1000*ke_avg/cnt_avg)))
        overall_results[cate]={}
        overall_results[cate]["Pe"] = pe_avg / cnt_avg
        overall_results[cate]["Sp"] = sp_avg / cnt_avg
        overall_results[cate]["Ke"] = ke_avg / cnt_avg
    print("All  AVG:\nPe/Sp/Ke: {}/{}/{}\n".format(int(1000*a_pe_avg/a_cnt_avg),int(1000*a_sp_avg/a_cnt_avg),int(1000*a_ke_avg/a_cnt_avg)))
    if a_n==0:
        a_n=int(model_bili*len(all_models_))
        a_c=1
    print("Avg Max Num:{}".format(a_n/a_c))
    overall_results["Overall_Average"]={}
    overall_results["Overall_Average"]["Pe"] = a_pe_avg/a_cnt_avg
    overall_results["Overall_Average"]["Sp"] = a_sp_avg/a_cnt_avg
    overall_results["Overall_Average"]["Ke"] = a_ke_avg/a_cnt_avg

    with open(to_store_dir,"w")as f:
        json.dump(overall_results,f)
        f.close()