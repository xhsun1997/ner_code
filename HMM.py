import os
import numpy as np
from process_data import *

def read_train_corpus(path):
    with open(path,'r',encoding='utf-8') as f:
        lines=f.readlines()
    sentences=[]
    sentence=[]
    for line in lines:
        if len(line)==0 or line.startswith('-DOCSTART-') or line[0]=="\n":
            if len(sentence)>0:
                sentences.append(sentence)
                sentence=[]
            continue
        line_split=line.strip().split()
        word=line_split[0]
        tag=line_split[-1]
        sentence.append((word,tag))
    if len(sentence)>0:
        sentences.append(sentence)
    return sentences

def get_word_tag_2id(sentences):
    tag_set=set()
    word2id={}
    tag2id={}
    all_words=[]
    for sentence_list in sentences:
        for word_tag in sentence_list:
            assert len(word_tag)==2
            word,tag=word_tag
            tag_set.add(tag)
            all_words.append(word)
    for tag in list(tag_set):
        tag2id[tag]=len(tag2id)
    from collections import Counter
    import operator
    count_words=Counter(all_words)
    sort_list=sorted(count_words.items(),key=operator.itemgetter(1),reverse=True)
    for word,freq in sort_list:
        word2id[word]=len(word2id)
    word2id['unk']=len(word2id)
    return word2id,tag2id

class HMM:
    def __init__(self,word2id,tag2id,sentences):
        self.state_nums=len(tag2id)
        self.all_observation_nums=len(word2id)
        self.word2id=word2id
        self.tag2id=tag2id
        self.train_data=sentences
        self.pi=np.zeros(shape=[self.state_nums])
        self.A=np.zeros(shape=[self.state_nums,self.state_nums])
        self.B=np.zeros(shape=[self.state_nums,self.all_observation_nums])
        self.correct_dict={}
        self.golden_tags_idlist=[]
        self.predict_tags_idlist=[]
        self.id2tag={id_:tag_ for tag_,id_ in tag2id.items()}
    
    def initial_(self):
        for each_sentence_list in self.train_data:
            for step,(observe,state) in enumerate(each_sentence_list[:-1]):#not contain the lastest state and observation
                if step==0:
                    self.pi[self.tag2id[state]]+=1
                current_state=self.tag2id[state]
                assert type(each_sentence_list[step+1])==tuple
                assert each_sentence_list[step+1][1] in self.tag2id
                next_state=self.tag2id[each_sentence_list[step+1][1]]
                self.A[current_state][next_state]+=1
                current_observe=self.word2id[observe]
                self.B[current_state][current_observe]+=1
                if step==len(each_sentence_list)-2:
                    assert each_sentence_list[step+1][1] in self.tag2id
                    current_state=self.tag2id[each_sentence_list[step+1][1]]
                    current_observe=self.word2id[each_sentence_list[step+1][0]]
                    self.B[current_state][current_observe]+=1
        self.A[self.A==0.]=1e-4
        print(self.pi)
        self.pi[self.pi==0.]=1e-4
        self.A/=np.sum(self.A,axis=1,keepdims=True)#axis=1 means sum in row
        self.pi/=np.sum(self.pi)
        self.B[self.B==0]=1e-4
        sum_b=np.sum(self.B,axis=1,keepdims=True)
        assert sum_b.shape==(self.state_nums,1)
        self.B/=sum_b

    def viterbi_decode(self,text,text_tag=None):
        if type(text)==str:
            text_list=text.strip().split()
            assert type(text_list)==list
            for i in text_list:
                assert type(i)==str
        else:
            assert type(text)==list
            text_list=text
            
        text_length=len(text_list)

        delta_arr=np.zeros(shape=[self.state_nums,text_length])
        predict_path=np.zeros(shape=delta_arr.shape,dtype=np.int32)
        
        for i in range(self.state_nums):
            if text_list[0] in self.word2id:
                initial_id=self.word2id[text_list[0]]
            else:
                initial_id=self.word2id['unk']
            delta_arr[i][0]=self.pi[i]*self.B[i][initial_id]
            predict_path[i][0]=0
        for t in range(1,text_length):
            for i in range(self.state_nums):
                temp=[delta_arr[j][t-1]*self.A[j][i] for j in range(self.state_nums)]
                temp_max=max(temp)
                temp_max_idx=temp.index(temp_max)
                if text_list[t] in self.word2id:
                    word_idx=self.word2id[text_list[t]]
                else:
                    word_idx=self.word2id['unk']
                delta_arr[i][t]=temp_max*self.B[i][word_idx]
                predict_path[i][t]=temp_max_idx

        t=text_length-1
        last_column=[delta_arr[i][t] for i in range(self.state_nums)]
        P_=max(last_column)
        last_index=last_column.index(P_)
        output_path=[]
        output_path.append(last_index)
        for t in range(text_length-1,0,-1):
            last_index=predict_path[last_index][t]
            output_path.append(last_index)
        output_path.reverse()
        if text_tag==None:
            return output_path
        correct,total=self.calculate_result(output_path,text_tag)
        return correct,total
    
    
    def calculate_result(self,output_path,text_tag):
        text_id_list=[self.tag2id[tag] for tag in text_tag]
        assert len(output_path)==len(text_id_list)
        self.golden_tags_idlist.append(text_id_list)
        self.predict_tags_idlist.append(output_path)
        assert len(output_path)==len(text_id_list)
        correct_,total_=0,0
        for predict_id,tag_id in zip(output_path,text_id_list):
            predict_tag=self.id2tag[predict_id]
            golden_tag=self.id2tag[tag_id]
            if predict_id==tag_id:
                assert golden_tag==predict_tag
                if golden_tag not in self.correct_dict:
                    self.correct_dict[golden_tag]=1
                else:
                    self.correct_dict[golden_tag]+=1#correct_tag 记录的是每一个被正确预测的标签所出现的测数
                correct_+=1
            total_+=1
        return correct_,total_

def get_test_data(path):
    with open(path,'r',encoding='utf-8') as f:
        lines=f.readlines()
    all_data_list,all_tag_list=[],[]
    data_list,tag_list=[],[]
    for line in lines:
        if len(line)==0 or line.startswith('-DOCSTART-') or line[0]=="\n":
            if len(data_list)>0:
                all_data_list.append(data_list)
                all_tag_list.append(tag_list)
                data_list,tag_list=[],[]
            continue
        line_split=line.strip().split()
        data_list.append(line_split[0])
        tag_list.append(line_split[-1])
    return all_data_list,all_tag_list

if __name__ == "__main__":
    train_path="./data/train.txt"
    os.path.isfile(train_path)
    test_path="./data/train.txt"
    sentences=read_train_corpus(train_path)
    word2id,tag2id=get_word_tag_2id(sentences)
    all_data_list,all_tag_list=get_test_data(test_path)
    model=HMM(word2id,tag2id,sentences)
    model.initial_()
    correct_predict=0
    total_amount=0
    for text_list,text_tag_list in zip(all_data_list,all_tag_list):
        assert len(text_list)==len(text_tag_list)
        correct_pred,total_number=model.viterbi_decode(text_list,text_tag_list)
        correct_predict+=correct_pred
        total_amount+=total_number
    print(correct_predict)
    print(total_amount)
    print(correct_predict/total_amount)
    
    golden_tags_idlist=flatten_list(model.golden_tags_idlist)
    predict_tags_idlist=flatten_list(model.predict_tags_idlist)
    correct_dict=model.correct_dict
    assert len(golden_tags_idlist)==len(predict_tags_idlist)
    from collections import Counter
    golden_tags_idlist_counter=Counter(golden_tags_idlist)#记录的是每一个真实标签应该出现的次数
    predict_tags_idlist_counter=Counter(predict_tags_idlist)#每一个预测的标签出现的次数
    precision_score=cal_precision(tag2id,correct_dict,predict_tags_idlist_counter)
    recall_score=cal_recall(tag2id,correct_dict,golden_tags_idlist_counter)
    f1_score=cal_f1(tag2id,precision_score,recall_score)
    print_scores(tag2id,precision_score,recall_score,f1_score,golden_tags_idlist_counter)
    print("Testing the model is over!")

    
    with open("./model_results/HMM_results.txt","w") as f:
        f.write("tag name"+"\t"+"precision_score"+"\t"+"recall_score"+"\t"+"f1_score"+"\n")
        for tag in tag2id:
            f.write(tag+"\t"+"  "+str(round(precision_score[tag],3))+"\t"+"\t"+str(round(recall_score[tag],3))+"\t"+"\t"+"  "+str(round(f1_score[tag],3))+"\n")
    
    text="Bruce Lee is my favorite star, he borned in Hong Kong and lived in American"
    #text_2="Jet Li is also my idol, and he once played in France."
    predict_path=model.viterbi_decode(text)
    text_list=text.strip().split()
    assert len(predict_path)==len(text_list)
    for predict_tag_id,word in zip(predict_path,text_list):
        predict_tag=model.id2tag[predict_tag_id]
        print(word+"/"+predict_tag)




    
    
