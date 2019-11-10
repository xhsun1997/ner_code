import numpy as np

def read_file(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
    all_sentence,all_sentence_tag=[],[]
    sentence,sentence_tag=[],[]
    for line in lines:
        if len(line)==0 or line.startswith('-DOCSTART-') or line=="\n":
            if len(sentence)>0:
                all_sentence.append(sentence)
                all_sentence_tag.append(sentence_tag)
                sentence,sentence_tag=[],[]
            continue
        line_split=line.strip().split()
        word=line_split[0]
        tag=line_split[-1]
        sentence.append(word)
        sentence_tag.append(tag)
    if len(sentence)>0:
        all_sentence.append(sentence)
        all_sentence_tag.append(sentence_tag)
    return all_sentence,all_sentence_tag

def get_parameter(all_sentence,all_sentence_tag,glove_path):
    with open(glove_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
    word_embedding=[]
    word2id={}
    tag2id={}
    word_dic={}
    for sentence,sentence_tag in zip(all_sentence,all_sentence_tag):
        assert len(sentence)==len(sentence_tag)
        for word,tag in zip(sentence,sentence_tag):
            if tag not in tag2id:
                tag2id[tag]=len(tag2id)
            word_dic[word.lower()]=True

    print(tag2id)
    for line in lines:
        line_split=line.strip().split()
        word=line_split[0]
        vector_string=line_split[1:]
        if len(word2id)==0:
            word2id['UNK']=len(word2id)
            word_embedding.append(np.random.uniform(-0.5,0.5,len(vector_string)))
            word2id['PAD']=len(word2id)
            word_embedding.append(np.zeros(len(vector_string)))
        if word in word_dic:
            word2id[word]=len(word2id)
            vector=[float(num) for num in vector_string]
            word_embedding.append(vector)
    word_embedding=np.array(word_embedding)
    return word2id,tag2id,word_embedding

def sentence_to_id(all_sentence,all_sentence_tag,word2id,tag2id):
    all_sentence_id,all_sentence_tag_id=[],[]
    for sentence,sentence_tag in zip(all_sentence,all_sentence_tag):
        sentence_id,sentence_tag_id=[],[]
        for word,tag in zip(sentence,sentence_tag):
            assert tag in tag2id
            tag_id=tag2id[tag]
            if word not in word2id:#因为word2id中的单词是训练语料库和glove词向量语料库的交集，所以有可能训练语料库中的单词在词向量语料库中没有，这种情况把这类单词看作UNK
                word_id=word2id['UNK']
            else:
                word_id=word2id[word]
            sentence_id.append(word_id)
            sentence_tag_id.append(tag_id)
        all_sentence_id.append(sentence_id)
        all_sentence_tag_id.append(sentence_tag_id)
    return all_sentence_id,all_sentence_tag_id

def pad_sentence_id(all_sentence_id,all_sentence_tag_id,max_seq_length=50):
    pad_all_sentence_id,pad_all_sentence_tag_id=[],[]
    actual_length_list=[]
    for sentence_id,sentence_tag_id in zip(all_sentence_id,all_sentence_tag_id):
        assert len(sentence_tag_id)==len(sentence_id)
        length=len(sentence_id)
        if length>=max_seq_length:
            actual_length_list.append(max_seq_length)
            pad_all_sentence_id.append(sentence_id[:max_seq_length])
            pad_all_sentence_tag_id.append(sentence_tag_id[:max_seq_length])
        else:
            actual_length_list.append(length)
            pad_all_sentence_id.append(sentence_id[:max_seq_length]+[0]*(max_seq_length-length))
            pad_all_sentence_tag_id.append(sentence_tag_id[:max_seq_length]+[0]*(max_seq_length-length))
    return pad_all_sentence_id,pad_all_sentence_tag_id,actual_length_list

def batch_yield(pad_all_sentence_id,pad_all_sentence_tag_id,actual_length_list,batch_size):
    assert len(pad_all_sentence_tag_id)==len(pad_all_sentence_id)==len(actual_length_list)
    shuffled_list=np.random.permutation(len(pad_all_sentence_tag_id))
    assert len(shuffled_list)==len(actual_length_list)
    yield_sentence_id=np.array(pad_all_sentence_id)[shuffled_list]
    yield_sentence_tag_id=np.array(pad_all_sentence_tag_id)[shuffled_list]
    yield_actual_length=np.array(actual_length_list)[shuffled_list]
    num_batches=len(pad_all_sentence_tag_id)//batch_size
    print("the number of batch_size is ",num_batches)
    start=0
    for i in range(num_batches):
        yield yield_sentence_id[start:start+batch_size],yield_sentence_tag_id[start:start+batch_size],yield_actual_length[start:start+batch_size]        
        start+=batch_size




def cal_precision(tag2id,correct_dic,predict_tags_counter):
    precision_scores={}
    for tag,id_ in tag2id.items():
        precision_scores[tag]=correct_dic.get(tag,0)/predict_tags_counter.get(id_,1e-3)
    return precision_scores

def cal_recall(tag2id,correct_dic,golden_tags_counter):
    recall_scores={}
    for tag,id_ in tag2id.items():
        recall_scores[tag]=correct_dic.get(tag,0)/golden_tags_counter.get(id_,1e-3)
    return recall_scores

def cal_f1(tag2id,precision_scores,recall_scores):
    f1_score={}
    for tag in tag2id:
        precision,recall=precision_scores[tag],recall_scores[tag]
        f1_score[tag]=2*precision*recall/(precision+recall+1e-5)
    return f1_score

def flatten_list(input_list):
    output_list=[]
    for li in input_list:
        if type(li)==list:
            output_list+=li
        elif type(li)==np.ndarray:
            for ii in li:
                output_list.append(ii)
    return output_list

def print_scores(tag2id,precision_score,recall_score,f1_score,golden_tags_counter):
    header_format='{:>9s}   {:9}   {:9}   {:9}   {:9}'
    header=['percision','recall','f1_score','support']
    print(header_format.format('',*header))
    row_fromat="{:>9s}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9}"
    for tag,id_ in tag2id.items():
        print(row_fromat.format(tag,precision_score[tag],recall_score[tag],f1_score[tag],golden_tags_counter[id_]))
