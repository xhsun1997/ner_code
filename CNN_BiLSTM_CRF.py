import numpy as np
import tensorflow as tf
import os
from process_data import *

os.environ['CUDA_VISIBLE_DEVICES']='0'
config_gpu=tf.ConfigProto()
config_gpu.gpu_options.allow_growth=True

def get_char2id(all_sentence):
    char2id={}
    for sentence in all_sentence:
        for word in sentence:
            for char in word:
                char2id[char]=len(char2id)
    return char2id 

def char_sentence_to_id(all_sentence,char2id):#将每一句话的每一个字符转换成char2id中的下标索引
    all_charsentence_id=[]
    for sentence in all_sentence:
        sentence_id=[]
        for word in sentence:
            word_id=[]
            for char in word:
                if char not in char2id:
                    char_id=char2id['UNK']
                else:
                    char_id=char2id[char]
                word_id.append(char_id)
            sentence_id.append(word_id)
        all_charsentence_id.append(sentence_id)
    assert len(all_charsentence_id)==len(all_sentence)
    return all_charsentence_id

def pad_charsentence_id(all_charsentence_id,max_seq_length,max_word_length):
    padded_char_matrix=np.empty(shape=[len(all_charsentence_id),max_seq_length,max_word_length],dtype=np.int32)
    for i in range(len(all_charsentence_id)):
        charsentence_id=all_charsentence_id[i]
        seq_length=len(charsentence_id)
        for j in range(min(seq_length,max_seq_length)):
            char_word_id=charsentence_id[j]
            word_length=len(char_word_id)
            for k in range(min(word_length,max_word_length)):
                char_id=char_word_id[k]
                padded_char_matrix[i,j,k]=char_id
            padded_char_matrix[i,j,min(word_length,max_word_length):]=0
        padded_char_matrix[i,min(seq_length,max_seq_length):,:]=0
    return padded_char_matrix

class CNN_BiLSTM_CRF:
    def __init__(self,word_embedding,char_embedding_matrix,tag2id,config,use_CRF=True):
        self.word_embedding=word_embedding
        self.char_embedding=char_embedding_matrix
        self.config=config
        self.embedding_dim=self.word_embedding.shape[1]
        self.embedding_size=self.word_embedding.shape[0]
        self.hidden_dim=self.config['hidden_dim']
        self.batch_size=self.config['batch_size']
        self.max_seq_length=self.config['max_seq_length']
        self.max_word_length=self.config['max_word_length']
        self.char_embedding_dim=self.config['char_embedding_dim']
        self.filter_size=self.config['filter_size']
        self.char_feature_size=self.config['char_feature_size']
        self.num_tags=len(tag2id)
        self.tag2id=tag2id
        self.epoches=config['epoches']
        self.use_CRF=use_CRF
        self.model_save_path=config['model_save_path']
        tf.reset_default_graph()
        print("Model has been initialed!")
        
    def placeholder_layer(self):
        self.word_ids=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_seq_length])
        self.char_ids=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_seq_length,self.max_word_length])
        self.label_ids=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_seq_length])
        self.seq_length=tf.placeholder(dtype=tf.int32,shape=[self.batch_size])  
        print("placeholder layer is over!")  
    
    def embedding_layer(self):
        word_embed=tf.constant(value=self.word_embedding,dtype=tf.float32)
        char_embed=tf.constant(value=self.char_embedding,dtype=tf.float32)
        self.word_embed=tf.nn.embedding_lookup(params=word_embed,ids=self.word_ids)
        assert self.word_embed.shape==(self.batch_size,self.max_seq_length,self.embedding_dim)
        self.char_embed=tf.nn.embedding_lookup(params=char_embed,ids=self.char_ids)
        assert self.char_embed.shape==(self.batch_size,self.max_seq_length,self.max_word_length,self.char_embedding_dim)
        print("embedding_layer is over!")
    
    def CNN_layer(self):
        cnn_inputs=tf.reshape(tensor=self.char_embed,shape=[self.batch_size*self.max_seq_length,self.max_word_length,self.char_embedding_dim])
        cnn_inputs=tf.expand_dims(cnn_inputs,axis=1)
        assert cnn_inputs.shape==(self.batch_size*self.max_seq_length,1,self.max_word_length,self.char_embedding_dim)
        #仅仅是在max_word_length这一个维度上做卷积 所以weight_height=1,weight_width=filter_size
        weights_cnn=tf.Variable(tf.random_normal(shape=[1,self.filter_size,self.char_embedding_dim,
                                                        self.char_feature_size],dtype=tf.float32))
        conv2d=tf.nn.conv2d(cnn_inputs,weights_cnn,strides=[1,1,1,1],padding="VALID")
        reduce_scale=self.max_word_length-self.filter_size+1
        assert conv2d.shape==(self.batch_size*self.max_seq_length,1,reduce_scale,self.char_feature_size)
        pool_=tf.nn.max_pool(tf.tanh(conv2d),ksize=[1,1,reduce_scale,1],strides=[1,1,1,1],padding="VALID")
        cnn_outputs=tf.reshape(pool_,shape=[self.batch_size,self.max_seq_length,self.char_feature_size])
        bilstm_inputs=tf.concat(values=[self.word_embed,cnn_outputs],axis=-1)
        self.BiLSTM_inputs=tf.nn.dropout(bilstm_inputs,keep_prob=0.5)
        assert self.BiLSTM_inputs.shape==(self.batch_size,self.max_seq_length,self.embedding_dim+self.char_feature_size)
        #concat char-level and word level representation and feed it into BiLSTM layer
        print("CNN layer is over!")
        
    def BiLSTM_layer(self):
        cell_fw=tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        cell_bw=tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,self.BiLSTM_inputs,dtype=tf.float32,
                                                       sequence_length=self.seq_length)
        outputs_fw,outputs_bw=outputs
        assert outputs_fw.shape==(self.batch_size,self.max_seq_length,self.hidden_dim)
        output_concat=tf.concat(values=[outputs_fw,outputs_bw],axis=-1)
        assert output_concat.shape==(self.batch_size,self.max_seq_length,self.hidden_dim*2)
        
        fully_weights=tf.Variable(tf.random_normal(shape=[self.batch_size,self.hidden_dim*2,self.num_tags],dtype=tf.float32))
        fully_biases=tf.Variable(tf.random_normal(shape=[self.num_tags],dtype=tf.float32))
        predict=tf.matmul(output_concat,fully_weights)+fully_biases
        self.logits=tf.nn.dropout(predict,keep_prob=0.5)
        assert self.logits.shape==(self.batch_size,self.max_seq_length,self.num_tags)
        print("BiLSTM layer is over!")
    
    def loss_layer(self):
        if self.use_CRF:
            log_likelihood,self.transition_matrix=tf.contrib.crf.crf_log_likelihood(self.logits,self.label_ids,self.seq_length)
            self.loss=tf.reduce_mean(-log_likelihood)
        
        else:
            softmax_losses=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.label_ids)
            mask=tf.sequence_mask(self.seq_length,maxlen=self.max_seq_length)
            self.loss=tf.reduce_mean(tf.boolean_mask(tensor=softmax_losses,mask=mask))
        self.train_op=tf.train.AdamOptimizer(0.01).minimize(self.loss)
        print("loss layer is over!")
    
    def build_graph(self):
        self.placeholder_layer()
        self.embedding_layer()
        self.CNN_layer()
        self.BiLSTM_layer()
        self.loss_layer()
        print("The graph has been built!")
    
    def train(self,pad_all_sentence_id, pad_all_sentence_tag_id, actual_length_list, padded_char_matrix):
        saver=tf.train.Saver(max_to_keep=1)
        num_batches=len(actual_length_list)//self.batch_size
        with tf.Session(config=config_gpu) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoches):
                batches=batch_yield_(pad_all_sentence_id, pad_all_sentence_tag_id, actual_length_list, padded_char_matrix,self.batch_size)
                total_loss=0.0
                for step,(batch_sentence_id,batch_sentence_tag_id,batch_actual_length,batch_char_matrix) in enumerate(batches):
                    assert batch_sentence_id.shape==self.word_ids.shape==batch_sentence_tag_id.shape
                    assert self.char_ids.shape==batch_char_matrix.shape
                    
                    feed_dict={self.word_ids:batch_sentence_id,self.label_ids:batch_sentence_tag_id,
                               self.char_ids:batch_char_matrix,self.seq_length:batch_actual_length}
                    _,loss_val=sess.run([self.train_op,self.loss],feed_dict)
                    total_loss+=loss_val
                if epoch%10==0:
                    print("loss value is ",total_loss/num_batches)
                    saver.save(sess,self.model_save_path)
                    print("It has train one epoch and save the model!")
    
    def test(self,pad_all_sentence_id_test,pad_all_sentence_tag_id_test,actual_length_list_test,padded_char_matrix_test):
        saver=tf.train.Saver()
        correct_num,total_num=0,0
        correct_dic={}#correct记录的是每一个标签被正确预测的次数
        id2tag={idx:tag for tag,idx in self.tag2id.items()}
        print(id2tag)
        golden_tags_list,predict_tags_list=[],[]
        with tf.Session(config=config_gpu) as sess:
            saver.restore(sess,self.model_save_path)
            batches=batch_yield_(pad_all_sentence_id_test,pad_all_sentence_tag_id_test,actual_length_list_test,padded_char_matrix_test,self.batch_size)
            for batch_sentence_id,batch_sentence_tag_id,batch_actual_length,batch_char_matrix in batches:
                feed_dict={self.word_ids:batch_sentence_id,self.char_ids:batch_char_matrix,self.seq_length:batch_actual_length}
                
                if self.use_CRF:
                    viterbi_seq,viterbi_score=tf.contrib.crf.crf_decode(self.logits,self.transition_matrix,sequence_length=self.seq_length)
                    predict_result=sess.run(viterbi_seq,feed_dict)
                else:
                    predict_argmax=tf.argmax(self.logits,axis=-1)
                    assert predict_argmax.shape==(self.batch_size,self.max_seq_length)
                    predict_result=tf.cast(predict_argmax,dtype=tf.int32)
                    predict_result=sess.run(predict_result,feed_dict)
                    
                assert predict_result.shape==batch_sentence_tag_id.shape
                for predict_tag_list,golden_tag_list,length in zip(predict_result,batch_sentence_tag_id,batch_actual_length):
                    assert len(predict_tag_list)==len(golden_tag_list)==self.max_seq_length
                    golden_tags_list.append(golden_tag_list[:length])#golden_tags_list记录的所有真实的标签的相应idx
                    predict_tags_list.append(predict_tag_list[:length])#predict_tags_list记录的是所有预测的标签的相应idx
                    for predict_tag,golden_tag in zip(predict_tag_list[:length],golden_tag_list[:length]):
                        predict_tag_=id2tag[predict_tag]
                        golden_tag_=id2tag[golden_tag]
                        if predict_tag==golden_tag:#predict_tag and golden_tag is idx
                            correct_num+=1
                            assert predict_tag_==golden_tag_
                            if predict_tag_ not in correct_dic:
                                correct_dic[predict_tag_]=1
                            else:
                                correct_dic[predict_tag_]+=1
                        total_num+=1
            print("correct num/total num is ",correct_num/total_num)
            golden_tags_list=flatten_list(golden_tags_list)
            predict_tags_list=flatten_list(predict_tags_list)
            assert len(golden_tags_list)==len(predict_tags_list)#一个列表，每一个元素是一个单词对应的标签的idx
            from collections import Counter
            predict_tags_counter=Counter(predict_tags_list)#记录的是每一个预测的标签被预测的次数
            golden_tags_counter=Counter(golden_tags_list)#记录的是测试集中每一个golden tags应该出现的次数
            precision_score=cal_precision(self.tag2id,correct_dic,predict_tags_counter)
            recall_score=cal_recall(self.tag2id,correct_dic,golden_tags_counter)
            assert len(precision_score)==len(recall_score)==self.num_tags
            f1_score=cal_f1(self.tag2id,precision_score,recall_score)

            print_scores(tag2id, precision_score, recall_score, f1_score, golden_tags_counter)
        import pickle
        if self.use_CRF:
            with open("./CNN_BiLSTM_CRF.pkl","wb") as f:
                pickle.dump(precision_score,f)
                pickle.dump(recall_score,f)
                pickle.dump(f1_score,f)
        else:
            with open('./CNN_BiLSTM.pkl','wb') as f:
                pickle.dump(precision_score,f)
                pickle.dump(recall_score,f)
                pickle.dump(f1_score,f)
        print("Has save results with pickle!")
        return precision_score,recall_score,f1_score
        
                    
                
            
def batch_yield_(pad_all_sentence_id, pad_all_sentence_tag_id, actual_length_list, padded_char_matrix,batch_size):
  
    num_batches=len(pad_all_sentence_tag_id)//batch_size
    shuffled=np.random.permutation(len(pad_all_sentence_id))
    
    shuffled_all_sentence_id=np.array(pad_all_sentence_id)[shuffled]
    shuffled_all_sentence_tag_id=np.array(pad_all_sentence_tag_id)[shuffled]
    shuffled_actual_length_list=np.array(actual_length_list)[shuffled]
    shuffled_char_matrix=np.array(padded_char_matrix)[shuffled]
    start=0
    for i in range(num_batches):
        yield shuffled_all_sentence_id[start:start+batch_size],shuffled_all_sentence_tag_id[start:start+batch_size],\
            shuffled_actual_length_list[start:start+batch_size],shuffled_char_matrix[start:start+batch_size]
        start+=batch_size
        




if __name__ == "__main__":
    file_path = r'D:\Name_Entity_Recogination_Review\运行文件\data\train.txt'
    glove_path = r'D:\Name_Entity_Recogination_Review\运行文件\data\glove.6B.100d.txt'
    test_file_path = r'D:\Name_Entity_Recogination_Review\运行文件\data\test.txt'
    dev_file_path=r'D:\Name_Entity_Recogination_Review\运行文件\data\dev.txt'
    config = {}
    config['epoches']=50
    config['batch_size'] = 64
    config['hidden_dim'] = 200
    config['max_seq_length'] = 170
    config['char_embedding_dim']=30
    config['max_word_length']=30
    config['filter_size']=5
    config['char_feature_size']=70
    config['model_save_path']='./CNN_BiLSTM_CRF_log/CNN_BiLSTM_CRF_model.ckpt'
    all_sentence_train,all_sentence_tag_train=read_file(file_path)
    dev_all_sentence,dev_all_sentence_tag=read_file(dev_file_path)
    all_sentence=all_sentence_train+dev_all_sentence
    all_sentence_tag=all_sentence_tag_train+dev_all_sentence_tag
    word2id, tag2id, word_embedding=get_parameter(all_sentence,all_sentence_tag,glove_path)
    all_sentence_id, all_sentence_tag_id=sentence_to_id(all_sentence,all_sentence_tag,word2id,tag2id)
    pad_all_sentence_id, pad_all_sentence_tag_id, actual_length_list=pad_sentence_id(all_sentence_id,all_sentence_tag_id,max_seq_length=config['max_seq_length'])

    char2id=get_char2id(all_sentence)
    char2id['UNK']=len(char2id)
    char_embedding_matrix=np.random.uniform(-3.0/np.sqrt(config['char_embedding_dim']),3.0/np.sqrt(config['char_embedding_dim']),
                                            size=(len(char2id),config['char_embedding_dim']))
    print("The length of char2id is ",len(char2id))
    all_charsentence_id=char_sentence_to_id(all_sentence,char2id)
    padded_char_matrix=pad_charsentence_id(all_charsentence_id,config['max_seq_length'],config['max_word_length'])
    
    #训练数据已经准备好，pad_all_sentence_id, pad_all_sentence_tag_id, actual_length_list，padded_char_matrix
    #分别对应着模型的四个占位符
    model=CNN_BiLSTM_CRF(word_embedding,char_embedding_matrix,tag2id,config,use_CRF=True)
    model.build_graph()
    model.train(pad_all_sentence_id, pad_all_sentence_tag_id, actual_length_list,padded_char_matrix)
    print("Next is test stage!")
    all_sentence_test,all_sentence_tag_test=read_file(test_file_path)
    all_sentence_id_test,all_sentence_tag_id_test=sentence_to_id(all_sentence_test,all_sentence_tag_test,word2id,tag2id)
    pad_all_sentence_id_test,pad_all_sentence_tag_id_test,actual_length_list_test=pad_sentence_id(all_sentence_id_test,all_sentence_tag_id_test,max_seq_length=config['max_seq_length'])
    all_charsentence_id_test=char_sentence_to_id(all_sentence_test,char2id)#把测试集中的所有句子的每一个句子的每一个单词的每一个字符转换成整数值
    padded_char_matrix_test=pad_charsentence_id(all_charsentence_id_test,config['max_seq_length'],config['max_word_length'])
    
    precision_score,recall_score,f1_score=model.test(pad_all_sentence_id_test,pad_all_sentence_tag_id_test,actual_length_list_test,padded_char_matrix_test)
    with open("./CNN_BiLSTM_CRF_results.txt","w",encoding="utf-8") as f:
        f.write("tag name"+"\t"+"precision_score"+"\t"+"recall_score"+"\t"+"f1_score"+"\n")
        for tag in tag2id:
            f.write(tag+"\t"+str(precision_score[tag])+"\t"+str(recall_score[tag])+"\t"+str(f1_score)+"\n")
            
            
    model=CNN_BiLSTM_CRF(word_embedding,char_embedding_matrix,tag2id,config,use_CRF=False)
    model.build_graph()
    model.train(pad_all_sentence_id, pad_all_sentence_tag_id, actual_length_list,padded_char_matrix)
    print("Next is test stage!")
    all_sentence_test,all_sentence_tag_test=read_file(test_file_path)
    all_sentence_id_test,all_sentence_tag_id_test=sentence_to_id(all_sentence_test,all_sentence_tag_test,word2id,tag2id)
    pad_all_sentence_id_test,pad_all_sentence_tag_id_test,actual_length_list_test=pad_sentence_id(all_sentence_id_test,all_sentence_tag_id_test,max_seq_length=config['max_seq_length'])
    all_charsentence_id_test=char_sentence_to_id(all_sentence_test,char2id)#把测试集中的所有句子的每一个句子的每一个单词的每一个字符转换成整数值
    padded_char_matrix_test=pad_charsentence_id(all_charsentence_id_test,config['max_seq_length'],config['max_word_length'])
    
    precision_score,recall_score,f1_score=model.test(pad_all_sentence_id_test,pad_all_sentence_tag_id_test,actual_length_list_test,padded_char_matrix_test)
    with open("./CNN_BiLSTM_results.txt","w",encoding="utf-8") as f:
        f.write("tag name"+"\t"+"precision_score"+"\t"+"recall_score"+"\t"+"f1_score"+"\n")
        for tag in tag2id:
            f.write(tag+"\t"+str(precision_score[tag])+"\t"+str(recall_score[tag])+"\t"+str(f1_score)+"\n")
    print("It's over!!!!!")
