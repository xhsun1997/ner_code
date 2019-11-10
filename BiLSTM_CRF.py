import tensorflow as tf
import numpy as np
import os
from process_data import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True


class BiLSTM_CRF_Model:
    def __init__(self,word_embedding,word2id,tag2id,config,use_CRF=True):
        self.word_embedding=word_embedding
        self.embedding_size=len(word2id)
        self.embedding_dim=self.word_embedding.shape[1]
        self.num_tags=len(tag2id)
        self.config=config
        self.use_CRF=use_CRF
        self.batch_size=config['batch_size']
        self.max_seq_length=config['max_seq_length']
        self.hidden_dim=config['hidden_dim']
        self.epoches=config['epoches']
        self.tag2id=tag2id
        self.model_save_path=config['model_save_path']
        tf.reset_default_graph()

    def placeholder_layer(self):
        self.word_ids=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_seq_length])
        self.label_ids=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_seq_length])
        self.seq_length=tf.placeholder(dtype=tf.int32,shape=[self.batch_size])
        print("placeholder layer is build")

    def embedding_lookup_layer(self):
        word_embeds=tf.constant(self.word_embedding,dtype=tf.float32)
        assert word_embeds.shape==(self.embedding_size,self.embedding_dim)
        self.BiLSTM_inputs=tf.nn.embedding_lookup(params=word_embeds,ids=self.word_ids)
        assert self.BiLSTM_inputs.shape==(self.batch_size,self.max_seq_length,self.embedding_dim)
        print("embedding_lookup_layer is build")

    def BiLSTM_layer(self):
        cell_fw=tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        cell_bw=tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
        (output_fw,output_bw),states=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs=self.BiLSTM_inputs,sequence_length=self.seq_length,
                                                                              dtype=tf.float32)
        assert output_bw.shape==(self.batch_size,self.max_seq_length,self.hidden_dim)==output_fw.shape
        output_concat=tf.concat(values=[output_fw,output_bw],axis=-1)
        assert output_concat.shape==(self.batch_size,self.max_seq_length,self.hidden_dim*2)
        self.BiLSTM_outputs=tf.nn.dropout(output_concat,keep_prob=0.5)
        print("BiLSTM layer is build")

    def Fully_connect_layer(self):
        weights=tf.Variable(tf.random_normal(shape=[self.batch_size,self.hidden_dim*2,self.num_tags],dtype=tf.float32))
        biases=tf.Variable(tf.random_normal(shape=[self.num_tags],dtype=tf.float32))
        matmul_results=tf.matmul(self.BiLSTM_outputs,weights)+biases
        self.logits=tf.nn.dropout(matmul_results,keep_prob=0.5)
        assert self.logits.shape==(self.batch_size,self.max_seq_length,self.num_tags)
        print("Fully_connect layer is built!")

    def loss_layer(self):
        if self.use_CRF:
            log_likelihood,self.transition_matrix=tf.contrib.crf.crf_log_likelihood(self.logits,self.label_ids,sequence_lengths=self.seq_length)
            self.loss=tf.reduce_mean(-log_likelihood)
        else:
            softmax_losses=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.label_ids)
            mask=tf.sequence_mask(self.seq_length,maxlen=self.max_seq_length)
            self.loss=tf.reduce_mean(tf.boolean_mask(tensor=softmax_losses,mask=mask))
        self.train_op=tf.train.AdamOptimizer(0.01).minimize(self.loss)
        print("loss layer is built")

    def build_graph(self):
        self.placeholder_layer()
        self.embedding_lookup_layer()
        self.BiLSTM_layer()
        self.Fully_connect_layer()
        self.loss_layer()
        print("The graph has been built!")

    def train(self,pad_all_sentence_id, pad_all_sentence_tag_id, actual_length_list):
        saver=tf.train.Saver(max_to_keep=1)
        with tf.Session(config=config_gpu) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoches):
                batch_data=batch_yield(pad_all_sentence_id, pad_all_sentence_tag_id, actual_length_list,self.batch_size)
                total_loss=0.0
                num_batches=len(pad_all_sentence_tag_id)//self.batch_size
                for step,(batch_sentence_id,batch_sentence_tag_id,batch_actual_length) in enumerate(batch_data):
                    assert self.word_ids.shape==batch_sentence_id.shape==self.label_ids.shape==batch_sentence_tag_id.shape
                    assert self.seq_length.shape==batch_actual_length.shape
                    feed_dict={self.word_ids:batch_sentence_id,self.label_ids:batch_sentence_tag_id,self.seq_length:batch_actual_length}
                    _,loss_val=sess.run([self.train_op,self.loss],feed_dict=feed_dict)
                    total_loss+=loss_val
                if epoch%10==0:
                    print("loss value is ",total_loss/num_batches)
                    saver.save(sess,self.model_save_path)
                    print("It has trained one epoch and save the model!")

    def test(self,pad_all_sentence_id_test,pad_all_sentence_tag_id_test,actual_length_list_test):
        saver=tf.train.Saver()
        correct,total=0,0
        correct_dic={}
        id2tag={}
        for tag_,id_ in self.tag2id.items():
            id2tag[id_]=tag_
        print(id2tag)
        golden_tags,predict_tags=[],[]
        with tf.Session(config=config_gpu) as sess:
            saver.restore(sess,self.model_save_path)
            print("It has restored the model")
            for word_ids_,label_ids_,seq_length_ in batch_yield(pad_all_sentence_id_test,pad_all_sentence_tag_id_test,actual_length_list_test,self.batch_size):
                assert self.word_ids.shape==word_ids_.shape==self.label_ids.shape==label_ids_.shape
                assert self.seq_length.shape==seq_length_.shape
                feed_dict={self.word_ids:word_ids_,self.seq_length:seq_length_}
                if self.use_CRF:
                    viterbi_seq,viterbi_score=tf.contrib.crf.crf_decode(self.logits,self.transition_matrix,sequence_length=self.seq_length)
                    assert viterbi_seq.shape==(self.batch_size,self.max_seq_length)
                    viterbi_seq_=sess.run(viterbi_seq,feed_dict=feed_dict)
                    assert viterbi_seq_.shape==label_ids_.shape
                    assert type(viterbi_seq_)==type(label_ids_)==np.ndarray
                    for predict_label,correct_label,actual_length in zip(viterbi_seq_,label_ids_,seq_length_):
                        assert len(predict_label)==len(correct_label)==self.max_seq_length
                        predict_label=predict_label[:actual_length]
                        correct_label=correct_label[:actual_length]
                        golden_tags.append(correct_label)
                        predict_tags.append(predict_label)
                        for pred,true in zip(predict_label,correct_label):
                            pred_tag=id2tag[pred]
                            true_tag=id2tag[true]
                            if pred==true:
                                correct+=1
                                assert pred_tag==true_tag
                                if pred_tag not in correct_dic:
                                    correct_dic[pred_tag]=1
                                else:
                                    correct_dic[pred_tag]+=1
                            total+=1

                else:
                    predict_argmax=tf.argmax(self.logits,axis=-1)
                    predict_tensor=tf.cast(predict_argmax,dtype=tf.int32)
                    predict_label_matrix=sess.run(predict_tensor,feed_dict)
                    assert predict_label_matrix.shape==(self.batch_size,self.max_seq_length)
                    for predict_label,correct_label,actual_length in zip(predict_label_matrix,label_ids_,seq_length_):
                        assert len(predict_label)==len(correct_label)==self.max_seq_length
                        assert len(predict_label)==len(correct_label)==self.max_seq_length
                        predict_label=predict_label[:actual_length]
                        correct_label=correct_label[:actual_length]
                        golden_tags.append(correct_label)
                        predict_tags.append(predict_label)
                        for pred,true in zip(predict_label,correct_label):
                            pred_tag=id2tag[pred]
                            true_tag=id2tag[true]
                            assert type(pred_tag)==type(true_tag)==str
                            if pred==true:
                                correct+=1
                                assert pred_tag==true_tag
                                if pred_tag not in correct_dic:
                                    correct_dic[pred_tag]=1
                                else:
                                    correct_dic[pred_tag]+=1
                            total+=1

            print("correct label num / total tag num is ",correct/total)

            golden_tags=flatten_list(golden_tags)
            predict_tags=flatten_list(predict_tags)
            assert len(golden_tags)==len(predict_tags)
            from collections import Counter
            assert type(golden_tags)==type(predict_tags)==list

            predict_tags_counter=Counter(predict_tags)
            golden_tags_counter=Counter(golden_tags)
            precision_score=cal_precision(self.tag2id,correct_dic,predict_tags_counter)
            recall_score=cal_recall(self.tag2id,correct_dic,golden_tags_counter)
            assert len(precision_score)==len(recall_score)==self.num_tags
            f1_score=cal_f1(self.tag2id,precision_score,recall_score)

            print_scores(tag2id, precision_score, recall_score, f1_score, golden_tags_counter)
            print("test the model is over!")
        import pickle
        if self.use_CRF:
            with open("./BiLSTM_CRF.pkl",'wb') as f:
                pickle.dump(precision_score,f)
                pickle.dump(recall_score,f)
                pickle.dump(f1_score,f)
        else:
            with open('./BiLSTM.pkl','wb') as f:
                pickle.dump(precision_score,f)
                pickle.dump(recall_score,f)
                pickle.dump(f1_score,f)

        
        return precision_score,recall_score,f1_score


#correct_dict是一个字典记录的是每一个标签被正确预测的次数
#predict_tags_counter 是Counter的字典对象，记录的是预测结果中每一个标签被预测的次数
#golden_tags_counter是真实标签中每一个标签应该出现的次数

if __name__ == "__main__":
    file_path = "./data/train.txt"
    glove_path = "./data/glove.6B.100d.txt"
    test_file_path = "./data/test.txt"
    dev_file_path="./data/dev.txt"
    config = {}
    config['epoches']=50
    config['batch_size'] = 64
    config['hidden_dim'] = 200
    config['max_seq_length'] = 150
    config['model_save_path']='./BiLSTM_CRF_log/BiLSTM_CRF_model.ckpt'
    all_sentence_train,all_sentence_tag_train=read_file(file_path)
    dev_all_sentence,dev_all_sentence_tag=read_file(dev_file_path)
    all_sentence=all_sentence_train+dev_all_sentence
    all_sentence_tag=all_sentence_tag_train+dev_all_sentence_tag
    word2id, tag2id, word_embedding=get_parameter(all_sentence,all_sentence_tag,glove_path)
    all_sentence_id, all_sentence_tag_id=sentence_to_id(all_sentence,all_sentence_tag,word2id,tag2id)
    pad_all_sentence_id, pad_all_sentence_tag_id, actual_length_list=pad_sentence_id(all_sentence_id,all_sentence_tag_id,max_seq_length=config['max_seq_length'])

    all_sentence_test,all_sentence_tag_test=read_file(test_file_path)
    all_sentence_id_test,all_sentence_tag_id_test=sentence_to_id(all_sentence_test,all_sentence_tag_test,word2id,tag2id)
    pad_all_sentence_id_test,pad_all_sentence_tag_id_test,actual_length_list_test=pad_sentence_id(all_sentence_id_test,all_sentence_tag_id_test,max_seq_length=config['max_seq_length'])

    model=BiLSTM_CRF_Model(word_embedding,word2id,tag2id,config,use_CRF=True)
    model.build_graph()
    model.train(pad_all_sentence_id, pad_all_sentence_tag_id, actual_length_list)
    print("BiLSTM_CRF Model has trained over!")
    precision_score,recall_score,f1_score=model.test(pad_all_sentence_id_test,pad_all_sentence_tag_id_test,actual_length_list_test)
    print("Finish model of BiLSTM_CRF")

    with open(".model_results/BiLSTM_CRF_results.txt","w",encoding="utf-8") as f:
        f.write("tag name"+"\t"+"precision_score"+"\t"+"recall_score"+"\t"+"f1_score"+"\n")
        for tag in tag2id:
            f.write(tag+"\t"+"  "+str(round(precision_score[tag],3))+"\t"+"  "+str(round(recall_score[tag],3))+"\t"+"  "+str(round(f1_score,3))+"\n")
    
    model=BiLSTM_CRF_Model(word_embedding,word2id,tag2id,config,use_CRF=False)
    model.build_graph()
    model.train(pad_all_sentence_id, pad_all_sentence_tag_id, actual_length_list)
    print("BiLSTM model has trained over")
    precision_score,recall_score,f1_score=model.test(pad_all_sentence_id_test, pad_all_sentence_tag_id_test, actual_length_list_test)
    print("Finish model of BiLSTM")
    with open(".model_results/BiLSTM_results.txt","w",encoding="utf-8") as f:
        f.write("tag name"+"\t"+"precision_score"+"\t"+"recall_score"+"\t"+"f1_score"+"\n")
        for tag in tag2id:
            f.write(tag+"\t"+"  "+str(round(precision_score[tag],3))+"\t"+"  "+str(round(recall_score[tag],3))+"\t"+"  "+str(round(f1_score,3))+"\n")

