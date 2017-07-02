# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train the skip-thoughts model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import tensorflow as tf
import numpy as np
from skip_thoughts.discriminator import Discriminator

from skip_thoughts import configuration
from skip_thoughts import skip_thoughts_model 
from skip_thoughts.ops import gru_cell
from tensorflow.python.platform import tf_logging as logging
from bleu_score import bleu

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("train_dir", None, "Directory for saving and loading checkpoints.")
tf.flags.DEFINE_string("reload_model", None, "A pretrained model.")
tf.flags.DEFINE_string("reload_model_all", None, "A pretrained model.")
tf.flags.DEFINE_string("vocab", None, "Vocab file.")
tf.flags.DEFINE_string("test_result", None, "The file to save testing result.")
tf.flags.DEFINE_string("train_corpus_en", None, "Training corpus (english).")
tf.flags.DEFINE_string("train_corpus_fr", None, "Training corpus (french)")
tf.flags.DEFINE_string("test_corpus", None, "Testing corpus.")
tf.flags.DEFINE_integer("maxlen", 30, "Maximum length.")
tf.flags.DEFINE_integer("given_num", 12, "Given num.")
tf.flags.DEFINE_integer("pretrain_G_steps", 0, "Pretrain steps for G.")
tf.flags.DEFINE_integer("pretrain_D_steps", 0, "Pretrain steps for D.")
tf.flags.DEFINE_boolean("adjustD", True, "Allow to adjust D.")
tf.flags.DEFINE_boolean("adjustG", True, "Allow to adjust G.")
tf.flags.DEFINE_integer("mixer_period", None, "Mixer period.")
tf.flags.DEFINE_integer("mixer_step", None, "Mixer step.")

#vocab=[]       # index to word
#nv_vocab={}   # word to index
MAXLEN = FLAGS.maxlen
default_given_num = FLAGS.given_num
weight_bias_d = 0
weight_bias_b = 0
weight_bias_lr = 0.01

np.random.seed(100)

def read_vocab(filename):
   f = open(filename, 'r')
   cnt = 0
   for line in f: 
      vocab.append(line.strip())
      inv_vocab[line.strip()] = cnt
      cnt += 1
   f.close()

class DataGenerator(object):
    def __init__(self, batch_size, max_length=MAXLEN, target_lstm=None):
       self.batch_size = batch_size
       self.max_length = max_length
       self.target_lstm = target_lstm
    def get_next_batch(self, sess):
       input_noise = np.random.normal(size=[self.batch_size, self.target_lstm.embed_dim])
       batch_samples = sess.run(self.target_lstm.rollout_samples, 
               feed_dict={self.target_lstm.input_noise: input_noise, self.target_lstm.input_sequence: np.zeros(shape=[self.batch_size, MAXLEN], dtype=np.int64) } )
       return batch_samples

class DataLoader(object):
    def __init__(self, batch_size, max_length=MAXLEN):
       self.max_length = max_length
       self.seqs_en = None
       self.seqs_fr = None
       self.cur_batch = 0
       self.batch_size = batch_size
    def load(self, filename_en, filename_fr):
       print("loading "+filename_en + " and " + filename_fr)
       f_en = open(filename_en, 'r')
       f_fr = open(filename_fr, 'r')
       self.seqs_en = []
       self.seqs_fr = []
       for line in f_en:
          sent = [int(x) for x in line.split()] + [0]
          line2 = f_fr.readline()
          sent2 = [int(x) for x in line2.split()] + [0]
          if(len(sent) <= self.max_length and len(sent2) <= self.max_length):
              self.seqs_en.append(sent)
              self.seqs_fr.append(sent2)
       f_en.close()
       f_fr.close()
       assert(len(self.seqs_en) == len(self.seqs_fr)) 
       print("loaded "+str(len(self.seqs_en))+" sentences")
        
    def get_data_num(self):
       return len(self.seqs_en)
 
    def get_range_batch(self, start, end):
       assert(start<=end and end-start<=self.batch_size)
       start = max(0, start)
       end=min(len(self.seqs_en), end)
       batch_en_ids = np.zeros([self.batch_size, self.max_length], dtype=np.int64)
       batch_fr_ids = np.zeros([self.batch_size, self.max_length], dtype=np.int64)
       batch_en_len = np.zeros([self.batch_size], dtype=np.int64)
       batch_fr_len = np.zeros([self.batch_size], dtype=np.int64)
       for i in range(start, end):
          batch_en_ids[i-start, 0:len(self.seqs_en[i])] = np.array(self.seqs_en[i], dtype=np.int64)
          batch_fr_ids[i-start, 0:len(self.seqs_fr[i])] = np.array(self.seqs_fr[i], dtype=np.int64)
          batch_en_len[i-start] = np.int64(len(self.seqs_en[i]))
          batch_fr_len[i-start] = np.int64(len(self.seqs_fr[i]))
       return batch_en_ids, batch_en_len, batch_fr_ids, batch_fr_len
  
    def get_next_batch(self):
       start = self.cur_batch*self.batch_size
       ret = self.get_range_batch(start, start+self.batch_size)
       self.cur_batch += 1
       if self.cur_batch * self.batch_size >= len(self.seqs_en):
           ### shuffle again ?? ###
           self.cur_batch = 0
       return ret

def np_to_list(nparr, max_length=MAXLEN):
    list1 = nparr.tolist()
    list2 = []
    for sent in list1:
       if 0 in sent: 
          sent = sent[:sent.index(0)+1]
       list2.append(sent)
    return list2

def list_to_np(list1, max_length=MAXLEN):
    sent_len = [len(x) for x in list1]
    width = max_length
    nparr = np.zeros([len(list1), width], dtype=np.int64)
    for i in range(len(list1)):
       nparr[i, :len(list1[i])] = np.array(list1[i], dtype=np.int64)
    return nparr, np.array(sent_len)
  
def cal_batch_bleu(samples_sents, ref_sents):
    assert(len(samples_sents)==len(ref_sents))
    bleu_np = np.zeros([len(samples_sents)], dtype=np.float32)
    for i in range(len(samples_sents)):
       cand = map(lambda w: str(w), samples_sents[i])
       ref = map(lambda w: str(w), ref_sents[i])
       bleu_np[i] = bleu(cand, [ref], [0.4, 0.4, 0.2])
    return bleu_np


def my_train_step(sess, train_op, model, train_data_loader, logf=None, train_sup=False, train_rl=False, disc_model=None, adjustG=False, adjustD=False, given_num=None):
   global weight_bias_b, weight_bias_d, weight_bias_lr
   global_step = model.global_step
   np_global_step_init = sess.run(global_step)

   fetches = {
      "rollout_samples": model.rollout_samples,
   }

   batch_en_ids, batch_en_len, batch_fr_ids, batch_fr_len = train_data_loader.get_next_batch()

   ############################ get one sample & supervised learning ##########################
   feed_dict = { model.encode_ids: batch_en_ids, 
                 model.encode_len: batch_en_len ,
                 model.autodecode_ids: batch_fr_ids,      
                 model.autodecode_len: batch_fr_len,        
                 model.autocomp_ids: batch_fr_ids,           
                 model.autocomp_len: batch_fr_len,           
                 model.autocomp_weight: np.ones_like(batch_fr_ids, dtype=np.float32),  
                 model.given_num: given_num 
               }
   if train_sup and adjustG:
      total_loss_sup, fetches_eval = sess.run([train_op, fetches], feed_dict=feed_dict)
   else:
      total_loss_sup, fetches_eval = sess.run([model.total_loss, fetches], feed_dict=feed_dict)

   neglikely_avg = 0
   

   if train_rl: 
      ########### processing MIXER  #########
      samples_sents = np_to_list(fetches_eval['rollout_samples'])
      batch_samples_ids, batch_samples_len = list_to_np(samples_sents)
   
      batch_mixed_ids = np.zeros_like(batch_samples_ids, dtype=np.int64)
      batch_mixed_ids[:,0:given_num] = batch_fr_ids[:,0:given_num]
      batch_mixed_ids[:,given_num:] = batch_samples_ids[:,given_num:]
      _, batch_mixed_len = list_to_np(np_to_list(batch_mixed_ids))
      batch_mixed_len = batch_fr_len
      batch_comp_ids, batch_comp_len = batch_mixed_ids, batch_mixed_len
      reward_matrix = np.zeros_like(batch_samples_ids, dtype=np.float32)
    
      ################# using discriminator as reward #################
      batch_mixed_label = np.array([[0]] * batch_mixed_ids.shape[0])
      batch_fr_label = np.array([[1]] * batch_fr_ids.shape[0])
      disc_feed_dict = { disc_model.input_x:  np.concatenate( [batch_mixed_ids, batch_fr_ids] , axis=0),
                         disc_model.input_y:  np.concatenate( [batch_mixed_label, batch_fr_label], axis=0), 
                         disc_model.dropout_keep_prob: 1.0 } 
    
      if adjustD:
         disc_loss, batch_all_score = sess.run([disc_model.train_op,  disc_model.ypred_for_auc], feed_dict=disc_feed_dict)
      else:
         batch_all_score = sess.run(disc_model.ypred_for_auc, feed_dict=disc_feed_dict)
         disc_loss = 0
      
      batch_mixed_score = batch_all_score[:batch_mixed_ids.shape[0], 0]
      batch_fr_score = batch_all_score[batch_mixed_ids.shape[0]:, 0]
      fake_avg = np.mean(batch_mixed_score)
      real_avg = np.mean(batch_fr_score)
      for j in range(given_num, MAXLEN):
          reward_matrix[:,j] += batch_mixed_score - weight_bias_d
      reward_matrix[:,:given_num] = 1.0
      weight_bias_d = weight_bias_lr * fake_avg + (1-weight_bias_lr)*weight_bias_d

      ################# using bleu score as reward ###################
      ref_sents = np_to_list(batch_fr_ids[:,given_num:])
      cand_sents = np_to_list(batch_mixed_ids[:,given_num:])
      bleu_weight = cal_batch_bleu(cand_sents, ref_sents)
      bleu_avg = np.mean(bleu_weight)
      for j in range(given_num, MAXLEN):
          reward_matrix[:,j] += bleu_weight - weight_bias_b
      weight_bias_b = weight_bias_lr * bleu_avg + (1-weight_bias_lr)*weight_bias_b

      ################ do REINFORCE #####################    
      feed_dict = { model.encode_ids: batch_en_ids, 
                    model.encode_len: batch_en_len,
                    model.autodecode_ids: batch_mixed_ids,
                    model.autodecode_len: batch_mixed_len,
                    model.autocomp_ids: batch_comp_ids,           
                    model.autocomp_len: batch_comp_len,           
                    model.autocomp_weight: reward_matrix,
                    model.given_num: MAXLEN
                  }
      if adjustG: 
         total_loss_rl, fetches_eval = sess.run([train_op, fetches], feed_dict=feed_dict)     
      else:
         total_loss_rl = 0 

   else:
      total_loss_rl = 0
      bleu_avg, fake_avg, real_avg = 0,0,0
      
   np_global_step = sess.run(model.set_global_step, feed_dict={model.next_global_step: np_global_step_init+1})
   return np_global_step, total_loss_sup, total_loss_rl, bleu_avg, fake_avg, real_avg, neglikely_avg


#def my_test_step(sess, model_test, test_result_filename, logf=None):
#  test_data_loader = DataLoader(vocab, inv_vocab, 128)
#  test_data_loader.load(FLAGS.test_corpus)  
#  print("writing to test_result...")
#
#  test_fetches = {
#     "encode_ids": model_test.autodecode_ids,
#     "cur_samples": model_test.rollout_samples
#  }
#
#  fres = open(test_result_filename, 'w')
#  for _ in range(1):
#      batch_en_ids, batch_en_len, batch_fr_ids, batch_fr_len = test_data_loader.get_next_batch()
#      feed_dict = { model_test.encode_ids: batch_en_ids, 
#                    model_test.encode_len: batch_en_len,
#                    model_test.autodecode_ids: batch_fr_ids,
#                    model_test.autodecode_len: batch_fr_len,   
#                    model_test.autocomp_len: batch_fr_len,    
#                    model_test.autocomp_ids: batch_fr_ids,    
#                    model_test.given_num: 0, 
#                  }
#      test_fetches_eval = sess.run(test_fetches, feed_dict=feed_dict)
#      encode_ids = test_fetches_eval['encode_ids']
#      cur_samples = test_fetches_eval['cur_samples']
#      for i in range(encode_ids.shape[0]):
#         fres.write(" <enc> ")
#         for j in range(encode_ids.shape[1]):
#            if(encode_ids[i,j]==0):  break
#            fres.write(vocab[encode_ids[i,j]]+" ")
#         fres.write("\n")
#         ##### dec #####
#         fres.write(" <dec> ")
#         for j in range(cur_samples.shape[1]):
#            if(cur_samples[i,j]==0): break
#            fres.write(vocab[cur_samples[i,j]]+" ")
#         fres.write("\n")
#         fres.write("\n")
#  fres.close()
#  print("end_writing")


def _setup_learning_rate(config, global_step):
  if config.learning_rate_decay_factor > 0:
    learning_rate = tf.train.exponential_decay(
        learning_rate=float(config.learning_rate),
        global_step=global_step,
        decay_steps=config.learning_rate_decay_steps,
        decay_rate=config.learning_rate_decay_factor,
        staircase=False)
  else:
    learning_rate = tf.constant(config.learning_rate)
  return learning_rate

def main(unused_argv):
  if not FLAGS.train_dir:
    raise ValueError("--train_dir is required.")
  
  #read_vocab(FLAGS.vocab)
  model_config = configuration.model_config()
  training_config = configuration.training_config()
  ################ define discriminator model ################
  disc_model = Discriminator(sequence_length=MAXLEN, 
                             num_classes=1,
                             vocab_size=model_config.vocab_size, 
                             embedding_size=model_config.word_embedding_dim, 
                             filter_sizes=[1,2,3,4,5,7,10], 
                             num_filters=[100,100,100,100,100,100,100])


  ################# define training model #################
  model = skip_thoughts_model.SkipThoughtsModel(model_config, mode="train")
  model.build()
  learning_rate = _setup_learning_rate(training_config, model.global_step)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  variables_to_train = [v for v in tf.trainable_variables()]
  variables_to_restore = [v for v in tf.all_variables() if ('discriminator' not in v.name)]

  print(len(variables_to_train))
  train_tensor = tf.contrib.slim.learning.create_train_op(
      total_loss=model.total_loss,
      optimizer=optimizer,
      clip_gradient_norm=training_config.clip_gradient_norm, 
      variables_to_train=variables_to_train)

 ######################define target lstm ####################
  #target_lstm = skip_thoughts_model.TargetLSTM(config=model_config)
  #synthesized = True
  target_lstm = None
  synthesized = False
  ################ define testing model ################
  #model_config_test = configuration.model_config()
  #model_test = skip_thoughts_model.SkipThoughtsModel(model_config_test, mode="eval")
  #model_test.build(is_testing=True)


  ################ define savers ################
  reloader = tf.train.Saver(var_list=variables_to_restore)
  reloader_all = tf.train.Saver()
  saver = tf.train.Saver(max_to_keep=1000)
  gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True),
                              allow_soft_placement=True,
                              log_device_placement=False)

  init_op = tf.global_variables_initializer()
  sess = tf.Session(config=gpu_config)
  run_metadata = tf.RunMetadata()
  sess.run(init_op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True), run_metadata=run_metadata)
  with open("/tmp/meta.txt", 'w') as f:
     f.write(str(run_metadata))

  if FLAGS.reload_model:
     reloader.restore(sess, FLAGS.reload_model)
  if FLAGS.reload_model_all:
     reloader_all.restore(sess, FLAGS.reload_model_all)

  ################ load training data ##############
  train_data_loader = DataLoader(128)
  train_data_loader.load(FLAGS.train_corpus_en, FLAGS.train_corpus_fr)

  total_loss_sup_list = []
  total_loss_rl_list = []
  bleu_list = []
  fake_list, real_list, neglikely_list = [],[],[]
  
  outf = open(os.path.join(FLAGS.train_dir, 'log.txt'), 'a')
  logf = open(os.path.join(FLAGS.train_dir, 'debug_log.txt'), 'w')

  ############### run training and testing #############
  for i in xrange(1000000):
      model_prefix = ""
      if i < FLAGS.pretrain_G_steps:
          model_prefix = "preG_"
          np_global_step, total_loss_sup, total_loss_rl, avg_bleu, avg_fake, avg_real, avg_neglikely = my_train_step(
                sess, train_tensor, model, train_data_loader, logf, train_sup=True, train_rl=False, 
                disc_model=disc_model, adjustD=False, adjustG=True, given_num=MAXLEN)

      elif i < FLAGS.pretrain_G_steps + FLAGS.pretrain_D_steps:
          model_prefix = "preD_"
          np_global_step, total_loss_sup, total_loss_rl, avg_bleu, avg_fake, avg_real, avg_neglikely  = my_train_step(
                sess, train_tensor, model, train_data_loader, logf, train_sup=False, train_rl=True, 
                disc_model=disc_model, adjustD=True, adjustG=False, given_num=0)

      elif FLAGS.mixer_period and FLAGS.mixer_step and FLAGS.mixer_period > 0:
          gn = default_given_num - (i-FLAGS.pretrain_G_steps-FLAGS.pretrain_D_steps)//FLAGS.mixer_period*FLAGS.mixer_step
          if gn < 0: gn=0
          model_prefix = "mixGN"+str(gn)+"_"
          if i % 10 == 0: 
              adjustD = FLAGS.adjustD
          else: 
              adjustD = False
          if i%200==0:
              print("gn=",gn)
          np_global_step, total_loss_sup, total_loss_rl, avg_bleu, avg_fake, avg_real, avg_neglikely  = my_train_step( \
                sess, train_tensor, model, train_data_loader, logf, train_sup=False, train_rl=True, \
                disc_model=disc_model, adjustD=adjustD, adjustG=FLAGS.adjustG, given_num=gn)

      else: 
          model_prefix = ""
          np_global_step, total_loss_sup, total_loss_rl, avg_bleu, avg_fake, avg_real, avg_neglikely  = my_train_step(
                       sess, train_tensor, model, train_data_loader, logf, train_sup=False, train_rl=True, 
                       disc_model=disc_model, adjustD=FLAGS.adjustD, adjustG=FLAGS.adjustG)

      total_loss_sup_list.append(total_loss_sup)
      total_loss_rl_list.append(total_loss_rl)
      fake_list.append(avg_fake)
      real_list.append(avg_real)
      bleu_list.append(avg_bleu)
      neglikely_list.append(avg_neglikely)

      if np_global_step%2000==0:
          saver.save(sess, os.path.join(FLAGS.train_dir, model_prefix+"model-"+str(np_global_step)))
      if np_global_step%20==0:
          # my_test_step(sess, model_test, FLAGS.test_result+'-'+str(np_global_step))
          print(np_global_step, np.mean(total_loss_sup_list), np.mean(total_loss_rl_list)) 
          print(np.mean(bleu_list), np.mean(fake_list), np.mean(real_list))
          print(np.mean(neglikely_list))
          outf.write(str(np_global_step) + " " + str(np.mean(total_loss_sup_list)) + " " + str(np.mean(total_loss_rl_list)) + " " + str(np.mean(bleu_list)) 
                     + " " + str(np.mean(fake_list)) + " " + str(np.mean(real_list)) + " " + str(np.mean(neglikely_list)) + "\n") 
          total_loss_sup_list, total_loss_rl_list, bleu_list, fake_list, real_list, neglikely_list = [],[],[],[],[],[]

if __name__ == "__main__":
  tf.app.run()
