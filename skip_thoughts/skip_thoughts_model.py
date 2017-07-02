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
"""Skip-Thoughts model for learning sentence vectors.

The model is based on the paper:

  "Skip-Thought Vectors"
  Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel,
  Antonio Torralba, Raquel Urtasun, Sanja Fidler.
  https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf

Layer normalization is applied based on the paper:

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
  https://arxiv.org/abs/1607.06450
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from skip_thoughts.ops import gru_cell
from skip_thoughts.ops import input_ops
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

MAXLEN=12


class TargetLSTM(object):
    def __init__(self, config):
       self.embed_dim = config.encoder_dim
       self.config=config
       self.input_noise = tf.placeholder(shape=[None, self.embed_dim], dtype=tf.float32)
       self.input_sequence = tf.placeholder(shape=[None, MAXLEN], dtype=tf.int64)

       self.uniform_initializer = tf.random_uniform_initializer(
           minval=-self.config.uniform_init_scale,
           maxval=self.config.uniform_init_scale)

       self.cell = gru_cell.LayerNormGRUCell(
           self.embed_dim,
           w_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
           u_initializer=random_orthonormal_initializer,
           b_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

       self.word_embedding = tf.get_variable('target_W', shape=[self.config.vocab_size, self.config.word_embedding_dim], 
                                             initializer=self.uniform_initializer)
       self.output_W = tf.get_variable('output_W', shape=[self.embed_dim, self.config.vocab_size], initializer=self.uniform_initializer)
       self.output_b = tf.get_variable('output_b', shape=[self.config.vocab_size], initializer=self.uniform_initializer)
       self.build_likelihood()
       self.build_rollout()

    def build_rollout(self):
       decoder_samples_arr = tensor_array_ops.TensorArray(dtype=tf.int64, size=MAXLEN, 
                                                       dynamic_size=False, infer_shape=True)
       cell = self.cell
       tmpstate = self.input_noise  
       tf.get_variable_scope().reuse_variables()
       def _g_recurrence_2(i, input_emb, tmpstate, samples_arr):
          (tmpout, tmpstate) = cell(input_emb, tmpstate)
          logits = tf.matmul(tmpout, self.output_W) + self.output_b
          #logits = tf.contrib.layers.fully_connected(
          #       inputs=tmpout, 
          #       num_outputs=self.config.vocab_size,
          #       activation_fn=None
          #)
          next_token = tf.reshape(tf.multinomial(logits, 1), [-1])
          next_input_emb = tf.nn.embedding_lookup(self.word_embedding, next_token)
          samples_arr = samples_arr.write(i, next_token)
          return i+1, next_input_emb, tmpstate,  samples_arr

       input_emb = tf.nn.embedding_lookup(self.word_embedding, tf.zeros(shape=[tf.shape(self.input_sequence)[0]], dtype=tf.int32))
       _, _, _, decoder_samples_arr = control_flow_ops.while_loop(
           cond=lambda i, _1, _2, _3: i < MAXLEN, 
           body=_g_recurrence_2, 
           loop_vars=(tf.constant(0, dtype=tf.int32), input_emb, tmpstate,  decoder_samples_arr), 
           name="target_recurrence_2"
       )
       self.rollout_samples = tf.transpose(decoder_samples_arr.stack(), perm=[1,0])

    def build_likelihood(self):
       cell = self.cell
       tmpstate = self.input_noise
       input_emb = tf.nn.embedding_lookup(self.word_embedding, tf.zeros(shape=[tf.shape(self.input_sequence)[0]], dtype=tf.int32))
       output_logits = []
       for i in range(MAXLEN):
          if i>0: tf.get_variable_scope().reuse_variables()
          (tmpout, tmpstate) = cell(input_emb, tmpstate)
          logits = tf.matmul(tmpout, self.output_W) + self.output_b
          #logits = tf.contrib.layers.fully_connected(
          #       inputs=tmpout,
          #       num_outputs=self.config.vocab_size, 
          #       activation_fn=None
          #)
          output_logits.append(logits)    # list of Tensor:[batch, vocab_size]
       output_logits = tf.stack(output_logits, 1)  # [batch, MAXLEN, vocab_size]
       output_logits = tf.reshape(output_logits, [-1, self.config.vocab_size]) 
       output_labels = tf.reshape(self.input_sequence, [-1])

       self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=output_labels,
              logits=output_logits
       )

def random_orthonormal_initializer(shape, dtype=tf.float32,
                                   partition_info=None):  # pylint: disable=unused-argument
  """Variable initializer that produces a random orthonormal matrix."""
  if len(shape) != 2 or shape[0] != shape[1]:
    raise ValueError("Expecting square shape, got %s" % shape)
  _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
  return u


class SkipThoughtsModel(object):
  def __init__(self, config, mode="train", input_reader=None):
    self.config = config
    self.mode = mode
    self.reader = input_reader if input_reader else tf.TFRecordReader()

    self.uniform_initializer = tf.random_uniform_initializer(
        minval=-self.config.uniform_init_scale,
        maxval=self.config.uniform_init_scale)

    self.target_cross_entropy_losses = []
    self.target_cross_entropy_loss_weights = []
    
  def build_inputs(self, is_testing=False):
    self.encode_ids = tf.placeholder(tf.int64, (None, None), name='encode_ids')
    self.autodecode_ids = tf.placeholder(tf.int64, (None, None), name='autodecode_ids')
    self.autocomp_ids = tf.placeholder(tf.int64, (None, None), name="autocomp_ids")

    self.encode_len = tf.placeholder(tf.int64, (None,), name='encode_len')
    self.autodecode_len = tf.placeholder(tf.int64, (None,), name='autodecode_len')
    self.autocomp_len = tf.placeholder(tf.int64, (None,), name="autocomp_len")

    self.encode_mask = tf.sequence_mask(self.encode_len-1, maxlen=MAXLEN, dtype=tf.int64)
    self.autodecode_mask = tf.sequence_mask(self.autodecode_len, maxlen=MAXLEN, dtype=tf.int64)
    self.autocomp_mask = tf.sequence_mask(self.autocomp_len, maxlen=MAXLEN, dtype=tf.int64)

    self.autocomp_weight = tf.placeholder(tf.float32, (None, None), name='autodecode_weight')
    self.given_num = tf.placeholder(tf.int32, name="given_num")
       
  def build_word_embeddings(self, is_testing=False):
    if(is_testing):
        tf.get_variable_scope().reuse_variables()

    self.word_emb_en = tf.get_variable(
        name="word_embedding_en",
        shape=[self.config.vocab_size, self.config.word_embedding_dim],
        initializer=self.uniform_initializer)
    self.word_emb_fr = tf.get_variable(
        name="word_embedding_fr",
        shape=[self.config.vocab_size, self.config.word_embedding_dim],
        initializer=self.uniform_initializer)

    self.encode_emb = tf.nn.embedding_lookup(self.word_emb_en, self.encode_ids)
    self.autodecode_emb = tf.nn.embedding_lookup(self.word_emb_fr, self.autodecode_ids)

  def build_attention_matrix(self, is_testing=False):
    if(is_testing):
        tf.get_variable_scope().reuse_variables()
    self.attention_W = tf.get_variable(
        name="attention_W", 
        shape = [self.config.encoder_dim, self.config.encoder_dim], 
        initializer=self.uniform_initializer
    )

  def _initialize_gru_cell(self, num_units):
    return gru_cell.LayerNormGRUCell(
        num_units,
        w_initializer=self.uniform_initializer,
        u_initializer=random_orthonormal_initializer,
        b_initializer=tf.constant_initializer(0.0))

  def build_attentive_encoder(self):
    with tf.variable_scope("attentive_encoder") as scope:
      length = tf.to_int32(tf.reduce_sum(self.encode_mask, 1), name="length")
      self.encoder_length = length
      cell = self._initialize_gru_cell(self.config.encoder_dim)
      state = cell.zero_state(tf.shape(self.encode_emb)[0], dtype=tf.float32)
      self.zero_state = state
      state_list = []
      with tf.variable_scope('encoder') as scope:
         for i in range(MAXLEN):
            state_list.append(state)
            if i>0: scope.reuse_variables()
            _, state = cell(self.encode_emb[:,i,:], state)
      thought_vectors = tf.identity(state, name="thought_vectors") 
      print("state shape=", state.get_shape())
      print("thought_vecotrs shape=", thought_vectors.get_shape())
    self.thought_vec_dev = tf.norm(thought_vectors, ord=2, axis=-1)
    self.thought_vectors = thought_vectors
    self.encoder_state_arr = tf.stack(state_list, axis=1)      # [batch_size, MAXLEN, state_size]
    self.encoder_cell = cell

  def _get_context_input(self, state, encoder_state_arr, encode_embed):
    state_exp = tf.expand_dims(state, 1)      # [batch_size, 1, decoder_state_size] # encoder_state_arr: [batch_size, MAXLEN, encoder_state_size]
    ###### to calculate h1'*W*h2, first calculate W*h2 ######
    q = 1
    state_list = tf.unstack(encoder_state_arr, axis=1)
    state_pad_list = [self.zero_state]*((q-1)//2) + state_list + [self.zero_state]*((q-1)//2)
    state_mul_list = []
    for i in range((q-1)//2, MAXLEN+(q-1)//2):
       tmp = self.zero_state
       for j in range(i-(q-1)//2, i+(q-1)//2+1):
          tmp = tmp + state_pad_list[j]
       tmp = tmp/np.float(q)
       state_mul_list.append(tf.matmul(tmp, self.attention_W))
    state_mul = tf.stack(state_mul_list, axis=1)
    #########################################################
    encode_mask = self.encode_mask

    alpha = tf.nn.softmax(tf.reduce_sum(state_exp*state_mul, 2) + 100*tf.cast(encode_mask, tf.float32) )    # [batch_size, MAXLEN]
    alpha_exp = tf.expand_dims(alpha, -1)  # [batch_size, MAXLEN, 1], # encode_embed: [batch_size, MAXLEN, embed_dim]
    context_input = tf.reduce_sum(encoder_state_arr*alpha_exp, 1)     # [batch_size, embed_dim]
    return context_input, alpha

  def _build_rollout(self, name, decoder_input_emb, decoder_input_ids, initial_state, 
                     given_num, is_testing=False, word_emb=None):

    decoder_samples_arr = tensor_array_ops.TensorArray(dtype=tf.int64, size=MAXLEN, 
                                                       dynamic_size=False, infer_shape=True)

    decoder_input_arr = tensor_array_ops.TensorArray(dtype=tf.float32, size=MAXLEN)
    decoder_input_arr = decoder_input_arr.unstack(tf.transpose(decoder_input_emb, perm=[1,0,2]))

    cell = getattr(self, name+"_cell")

    def _g_recurrence_1(i, input_emb, tmpstate, given_num, samples_arr): 
        with tf.variable_scope(name, reuse=True) as scope:
           context_input, _ = self._get_context_input(tmpstate, self.encoder_state_arr, self.encode_emb)
           (tmpout, tmpstate) = cell(tf.concat([input_emb, context_input], 1), tmpstate)
        with tf.variable_scope("logits", reuse=True) as scope:
           logits = tf.contrib.layers.fully_connected(
                 inputs=tmpout, 
                 num_outputs=self.config.vocab_size,
                 activation_fn=None,
                 scope=scope
           )
        logits = logits - tf.one_hot(indices=[1], depth=self.config.vocab_size)*300
        sample_token = tf.reshape(tf.multinomial(logits, 1), [-1])
        samples_arr = samples_arr.write(i, sample_token)
        next_input_emb = decoder_input_arr.read(i)
        return i+1, next_input_emb, tmpstate, given_num, samples_arr

    def _g_recurrence_2(i, input_emb, tmpstate, given_num, samples_arr):
        with tf.variable_scope(name, reuse=True) as scope:
           context_input, _ = self._get_context_input(tmpstate, self.encoder_state_arr, self.encode_emb)
           (tmpout, tmpstate) = cell(tf.concat([input_emb, context_input], 1), tmpstate)
        with tf.variable_scope("logits", reuse=True) as scope:
           logits = tf.contrib.layers.fully_connected(
                 inputs=tmpout, 
                 num_outputs=self.config.vocab_size,
                 activation_fn=None,
                 scope=scope
           )
        logits = logits - tf.one_hot(indices=[1], depth=self.config.vocab_size)*300
        next_token = tf.reshape(tf.multinomial(logits, 1), [-1])
        next_input_emb = tf.nn.embedding_lookup(word_emb, next_token)
        samples_arr = samples_arr.write(i, next_token)
        return i+1, next_input_emb, tmpstate, given_num, samples_arr

    initial_emb = tf.nn.embedding_lookup(word_emb, tf.zeros_like(self.autodecode_len, dtype=tf.int64))
    i, input_emb, tmpstate, given_num, decoder_samples_arr = control_flow_ops.while_loop(
        cond=lambda i, _1, _2, given_num, _4: i < given_num, 
        body=_g_recurrence_1, 
        loop_vars=(tf.constant(0, dtype=tf.int32), initial_emb, initial_state, given_num, decoder_samples_arr), 
        name="recurrence_1"
    )

    _, _, _, _, decoder_samples_arr = control_flow_ops.while_loop(
        cond=lambda i, _1, _2, _3, _4: i < MAXLEN, 
        body=_g_recurrence_2, 
        loop_vars=(i, input_emb, tmpstate, given_num, decoder_samples_arr), 
        name="recurrence_2"
    )

    self.rollout_samples = tf.transpose(decoder_samples_arr.stack(), perm=[1,0])

  def _build_decoder(self, name, decoder_input_emb, targets, targets_weight, mask, initial_state,
                     reuse_logits, is_testing=False, word_emb=None):
    cell = self._initialize_gru_cell(self.config.encoder_dim)
    setattr(self, name+"_cell", cell)

    if not is_testing: 
        print("building training module...")
        length = tf.reduce_sum(mask, 1, name="length")
        decoder_input = tf.pad(
           decoder_input_emb[:, :-1, :], [[0, 0], [1, 0], [0, 0]], name="input")

        state = initial_state
        decoder_output=[]
        attention_output=[]
        with tf.variable_scope(name) as scope:
           for i in range(MAXLEN):
              if i>0: scope.reuse_variables()
              context_input, alpha = self._get_context_input(state, self.encoder_state_arr, self.encode_emb)
              output, state = cell(tf.concat([decoder_input[:,i,:], context_input], 1), state)
              decoder_output.append(output)
              attention_output.append(alpha)   
        attention_output = tf.stack(attention_output, axis=1)    # [batch_size, MAXLEN, MAXLEN]
        self.attention_output = attention_output

        # Stack batch vertically.
        decoder_output = tf.stack(decoder_output, axis=1)
        print("decoder_output.shape=", decoder_output.get_shape())
        decoder_output = tf.reshape(decoder_output, [-1, self.config.encoder_dim])
        targets = tf.reshape(targets, [-1])
        weights = tf.reshape(tf.cast(mask, tf.float32) * targets_weight, [-1])

        # Logits.
        with tf.variable_scope("logits", reuse=reuse_logits) as scope:
           logits = tf.contrib.layers.fully_connected(
               inputs=decoder_output,
               num_outputs=self.config.vocab_size,
               activation_fn=None,
               weights_initializer=self.uniform_initializer,
               scope=scope)
                      
 
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits)
        batch_loss = tf.reduce_sum(losses * weights)
        tf.losses.add_loss(batch_loss)
        tf.summary.scalar("losses/" + name, batch_loss)

        self.target_cross_entropy_losses.append(losses)
        self.target_cross_entropy_loss_weights.append(weights)

  def build_decoders(self, is_testing=False):
      if(is_testing):
         tf.get_variable_scope().reuse_variables()
      self._build_decoder(name="autodecoder", 
                          decoder_input_emb=self.autodecode_emb,
                          targets=self.autocomp_ids, 
                          targets_weight=self.autocomp_weight, 
                          mask=self.autodecode_mask,
                          initial_state=self.thought_vectors, 
                          reuse_logits=False, 
                          is_testing=is_testing, 
                          word_emb=self.word_emb_fr)

  def build_rollout(self, is_testing=False):
      self._build_rollout(name="autodecoder", 
                          decoder_input_emb=self.autodecode_emb, 
                          decoder_input_ids=self.autodecode_ids, 
                          initial_state=self.thought_vectors, 
                          given_num=self.given_num, 
                          is_testing=is_testing, 
                          word_emb=self.word_emb_fr)

  def build_loss(self):
    if self.mode != "encode":
      total_loss = tf.losses.get_total_loss()
      tf.summary.scalar("losses/total", total_loss)
      self.total_loss = total_loss

  def build_global_step(self, is_testing=False):
    if(is_testing): 
      return
    self.global_step = tf.contrib.framework.create_global_step()
    self.next_global_step = tf.placeholder(tf.int64)
    self.set_global_step = tf.assign(self.global_step, self.next_global_step)

  def build(self, is_testing=False):
    self.build_attention_matrix(is_testing=is_testing)
    self.build_inputs(is_testing=is_testing)
    self.build_word_embeddings(is_testing=is_testing)
    self.build_attentive_encoder()
    self.build_decoders(is_testing=is_testing)
    self.build_loss()
    self.build_rollout(is_testing=is_testing)
    self.build_global_step(is_testing=is_testing)

