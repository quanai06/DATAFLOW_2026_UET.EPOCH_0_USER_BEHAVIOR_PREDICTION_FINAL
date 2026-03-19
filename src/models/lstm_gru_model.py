import tensorflow as tf
from tensorflow.keras import layers, models, Input

from metrics.metrics import scaled_weighted_mse

import os
import dotenv
dotenv.load_dotenv()
FINAL_MAX_LEN = int(os.getenv("FINAL_MAX_LEN", 37))

def build_model(m_type, vocab_size, n_stats):
    inp_seq = Input(shape=(FINAL_MAX_LEN,), name='input_ids')
    emb = layers.Embedding(vocab_size, 128, mask_zero=True)(inp_seq)
    
    rnn = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3) if m_type=='lstm' 
                               else layers.GRU(128, return_sequences=True, dropout=0.3))(emb)
    
    att = layers.Activation('softmax', name='attention_weights')(layers.Flatten()(layers.Dense(1, activation='tanh')(rnn)))
    ctx = layers.Dot(axes=1)([att, rnn])
    
    inp_st = Input(shape=(n_stats,))
    merged = layers.Concatenate()([ctx, layers.Dense(64, activation='relu')(inp_st)])
    
    # Split-Head
    d_out = layers.Dense(4, activation='sigmoid')(layers.Dense(128, activation='relu')(merged))
    c_out = layers.Dense(2, activation='sigmoid')(layers.Dense(256, activation='relu')(merged))
    
    final = layers.Concatenate(name='final_output')([d_out[:,:2], c_out[:,:1], d_out[:,2:], c_out[:,1:]])
    
    model = models.Model(inputs=[inp_seq, inp_st], outputs=final)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0), loss=scaled_weighted_mse)
    return model