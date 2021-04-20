
import keras
from keras.models import Sequential,Model
from keras.layers import Embedding, Dense, merge, SimpleRNN, Activation, LSTM, GRU, Dropout,Input,TimeDistributed,BatchNormalization
from keras.layers import Concatenate,Add
from keras import optimizers
from EmbeddingMatrix import EmbeddingMatrix
from keras.utils.np_utils import to_categorical
import config
from mode_normalization import ModeNormalization
from keras_self_attention import SeqSelfAttention
from keras_pos_embd import PositionEmbedding,TrigPosEmbedding
GRID_COUNT = config.GRID_COUNT

TEXT_K = config.text_k





def serm(user_dim, len,word_vec, place_dim = GRID_COUNT*GRID_COUNT, time_dim=config.time_dim,
                            pl_d=config.pl_d, time_k=config.time_k, hidden_neurons=config.hidden_neurons,
                                   learning_rate=config.learning_rate):
    # RNN model construction
    pl_input = Input(shape=(len-1,), dtype='int32', name = 'pl_input')
    time_input = Input(shape=(len-1,), dtype='int32', name = 'time_input')
    user_input = Input(shape=(len-1,), dtype='int32', name='user_input')
    text_input = Input(shape=(len-1, word_vec.shape[0]), dtype='float32', name='text_input')


    pl_embedding = Embedding(input_dim=place_dim + 1, output_dim=pl_d, name ='pl_embedding' ,
                              mask_zero=True)(pl_input)
    time_embedding = Embedding(input_dim=time_dim + 1, output_dim=time_k, name='time_embedding',
                               mask_zero=True)(time_input)
    user_embedding = Embedding(input_dim=user_dim + 1, output_dim=place_dim + 1, name='user_embedding',
                               mask_zero=True)(user_input)

    # text_embedding = Embedding(input_dim=word_vec.shape[0],output_dim= TEXT_K,
    #                           weights=[word_vec],name="text_embeddng")(text_input)
    text_embedding = EmbeddingMatrix(TEXT_K, weights=[word_vec], name="text_embeddng", trainable=True)(text_input)


    attrs_latent = Concatenate(axis=-1)([pl_embedding,time_embedding,text_embedding])


    lstm_out = LSTM(hidden_neurons, return_sequences=True,name='lstm_layer0')(attrs_latent)

    dense = Dense(place_dim + 1, name='dense')(lstm_out)
    out_vec = Add()([dense,user_embedding])
    bnorm=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',moving_mean_initializer='zeros', moving_variance_initializer='ones', \
          beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint    =None)(out_vec)

    pred = Activation('softmax')(bnorm)
    #pred = Activation('softmax')(out_vec)
    model = Model([pl_input,time_input,user_input,text_input], pred)

    # model.load_weights('./model/User_RNN_Seg_Epoch_0.3_rmsprop_300.h5')
    # Optimization
    sgd = optimizers.SGD(lr=learning_rate)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy')
    model.summary()
    return model

def multi_model(user_dim, len,word_vec, place_dim = GRID_COUNT*GRID_COUNT, time_dim=config.time_dim,
                            pl_d=config.pl_d, time_k=config.time_k, hidden_neurons=config.hidden_neurons,
                                   learning_rate=config.learning_rate):
    # RNN model construction
    pl_input = Input(shape=(len-1,), dtype='int32', name = 'pl_input')
    time_input = Input(shape=(len-1,), dtype='int32', name = 'time_input')
    user_input = Input(shape=(len-1,), dtype='int32', name='user_input')
    text_input = Input(shape=(len-1, word_vec.shape[0]), dtype='float32', name='text_input')

    pl_embedding = Embedding(input_dim=place_dim + 1, output_dim=pl_d, name ='pl_embedding' ,
                              mask_zero=True)(pl_input)
    time_embedding = Embedding(input_dim=time_dim + 1, output_dim=time_k, name='time_embedding',
                               mask_zero=True)(time_input)
    user_embedding = Embedding(input_dim=user_dim + 1, output_dim=place_dim + 1, name='user_embedding', mask_zero=True)(user_input)
    #
    text_embedding = EmbeddingMatrix(TEXT_K, weights=[word_vec], name="text_embeddng", trainable=True)(text_input)
    #pl_em=PositionEmbedding(input_dim=place_dim+1,output_dim=pl_d,name="pl_embd",mask_zero=True)(pl_input)
    pl_em = TrigPosEmbedding(output_dim=pl_d, name="pl_embd")(pl_embedding)
    self_pl = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                    kernel_regularizer=keras.regularizers.l2(1e-6))(pl_embedding)
    #time_em=PositionEmbedding(input_dim=time_dim+1,output_dim=time_k,name="time_embd",mask_zero=True)(time_input)
    pl=Add()([pl_em,self_pl])


    attrs_latent = Concatenate(axis=-1)([pl,time_embedding,text_embedding])

    #lstm_time=LSTM(hidden_neurons, return_sequences=True, name='lstm_layer2')(time_embedding)
    # lstm_pl = LSTM(hidden_neurons, return_sequences=True,name='lstm_layer1')(pl_embedding)
    # lstm_time = LSTM(hidden_neurons, return_sequences=True, name='lstm_layer2')(time_embedding)
    # lstm_text = LSTM(hidden_neurons, return_sequences=True, name='lstm_layer3b')(text_embedding)
    lstm_out = LSTM(hidden_neurons, return_sequences=True,name='lstm_layer1')(attrs_latent)
    # lstm_out=Concatenate(axis=-1)([lstm_pl,lstm_time,lstm_text])
    #self1 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-6))(lstm_out)
    #
    # self2 = SeqSelfAttention(attention_width=15,attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,attention_activation=None,kernel_regularizer=keras.regularizers.l2(1e-6))(time_embedding)
    # self3 = SeqSelfAttention(attention_width=15,attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,attention_activation=None,kernel_regularizer=keras.regularizers.l2(1e-6))(text_embedding)
    # self_out=Concatenate(axis=-1)([self1,self2,self3])
    #attrs_latent = Concatenate(axis=-1)([self_pl, lstm_time, text_embedding])
    dense = Dense(place_dim + 1, name='dense')(lstm_out)
    out_vec = Add()([dense,user_embedding])
    bnorm=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',moving_mean_initializer='zeros', moving_variance_initializer='ones', \
          beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(out_vec)
    #modeNorm=ModeNormalization(k=2)(out_vec)
    pred = Activation('softmax')(bnorm)
    #pred = Activation('softmax')(out_vec)
    model = Model([pl_input,time_input,user_input,text_input], pred)

    # model.load_weights('./model/User_RNN_Seg_Epoch_0.3_rmsprop_300.h5')
    # Optimization
    sgd = optimizers.SGD(lr=learning_rate)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy')
    model.summary()
    return model
def cdapm(user_dim, len,word_vec, place_dim = GRID_COUNT*GRID_COUNT, time_dim=config.time_dim,
                            pl_d=config.pl_d, time_k=config.time_k, hidden_neurons=config.hidden_neurons,
                                   learning_rate=config.learning_rate):
    # RNN model construction
    pl_input = Input(shape=(len-1,), dtype='int32', name = 'pl_input')
    time_input = Input(shape=(len-1,), dtype='int32', name = 'time_input')
    user_input = Input(shape=(len-1,), dtype='int32', name='user_input')
    text_input = Input(shape=(len-1, word_vec.shape[0]), dtype='float32', name='text_input')
    pl_embedding = Embedding(input_dim=place_dim + 1, output_dim=pl_d, name ='pl_embedding' ,
                              mask_zero=True)(pl_input)
    time_embedding = Embedding(input_dim=time_dim + 1, output_dim=time_k, name='time_embedding',
                               mask_zero=True)(time_input)
    user_embedding = Embedding(input_dim=user_dim + 1, output_dim=place_dim + 1, name='user_embedding', mask_zero=True)(user_input)
    #
    text_embedding = EmbeddingMatrix(TEXT_K, weights=[word_vec], name="text_embeddng", trainable=True)(text_input)



    attrs_latent = Concatenate(axis=-1)([pl_embedding,time_embedding,text_embedding])
    attrs_latent=TrigPosEmbedding(output_dim=pl_d*3, name="postion_embd")(attrs_latent)
    self_all = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                               kernel_regularizer=keras.regularizers.l2(1e-6))(attrs_latent)

    lstm_out = LSTM(hidden_neurons, return_sequences=True,name='lstm_layer1')(self_all)

    dense = Dense(place_dim + 1, name='dense')(lstm_out)
    out_vec = Add()([dense,user_embedding])
#    bnorm=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',moving_mean_initializer='zeros', moving_variance_initializer='ones', \
#          beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(out_vec)
    mnorm=ModeNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',moving_mean_initializer='zeros', moving_variance_initializer='ones', \
           beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(out_vec)
    pred = Activation('softmax')(mnorm)
    model = Model([pl_input,time_input,user_input,text_input], pred)

    sgd = optimizers.SGD(lr=learning_rate)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy')
    model.summary()
    return model