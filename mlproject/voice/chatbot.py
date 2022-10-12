from konlpy.tag import Okt
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import tensorflow as tf
import warnings
import numpy as np
import re
import pandas as pd
chatbot_data = pd.read_csv('ChatbotData_.csv', encoding='utf-8')
question, answer = list(chatbot_data['Q']), list(chatbot_data['A'])
texts = []
pairs = []
for i, (text, pair) in enumerate(zip(question, answer)):
    texts.append(text)
    pairs.append(pair)
    if i >= 5000:
        break


def clean_sentence(sentence):
    sentence = re.sub(r'[^0-9ㄱ-ㅎㅏ-ㅣ가-힣 ]', r'', sentence)
    return sentence


okt = Okt()


def process_morph(sentence):
    return ' '.join(okt.morphs(sentence))


# 문장 전처리
def clean_and_morph(sentence, is_question=True):
    sentence = clean_sentence(sentence)
    sentence = process_morph(sentence)
    if is_question:
        return sentence
    else:
        return ('<START> ' + sentence, sentence + ' <END>')


def preprocess(texts, pairs):
    questions = []
    answer_in = []
    answer_out = []
    for text in texts:
        question = clean_and_morph(text, is_question=True)
        questions.append(question)

    for pair in pairs:
        in_, out_ = clean_and_morph(pair, is_question=False)
        answer_in.append(in_)
        answer_out.append(out_)

    return questions, answer_in, answer_out


questions, answer_in, answer_out = preprocess(texts, pairs)

all_sentences = questions + answer_in + answer_out


warnings.filterwarnings('ignore')


tokenizer = Tokenizer(filters='', lower=False, oov_token='<OOV>')
tokenizer.fit_on_texts(all_sentences)


# 치환: 텍스트를 시퀀스로 인코딩 (texts_to_sequences)
question_sequence = tokenizer.texts_to_sequences(questions)
answer_in_sequence = tokenizer.texts_to_sequences(answer_in)
answer_out_sequence = tokenizer.texts_to_sequences(answer_out)

# 문장의 길이 맞추기 (pad_sequences)
MAX_LENGTH = 30
question_padded = pad_sequences(question_sequence,
                                maxlen=MAX_LENGTH,
                                truncating='post',
                                padding='post')
answer_in_padded = pad_sequences(answer_in_sequence,
                                 maxlen=MAX_LENGTH,
                                 truncating='post',
                                 padding='post')
answer_out_padded = pad_sequences(answer_out_sequence,
                                  maxlen=MAX_LENGTH,
                                  truncating='post',
                                  padding='post')

question_padded.shape, answer_in_padded.shape, answer_out_padded.shape

# 라이브러리 로드


# ## 학습용 인코더 (Encoder)

# 인코더
class Encoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, time_steps):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size,
                                   embedding_dim,
                                   input_length=time_steps)
        self.dropout = Dropout(0.2)
        self.lstm = LSTM(units, return_state=True)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x, hidden_state, cell_state = self.lstm(x)
        return [hidden_state, cell_state]


# ## 학습용 디코더 (Decoder)

# 디코더
class Decoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, time_steps):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size,
                                   embedding_dim,
                                   input_length=time_steps)
        self.dropout = Dropout(0.2)
        self.lstm = LSTM(units,
                         return_state=True,
                         return_sequences=True,
                         )
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        x = self.dropout(x)
        x, hidden_state, cell_state = self.lstm(x, initial_state=initial_state)
        x = self.dense(x)
        return x, hidden_state, cell_state


# 모델 결합
class Seq2Seq(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim, time_steps, start_token, end_token):
        super(Seq2Seq, self).__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.time_steps = time_steps

        self.encoder = Encoder(units, vocab_size, embedding_dim, time_steps)
        self.decoder = Decoder(units, vocab_size, embedding_dim, time_steps)

    def call(self, inputs, training=True):
        if training:
            encoder_inputs, decoder_inputs = inputs
            context_vector = self.encoder(encoder_inputs)
            decoder_outputs, _, _ = self.decoder(inputs=decoder_inputs,
                                                 initial_state=context_vector)
            return decoder_outputs
        else:
            context_vector = self.encoder(inputs)
            target_seq = tf.constant([[self.start_token]], dtype=tf.float32)
            results = tf.TensorArray(tf.int32, self.time_steps)

            for i in tf.range(self.time_steps):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(target_seq,
                                                                            initial_state=context_vector)
                decoder_output = tf.cast(tf.argmax(decoder_output, axis=-1),
                                         dtype=tf.int32)
                decoder_output = tf.reshape(decoder_output, shape=(1, 1))
                results = results.write(i, decoder_output)

                if decoder_output == self.end_token:
                    break

                target_seq = decoder_output
                context_vector = [decoder_hidden, decoder_cell]

            return tf.reshape(results.stack(), shape=(1, self.time_steps))


# ## 단어별 원핫인코딩 적용

VOCAB_SIZE = len(tokenizer.word_index)+1


def convert_to_one_hot(padded):
    one_hot_vector = np.zeros((len(answer_out_padded),
                               MAX_LENGTH,
                               VOCAB_SIZE))
    for i, sequence in enumerate(answer_out_padded):
        for j, index in enumerate(sequence):
            one_hot_vector[i, j, index] = 1

    return one_hot_vector


answer_in_one_hot = convert_to_one_hot(answer_in_padded)
answer_out_one_hot = convert_to_one_hot(answer_out_padded)

# 변환된 index를 다시 단어로 변환


def convert_index_to_text(indexs, end_token):
    sentence = ''
    for index in indexs:
        if index == end_token:
            break
        if index > 0 and tokenizer.index_word[index] is not None:
            sentence += tokenizer.index_word[index]
        else:
            sentence += ''

        # 빈칸 추가
        sentence += ' '
    return sentence


# 하이퍼 파라미터 정의
BUFFER_SIZE = 1000
BATCH_SIZE = 16
EMBEDDING_DIM = 100
TIME_STEPS = MAX_LENGTH
START_TOKEN = tokenizer.word_index['<START>']
END_TOKEN = tokenizer.word_index['<END>']
UNITS = 128
VOCAB_SIZE = len(tokenizer.word_index)+1
DATA_LENGTH = len(questions)
SAMPLE_SIZE = 3
NUM_EPOCHS = 20

# 체크포인트 생성
checkpoint_path = 'model/seq2seq-chatbot-no-attention-checkpoint.ckpt'
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='loss',
                             verbose=1
                             )

# seq2seq
seq2seq = Seq2Seq(UNITS,
                  VOCAB_SIZE,
                  EMBEDDING_DIM,
                  TIME_STEPS,
                  START_TOKEN,
                  END_TOKEN)

seq2seq.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['acc'])

seq2seq.load_weights(checkpoint_path)


def make_prediction(model, question_inputs):
    results = model(inputs=question_inputs, training=False)
    results = np.asarray(results).reshape(-1)
    return results

# 자연어 (질문 입력) 대한 전처리 함수


def make_question(sentence):
    sentence = clean_and_morph(sentence)
    question_sequence = tokenizer.texts_to_sequences([sentence])
    question_padded = pad_sequences(
        question_sequence, maxlen=MAX_LENGTH, truncating='post', padding='post')
    return question_padded

# 챗봇


def run_chatbot(question):
    question_inputs = make_question(question)
    results = make_prediction(seq2seq, question_inputs)
    results = convert_index_to_text(results, END_TOKEN)
    return results


# 챗봇 실행
if __name__ == '__main__':
    while True:
        user_input = input('<< 말을 걸어 보세요!\n')
        if user_input == 'q':
            break
        print('>> 챗봇 응답: {}'.format(run_chatbot(user_input)))
