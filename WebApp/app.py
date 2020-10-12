import flask
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np

ARTICLE_CHARS = 1000
ARTICLE_WORDS = 100

text = open("corpus.txt", 'rb').read().decode(encoding='utf-8')
vocab = np.array(sorted(set(text)))
word_indices = dict((i,c) for i,c in enumerate(vocab))
indices_words = dict((c,i) for c,i in enumerate(vocab))

def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():

    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':

        prompt = flask.request.form['prompt']
        modeltype = flask.request.form['model']

        text_generated = list()

        if modeltype == "char":
            with open(f'model/char_model.pkl', 'rb') as f:
                model = pickle.load(f)
            for i in range(ARTICLE_CHARS):
                predictions = model(prompt)
                predictions = tf.squeeze(predictions, 0)
                id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
                text_generated.append(vocab[id])
            output = ''.join(text_generated)
        else:
            with open(f'model/word_model.pkl', 'rb') as f:
                model = pickle.load(f)
            for i in range(ARTICLE_WORDS):
                predict = np.zeros((1, 10, vocab))
                for t, word in enumerate(prompt):
                    predict[0, t, word_indices] = 1.
                predictions = model.predict(predict, verbose=0)[0]
                nextindex = sample(predictions)
                nextword = indices_words[nextindex]
                text_generated.append(nextword)
            output = ''.join(text_generated)

        return flask.render_template('main.html', original_input={'prompt':prompt,
                                                                  'model':modeltype},
                                    result = output,)

    return(flask.render_template('main.html'))

if __name__ == '__main__':
    app.run()