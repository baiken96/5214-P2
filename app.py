import flask
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np

ARTICLE_CHARS = 1000
ARTICLE_WORDS = 100

#text = open("corpus.txt", 'rb').read().decode(encoding='utf-8')
#vocab = np.array(sorted(set(text)))
#word_indices = dict((i,c) for i,c in enumerate(vocab))
#indices_words = dict((c,i) for c,i in enumerate(vocab))
#file = open('model/idx2char.txt', 'rb')
#idx2char = pickle.load(file)
#file = open('model/char2idx.txt', 'rb')
#char2idx= pickle.load(file)

def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

def generate_text(model, start_string, num_generate, char2idx, idx2char):
  # Evaluation step (generating text using the learned model)

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
    # hopefully the modeling and squeezing business will
    # make a bit more sense once i learn a bit more about
    # how tf models work and dissect this one in particular.
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    # if i had to guess, i'd say that the [-1,0] gets the last number
    # in the tensor, but idk how it works, you don't see syntax like
    # that in Python very often. i don't even know what its name is.
    # ...ok apparently it gets automatically converted to a tuple,
    # which means that Tensor's __getitem__ override is "special":
    # https://stackoverflow.com/questions/1957780/how-to-override-the-operator-in-python#1957793
    # ok i found out why:
    # "This operation extracts the specified region from the tensor.
    # The notation is similar to NumPy with the restriction that currently
    # only support basic indexing. That means that using a non-scalar tensor
    # as input is not currently allowed."
    # - from https://www.tensorflow.org/api_docs/python/tf/Tensor#__getitem__
    # ...ok, not finding this under basic indexing in the numpy docs:
    # https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing
    # ...but after playing around with it, it definitely means "first element
    # of last array", which is what we need here. i'll dig into it more if we
    # need fancier indexing at some later point.
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    # We pass the predicted character as the next input to the model
    # along with the previous hidden state
    # ...i.e. it gets passed to the model at the next loop iteration,
    # while the model has already ingested the start text.
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():

    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':

        prompt = flask.request.form['prompt']
        modeltype = "char"#flask.request.form['model']
        epochs = flask.request.form['epochs']
        outputsize = flask.request.form['outputsize']
        epochs=int(epochs)

        #text_generated = list()
        idx2char = None
        char2idx = None

        if modeltype == "char":
            model = tf.keras.models.load_model('model/char{}_model.h5'.format(epochs))
            file = open('model/idx2char_{}.txt'.format(epochs), 'rb')
            idx2char = pickle.load(file)
            file = open('model/char2idx_{}.txt'.format(epochs), 'rb')
            char2idx = pickle.load(file)
            output = generate_text(model, prompt, int(outputsize), char2idx, idx2char)
        """
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
        """
        return flask.render_template('main.html', original_input={'prompt':prompt,
                                                                  'model':modeltype,
                                                                  'epochs':epochs,
                                                                  'outputsize':outputsize},
                                    result = output,)

    return(flask.render_template('main.html'))

if __name__ == '__main__':
    app.run()