import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score

from model_fit import vectorize_data
from sklearn.feature_extraction.text import TfidfVectorizer
from model_fit import train_model
from data.data_processing import DisasterProcessor, EmotionProcessor

from net_fit import train_net

emotions_test = {0: "angry", 1: "fear", 2: "sad"}

# emotions_test = {0: "angry", 1:"afraid", 2:"happy",3:"in love",4:"sad",5:"surprised"}


def apply_model():
     X_emotion = ["I am sad and cannot sleep",
                  "I am joyful and happy and nothing can bring me down",
                  "I feel empty inside",
                  "My friends all hate me",
                  "How could she do this to me?",
                  "I am afraid of the dark",
                  "I am so mad that I can kill someone",
                  "I love you baby so much",
                  "I adore the color of your eyes"]
     # saddness = 4
     # anger = 0
     # love = 3
     # surprise = 5
     # fear = 1
     # joy = 2
     X = ["I love you baby",
          "I am okay very well today",
          "House is on fire",
          "airplane is crashing",
          "it is all relaxing here and nice",
          "call 911 there was a car crash",
          "I miss my parents, going to fly soon",
          "ukraine invasion war russia",
          "my neighbor just got murdered call 911",
          "someone killed my neighbor",
          "alarm went off someone broke into my house",
          "missiles hit ukraine kiev tonight"]

     y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
     y_emotion = [4,2,4,4,0,1,0,3,3]

     vectorizer = pickle.load(open("outputs/vectorizer.sav", "rb"))

     X = vectorizer.transform(X_emotion).toarray()

     loaded_model = pickle.load(open('outputs/model.sav', 'rb'))

     y_pred = loaded_model.predict(X)

     print(y_pred)

     print(accuracy_score(y_pred, y_emotion))

     # disaster_res = pd.DataFrame([X], columns = ["text"])
     # disaster_res["predicted"] = y_pred
     # disaster_res["actual"] = y_true
     # disaster_res.to_csv("data/Predictions.csv")

def apply_net():
     X_emotion = ["I am sad and cannot sleep",
                  # "I am joyful and happy and nothing can bring me down",
                  "I feel empty inside",
                  "My friends all hate me",
                  "How could she do this to me?",
                  "I am afraid of the dark",
                  "I want to run away from everybody",
                  "I have not eaten in days",
                  "Do not yell at me",
                  "Shut up and leave me alone !",
                  "I feel so depressed",
                  "My wife left me and the kids",
                  "life is not worth it anymore",
                  "I feel like shit and tired",
                  "I feel hopeless, everybody hates me",
                  "She does not love me back",
                  "My wife doesnt love me anymore",
                  "I have had enough of you",
                  "He keeps following me at night",
                  "I see ghosts everywhere"
                  # "I love you baby so much",
                  # "I adore the color of your eyes"
                  ]
     # saddness = 4
     # anger = 0
     # love = 3
     # surprise = 5
     # fear = 1
     # joy = 2

     with open('outputs/tokenizer.pickle', 'rb') as handle:
          tok = pickle.load(handle)

     sequences = tok.texts_to_sequences(X_emotion)

     X = pad_sequences(sequences, maxlen=200)

     X = X.astype(np.float)

     # X = X.reshape(X.shape[0], 1, X.shape[1])  # Comment to not reshape

     y_emotion = [2, 2, 2, 0, 1, 0, 2, 0, 0, 2, 2, 2, 0, 2, 0, 0, 2, 1, 1]

     model = tf.keras.models.load_model('outputs/network.h5')

     predictions = model.predict([X,X])

     y_hat = np.argmax(predictions, axis = 1)

     print(y_hat)

     for i in range(len(X_emotion)):
         print(X_emotion[i] + ":",emotions_test[y_hat[i]])
     print(accuracy_score(y_hat, y_emotion))

     # disaster_res = pd.DataFrame(np.array([X],dtype=str), columns=["text"])
     # disaster_res["predicted"] = y_pred
     # disaster_res["actual"] = y_true
     # disaster_res.to_csv("data/Predictions.csv")

if __name__ == "__main__":
     # try:
          np.random.seed(42)
          # train_model(EmotionProcessor)
          # apply_model()
          train_net(EmotionProcessor, reshape=False)
          apply_net()
     # except Exception as e:
     #      print(e)
