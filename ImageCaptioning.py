import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from keras.preprocessing import image, sequence
from keras.applications import VGG16
from keras.layers import Dense, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector, Merge
from keras.models import Sequential, Model
from keras.optimizers import Nadam

import sys
sys.setrecursionlimit(15000)

from tkinter import *             #arayuz
from tkinter.ttk import *
from tkinter import messagebox
window = Tk()
window.title("ImageCaptioning")
window.geometry('400x400')
lbl = Label(window, text="Enter Image Path : ")
lbl.grid(column=0, row=0)
txt = Entry(window,width=23)
txt.grid(column=3, row=0)




def clicked():         #arayuz
    img=txt.get()

    global window
    window.destroy()
btn = Button(window, text="Go", command=clicked)
btn.grid(column=4, row=4)
            
window.mainloop()

images_dir = os.listdir("./Flicker8k_Dataset/")

images_path = './Flicker8k_Dataset/'
captions_path = './Flickr8k.token.txt'
train_path = './Flickr_8k.trainImages.txt'
val_path = './Flickr_8k.devImages.txt'

captions = open(captions_path, 'r').read().split("\n")[:-1]


x_train = open(train_path, 'r').read().split("\n")[:-1]


x_test = open(val_path, 'r').read().split("\n")[:-1]

tokens = {}

for ix in range(len(captions)):
    temp = captions[ix].split("#")
    if temp[0] in tokens:
        tokens[temp[0]].append(temp[1][2:])
    else:
        tokens[temp[0]] = [temp[1][2:]]

temp = captions[100].split("#")

vgg = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))

def preprocess_input(img):
    img = img[:, :, :, ::-1] #RGB to BGR
    img[:, :, :, 0] -= 103.939 
    img[:, :, :, 1] -= 116.779
    img[:, :, :, 2] -= 123.68
    return img

def preprocessing(img_path):
    im = image.load_img(img_path, target_size=(224,224,3))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    return im

vgg = Model(inputs=vgg.input, outputs=vgg.layers[-2].output)

def get_encoding(model, img):
    image = preprocessing(images_path+img)
    pred = model.predict(image)
    pred = np.reshape(pred, pred.shape[1])
    return pred

pd_dataset = pd.read_csv("./flickr_8k_train_dataset.txt", delimiter='\t')
ds = pd_dataset.values


sentences = []
for ix in range(ds.shape[0]):
    sentences.append(ds[ix, 1])
    


words = [i.split() for i in sentences]

unique = []
for i in words:
    unique.extend(i)



unique = list(set(unique))


vocab_size = len(unique)


with open('word_2_indices','rb') as fp:
    word_2_indices=pickle.load(fp)
with open('indices_2_word','rb') as fp:
    indices_2_word=pickle.load(fp)



max_len = 0

for i in sentences:
    i = i.split()
    if len(i) > max_len:
        max_len = len(i)


captions = np.load("./captions.npy")
next_words = np.load("./next_words.npy")


images = np.load("./images.npy")



image_names = np.load("./image_names.npy")


embedding_size = 128

image_model = Sequential()      #we are setting the our training model here like lstm for rnn

image_model.add(Dense(embedding_size, input_shape=(4096,), activation='relu'))
image_model.add(RepeatVector(max_len))


language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))



model = Sequential()

model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
model.add(LSTM(1000, return_sequences=False))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.load_weights("./150Batch512model_weights.h5")
model.compile(loss='categorical_crossentropy', optimizer=Nadam(), metrics=['accuracy'])



#model.fit([images, captions], next_words, batch_size=512, epochs=150)

#model.save_weights("./150Batch512model_weights.h5")

#ortaimg = "./240583223_e26e17ee96.jpg"
img = "./3320032226_63390d74a6.jpg"
#ortaimg = "./IMG_7072.JPG"
#img = "./IMG_7072.JPG"
#img = "./IMG_3447.JPG"
#img  =  "./3623302162_099f983d58.JPG"
#kötüimg="./3679407035_708774de34.jpg"
#kötüimg="./3676432043_0ca418b861.jpg"

#ortaimg="./3585487286_ef9a8d4c56.jpg"
#cokiyiimg="./240696675_7d05193aa0.jpg"

#kotuimg="./2258662398_2797d0eca8.jpg"

#kotuimg="./2276120079_4f235470bc.jpg"
#kotuimg="./2343879696_59a82f496f.jpg"
#img="./2432061076_0955d52854.jpg"
#fenadegilimg="./2594459477_8ca0121a9a.jpg"
#img="./2646615552_3aeeb2473b.jpg"
#img="./2858439751_daa3a30ab8.jpg"
#img="./2908859957_e96c33c1e0.jpg"
from IPython.display import Image, display
z = Image(filename=images_path+img)
display(z)

test_img = get_encoding(vgg, img)



def beam_search_predictions(image, beam_index = 3):     #prediction function for the captions
    start = [word_2_indices["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            preds = model.predict([np.array([image]), np.array(par_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:] #Top n prediction
            
            for w in word_preds: #new list so as to feed it to model again
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:] # Top n words
    
    start_word = start_word[-1][0]
    intermediate_caption = [indices_2_word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = (' '.join(final_caption[1:]))
    return final_caption


Beam_Search_index_3 = beam_search_predictions(test_img, beam_index=3)
Beam_Search_index_5 = beam_search_predictions(test_img, beam_index=5)
Beam_Search_index_7 = beam_search_predictions(test_img, beam_index=7)
Beam_Search_index_9 = beam_search_predictions(test_img, beam_index=9)
Beam_Search_index_11 = beam_search_predictions(test_img, beam_index=11)



print ("Beam Search Prediction with Index = 3 : ",Beam_Search_index_3)
print ("Beam Search Prediction with Index = 5 : ",Beam_Search_index_5)
print ("Beam Search Prediction with Index = 7 : ",Beam_Search_index_7)
print ("Beam Search Prediction with Index = 9 : ",Beam_Search_index_9)
print ("Beam Search Prediction with Index = 11 : ",Beam_Search_index_11)




