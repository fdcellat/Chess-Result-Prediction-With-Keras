# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 23:02:54 2022

@author: fdcel
"""


from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup
import pydotplus
from keras.utils import to_categorical

df = pd.read_csv("games.csv")
df2=pd.read_csv("games_with_shortname.csv")

df["opening_name"]=df2["opening_shortname"]

df = df.drop(df[df.winner == "draw"].index)

labelencoder_x = LabelEncoder()
df["winner"]=labelencoder_x.fit_transform(df["winner"])

#Getting rid of all of the games that are one move only
df = df[df.moves.str.len() >= 6] #If you were wondering, I did not notice the 'turns' column until way after this lol

#Making a new Series in the dataframe to contain a list of all the moves
df['moves_list'] = df.moves.apply(lambda x: x.split())

#Add columns called opening_move and response, which are the first moves by white and black, respectively
df['opening_move'] = df.moves_list.apply(lambda x: x[0])
df['response'] = df.moves_list.apply(lambda x: x[1])

#Adding an opening_name column to name a few common opening moves
df["opening_name_det"]=df["opening_name"]
df['opening_name'] = df.moves_list.apply(lambda x: 'King\'s Pawn' if x[0] == 'e4' else ('Queen\'s Pawn' if x[0] == 'd4' else ('English' if x[0] == 'c4' else  'Other')))

games=df
games['increment_code']
games['game_category'] = games['increment_code'].str.split('+').str[0]
games['increment'] = games['increment_code'].str.split('+').str[1]

games['game_category'] = games['game_category'].astype(int)
games['increment'] = games['increment'].astype(int)

#This approach is more like a short-cut and not a very generalized approach. This will be fixed soon.
games['game_category'][games['game_category'] >= 10] = 11
games['game_category'][games['game_category'] < 3] = 2
games['game_category'][(games['game_category'] >= 3) & (games['game_category'] < 10)] = 5

games['game_category'].replace(11, 'rapid', inplace = True)
games['game_category'].replace(2, 'bullet', inplace = True)
games['game_category'].replace(5, 'blitz', inplace = True)

for i in range (1,4):
  df[f"hamle_{i}"]=df['moves'].str.split(' ',expand=True)[i-1]

df["hamleler"] = df["hamle_1"] + df["hamle_2"]+df["hamle_3"]  

dfx=games.drop(columns={"last_move_at","created_at", "id", "white_id", "black_id","turns","winner","moves","increment","increment_code","moves_list","opening_name","response","hamle_1","hamle_2","hamle_3","hamleler","opening_move"})
dfx['target']=games['winner']
dataframe=dfx

val_dataframe = dataframe.sample(frac=0.15)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()
    
    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    
    # Learn the statistics of the data
    normalizer.adapt(feature_ds)
    
    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")
    
    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    
    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)
    
    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature



dataframe["rated"]=labelencoder_x.fit_transform(dataframe["rated"])




# Categorical features encoded as integers
rated = keras.Input(shape=(1,), name="rated", dtype="int64")
opening_ply = keras.Input(shape=(1,), name="opening_ply", dtype="int64")

# Categorical feature encoded as string
victory_status = keras.Input(shape=(1,), name="victory_status", dtype="string")
game_category =keras.Input(shape=(1,), name="game_category", dtype="string")
opening_eco = keras.Input(shape=(1,), name="opening_eco", dtype="string")
opening_name_det = keras.Input(shape=(1,), name="opening_name_det", dtype="string")

# Numerical features

white_rating = keras.Input(shape=(1,), name="white_rating")
black_rating = keras.Input(shape=(1,), name="black_rating")

all_inputs = [
    rated,
    opening_ply,
    victory_status,
    game_category,
    opening_eco,
    opening_name_det,
    white_rating,
    black_rating
   ]


# Integer categorical features
rated_encoded = encode_categorical_feature(rated, "rated", train_ds, False)

opening_ply_encoded = encode_categorical_feature(opening_ply, "opening_ply", train_ds, False)

# String categorical features
victory_status_encoded = encode_categorical_feature(victory_status, "victory_status", train_ds, True)
game_category_encoded = encode_categorical_feature(game_category, "game_category", train_ds, True)
opening_eco_encoded=encode_categorical_feature(opening_eco, "opening_eco", train_ds, True)
opening_name_det_encoded=encode_categorical_feature(opening_name_det, "opening_name_det", train_ds, True)

# Numerical features
white_rating_encoded = encode_numerical_feature(white_rating, "white_rating", train_ds)
black_rating_encoded = encode_numerical_feature(black_rating, "black_rating", train_ds)

all_features = layers.concatenate(
    [
        rated_encoded,
        opening_ply_encoded,
        victory_status_encoded,
        game_category_encoded,
        opening_eco_encoded,
        opening_name_det_encoded,
        white_rating_encoded,
        black_rating_encoded,
    
    ]
)




x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

keras.utils.plot_model(model,show_shapes="Ture",rankdir="LR")


history=model.fit(train_ds, epochs=100, validation_data=val_ds)

pred_test= model.predict(val_ds)


scores = model.evaluate(train_ds, verbose=0)
scores2 = model.evaluate(val_ds, verbose=0)

print('Training Loss : {}% \nTraining Accuracy: {}%'.format(round((100 - 100*scores[1]),2),round((scores[1]*100),2)))
print('Test Loss:  {}% \nTest Accuracy:{}'.format( round((100 - 100*scores2[1]),2),round((scores2[1]*100),2)))


sample = {
    "rated": 1,
    "opening_ply": 3,
    "victory_status": "outoftime",
    "game_category" : "bullet",
    "opening_eco"  : "B20",
    "opening_name_det" : "Sicilian Defense",
    "white_rating" : 1300,
    "black_rating" : 1300,
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(input_dict)

print(
    "White winning percentage : %.1f ." % (100 * predictions[0][0],)
)



plot_metrics = ['loss', 'accuracy']

f, ax = plt.subplots(1,2,figsize = [12,4])
for p_i,metric in enumerate(plot_metrics):
    ax[p_i].plot(history.history[metric], label='Train ' + metric, )
    ax[p_i].plot(history.history['val_' + metric], label='Val ' + metric)
    ax[p_i].set_title("Loss Curve - {}".format(metric))
    ax[p_i].set_ylabel(metric.title())
    ax[p_i].legend()
plt.show()
