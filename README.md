# Comment Toxicity Detection Model

This project uses a deep learning approach to detect toxic comments using TensorFlow and Keras. The notebook walks through data loading, preprocessing, model building, training, evaluation, and deployment with Gradio.

## Table of Contents

- [Setup](#setup)
- [Data Loading](#data-loading)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing](#preprocessing)
- [Text Vectorization](#text-vectorization)
- [Dataset Preparation](#dataset-preparation)
- [Train/Validation/Test Split](#trainvalidationtest-split)
- [Model Building](#model-building)
- [Model Training](#model-training)
- [Performance Visualization](#performance-visualization)
- [Prediction](#prediction)
- [Evaluation](#evaluation)
- [Deployment with Gradio](#deployment-with-gradio)

---

## Setup

```python
import os
import pandas as pd
import tensorflow as tf
import numpy as np
```
**Explanation:**  
Import necessary libraries for file handling, data manipulation, deep learning, and numerical operations.

---

## Data Loading

```python
df = pd.read_csv('train.csv', on_bad_lines="skip")
```
**Explanation:**  
Load the training data from `train.csv` into a pandas DataFrame, skipping any problematic lines.

---

## Exploratory Data Analysis

```python
df.head()
df.shape
```
**Explanation:**  
View the first few rows and the shape of the dataset to understand its structure.

---

## Preprocessing

```python
from tensorflow.keras.layers import TextVectorization
```
**Explanation:**  
Import the `TextVectorization` layer for converting text to integer sequences.

---

## Text Vectorization

```python
X = df['comment_text']
y = df[df.columns[2:]].values
MAX_FEATURES = 200000 # number of words in vocab
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(X.values)
vectorizer.get_vocabulary()
vectorizer('Hello world, life is great')[:5]
vectorized_text = vectorizer(X.values)
```
**Explanation:**  
- Extract comment texts and labels.
- Set vocabulary size and sequence length.
- Create and adapt the vectorizer to the dataset.
- Preview vocabulary and vectorization output.

---

## Dataset Preparation

```python
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)
```
**Explanation:**  
Create a TensorFlow dataset, cache, shuffle, batch, and prefetch for efficient training.

---

## Train/Validation/Test Split

```python
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))
```
**Explanation:**  
Split the dataset into training, validation, and test sets.

---

## Model Building

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

model = Sequential()
model.add(Embedding(input_dim=MAX_FEATURES+1, output_dim=32, input_shape=(100,)))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='BinaryCrossentropy', optimizer='Adam')
model.summary()
```
**Explanation:**  
Build a sequential model with embedding, bidirectional LSTM, dense layers, and sigmoid output for multi-label classification. Compile and summarize the model.

---

## Model Training

```python
history = model.fit(train, epochs=5, validation_data=val)
```
**Explanation:**  
Train the model for 5 epochs using the training and validation sets.

---

## Performance Visualization

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()
```
**Explanation:**  
Plot training and validation metrics to visualize model performance.

---

## Prediction

```python
input_text = vectorizer('You freaking suck! I am going to kill you')
model.predict(np.expand_dims(input_text, 0))
```
**Explanation:**  
Vectorize a sample input and predict its toxicity using the trained model.

---

## Evaluation

```python
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator():
    X_true, y_true = batch
    yhat = model.predict(X_true)
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)

print(f'Precision: {pre.result().numpy()}, Recall : {re.result().numpy()}, Accuracy : {acc.result().numpy()}')
```
**Explanation:**  
Calculate precision, recall, and accuracy on the test set.

---

## Deployment with Gradio

```python
!pip install gradio jinja2
import gradio as gr

def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    text = ""
    for idx, col in enumerate(df.columns[2:-1]):
        text += f"{col}: {results[0][idx] > 0.5}\\n"
    return text

interface = gr.Interface(fn=score_comment,
                         inputs=gr.Textbox(lines=2, placeholder='Comment to score'),
                         outputs=gr.Textbox())
interface.launch(share=True)
```
**Explanation:**  
Install Gradio, define a scoring function, and launch a web interface for real-time toxicity prediction.

---

## Model Saving and Loading

```python
model.save('toxicity.h5')
model = tf.keras.models.load_model('toxicity.h5')
```
**Explanation:**  
Save and reload the trained model for future use.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.