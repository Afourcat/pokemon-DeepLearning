#!/usr/bin/env python3
"""Pokemon combat predictor"""
import csv
import tensorflow as tf
import random

epochs = 10000
types = {'': 0, "Ghost":1, "Ground": 2, "Electric": 3, "Grass": 4,
        "Fire": 5, "Bug": 6, "Water": 7, "Dragon": 8, "Rock": 9, "Fairy": 10,
        "Steel": 11, "Flying": 12, "Psychic": 13, "Dark": 14, "Fighting": 15, "Normal": 16, "Poison": 17, "Ice": 18}

def parse_file():
    combats = []
    targets = []
    features = []
    with open("dataset/combats.csv", 'r') as csvfile:
        combats = csv.reader(csvfile)
        combats = list(combats)
        combats.pop(0)
        for row, i in zip(combats, range(len(combats))):
            combats[i] = list(map(int, row))
    with open("dataset/pokemon.csv", 'r') as csvfile:
        features = csv.reader(csvfile)
        features = list(features)
        features.pop(0)
        for row, i in zip(features, range(len(features))):
            row[11] = 0 if row[11] == 'False' else 1
            row[2] = types[row[2]]
            row[3] = types[row[3]]
            row.pop(0)
            row.pop(0)
            row = list(map(float, row))
    return combats[:40000], features, combats[40001:]

def get_batchs(combats, features):
    random.shuffle(combats)
    batch_y = []
    batch_x = combats[:100]
    print(batch_x)
    for row in batch_x:
        batch_y.append([1., 0.] if row.pop(2) == row[0] else [0., 1.])
        row[0] = features[row[0] + 1]
        row[1] = features[row[1] + 1]
    return batch_x, batch_y 

def run():
    combats, features, tests = parse_file()
    
    X = tf.placeholder(tf.float32, [None, 2, 10]);
    Y_ = tf.placeholder(tf.float32, [None, 2])
    
    W1 = tf.Variable(tf.random_normal([2, 10, 50]))
    B1 = tf.Variable(tf.zeros([50]))
 
    W2 = tf.Variable(tf.random_normal([50, 2]))
    B2 = tf.Variable(tf.zeros([2]))

    Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
    Y1 = tf.reshape(Y1, [-1, 50])
    Y2 = tf.matmul(Y1, W2) + B2

    Y = tf.nn.softmax(Y2)

    tf.nn.dropout(Y1, 0.75)
    
    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_, logits=Y2)
    prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))

    accurency = tf.reduce_mean(tf.cast(prediction, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
        batch_x, batch_y = get_batchs(combats, features)

        sess.run(train, feed_dict={
                X: batch_x,
                Y_: batch_y
            })
        if e % 500 == 0:
            batch_x, batch_y = get_batchs(tests, features)
            print("Accurency:", sess.run(accurency, feed_dict={
                    X: batch_x,
                    Y_: batch_y
                }))

























    
if __name__ == '__main__':
    run()
