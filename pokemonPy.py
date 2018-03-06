#!/usr/bin/env python3
"""Pokemon combat predictor"""
import csv
import tensorflow as tf
import random
import json
import datetime

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
    batch_x = json.loads(json.dumps(combats[:100]))
        
    for row in batch_x:
        batch_y.append([1., 0.] if row.pop(2) == row[0] else [0., 1.])
        row[0] = features[row[0] - 1]
        row[1] = features[row[1] - 1]
    return batch_x, batch_y 

def run():
    combats, features, tests = parse_file()
    lr = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32, [None, 2, 10]);
    Y_ = tf.placeholder(tf.float32, [None, 2])
    
    W1 = tf.Variable(tf.random_normal([20, 100]))
    B1 = tf.Variable(tf.ones([100]))
 
    W2 = tf.Variable(tf.random_normal([100, 80]))
    B2 = tf.Variable(tf.ones([80]))
    
    W3 = tf.Variable(tf.random_normal([80, 50]))
    B3 = tf.Variable(tf.ones([50]))
    
    W4 = tf.Variable(tf.random_normal([50, 20]))
    B4 = tf.Variable(tf.ones([20]))

    W5 = tf.Variable(tf.random_normal([20, 2]))
    B5 = tf.Variable(tf.ones([2]))

    X1 = tf.reshape(X, [-1, 20])

    Y1 = tf.nn.relu(tf.matmul(X1, W1) + B1)
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
    Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
    Y5 = tf.matmul(Y4, W5) + B5


    Y = tf.nn.softmax(Y5)

    tf.nn.dropout(Y1, 0.75) 
    tf.nn.dropout(Y2, 0.60)

#    cost = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))
    cost = tf.losses.softmax_cross_entropy(onehot_labels=Y_, logits=Y5)
    prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))

    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(lr)
    train = optimizer.minimize(cost)

    train_err = tf.summary.scalar("Train err", cost)
    train_acc = tf.summary.scalar("Train acc", accuracy)

    train_summary = tf.summary.merge([train_err, train_acc])
    filename = "./logs/poke_" + str(datetime.datetime.now())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(filename, graph=sess.graph)
    varlr = 0.003
    for e in range(epochs):
        batch_x, batch_y = get_batchs(combats, features)
        _, graphic = sess.run([train, train_summary], feed_dict={
                X: batch_x,
                Y_: batch_y,
                lr: varlr
            })
        writer.add_summary(graphic, e)

        if e % 10 == 0:
            batch_w, batch_v = get_batchs(tests, features)
            print("Accuracy:", sess.run(accuracy, feed_dict={
                    X: batch_w,
                    Y_: batch_v,
                    lr: varlr
                }))
    if varlr - 0.000001 > 0:
            varlr -= 0.000001

if __name__ == '__main__':
    run()
