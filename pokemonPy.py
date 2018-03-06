"""Pokemon combat predictor"""
import csv
import tensorflow as tf

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
            targets.append([1, 0] if combats[i].pop(2) == combats[i][0] else [0, 1])
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
    return combats, features, targets

def run():
    combats, features, targets = parse_file()
    print("combats[0:100]:", combats[:100])
    print("features[0:100]:", features[:100])
    print("targets[0:100]:", targets[:100])
    


if __name__ == '__main__':
    run()
