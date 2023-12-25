import os
import json

def create_ids(json_file, db_id_file):
    with open(json_file) as f:
        data = json.load(f)

    db = []
    for i, interaction in enumerate(data):
        db.append(interaction['database_id'])

    with open(db_id_file, 'w') as f:
        for i in set(db):
            f.write(i+'\n')

train_json = 'train.json'
train_db_id = 'train_db_ids.txt'
create_ids(train_json, train_db_id)

dev_json = 'dev.json'
dev_db_id = 'dev_db_ids.txt'
create_ids(dev_json, dev_db_id)