import json
import os
import time 

st = time.time()

utterances = [
    "Can you list the names and ids of courses?",
    "Can you list the names of courses only?"
]

db_id = 'college_2'

data = {}
data['final'] = {'utterance': "", 'query':""}
data['database_id'] = db_id
data['interaction'] = [
    {
        'utterance': x,
        'utterance_toks': x.split(' '),
        "query": "select",
        "query_toks_no_value": ["select"]
    } for x in utterances
]

json_object = json.dumps([data], indent=4)
with open('/home/dapici/editsql/data/data_pi/task1/dev.json', "w") as f:
    f.write(json_object)

os.system("python3 preprocess.py --dataset=cosql --pi_working --remove_from")


print(time.time() - st)

print("="*80)

os.system("rm -rf /home/dapici/editsql/processed_data_cosql_removefrom")

os.system("bash test_cosql_editsql.sh")

print(time.time() - st)


