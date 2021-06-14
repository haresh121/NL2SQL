from sklearn.metrics import accuracy_score
import numpy as np
import json
import pickle
from os import path
from itertools import chain

from src.constants import const


def load_object(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_object(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_data(data_dir, split='train', debug=False):
    data_path = data_dir + split + '_tok.json'
    table_path = data_dir + split + '_tok.tables.json'
    db_path = data_dir + split + '.db'
    query_list = []
    sql_list = []
    table_data = {}
    with open(data_path) as f:
        for idx, line in enumerate(f):
            if debug and idx > const.DEBUG_DATA_SIZE:
                break
            data = json.loads(line.strip())
            query_list.append(data['question_tok'])
            sql_list.append(data['sql'])
    with open(table_path) as f:
        for _, line in enumerate(f):
            t_data = json.loads(line.strip())
            table_data[t_data['id']] = t_data
    return query_list, sql_list, table_data, db_path


def get_numpy(arr, gpu=False):
    if gpu:
        return arr.data.cuda().cpu()
    else:
        return arr.data.cpu().numpy()


def accuracy(true_output, logits, gpu=False):
    logits = get_numpy(logits, gpu)
    predicted_output = np.argmax(logits, 1)
    true_output = get_numpy(true_output, gpu)
    return accuracy_score(true_output, predicted_output)


def append(a, b, gpu=False):
    b = get_numpy(b, gpu)
    return np.append(a, b)