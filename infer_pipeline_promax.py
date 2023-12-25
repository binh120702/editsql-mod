from parse_args import interpret_parser
import run_promax
import preprocess_promax
import postprocess_eval_promax
import shutil
import os 
from time import time
import json


# GLOVE_PATH="/home/dapici/glove.840B.300d.txt" 
# LOGDIR="logs/logs_cosql_editsql"
# SAVE_FILE=f"{LOGDIR}/save_12_cosql_editsql"
# ARGS = [
#     "--raw_train_filename", "data/cosql_data_removefrom/train.pkl",
#     "--raw_validation_filename", "data/cosql_data_removefrom/dev.pkl",
#     "--database_schema_filename", "data/cosql_data_removefrom/tables.json",
#     "--embedding_filename", f"{GLOVE_PATH}",
#     "--data_directory", "processed_data_cosql_removefrom",
#     "--input_key", "utterance",
#     "--state_positional_embeddings", "1",
#     "--discourse_level_lstm", "1",
#     "--use_utterance_attention", "1",
#     "--use_previous_query", "1",
#     "--use_query_attention", "1",
#     "--use_copy_switch", "1",
#     "--use_schema_encoder", "1",
#     "--use_schema_attention", "1",
#     "--use_encoder_attention", "1",
#     "--use_bert", "1",
#     "--bert_type_abb", "uS",
#     "--fine_tune_bert", "1",
#     "--use_schema_self_attention", "1",
#     "--use_schema_encoder_2", "1",
#     "--interaction_level", "1",
#     "--reweight_batch", "1",
#     "--freeze", "1",
#     "--logdir", f"{LOGDIR}",
#     "--evaluate", "1",
#     "--evaluate_split", "valid",
#     "--use_predicted_queries", "1",
#     "--save_file", f"{SAVE_FILE}",
#     '--infer_only_dev', "1"
# ]

class InferModel:
    def __init__(self, config_path):        
        print("STARTING INIT")
        print("="*100)
        time_st = time()

        with open(config_path, "r") as f:
            config_all = json.load(f)
        config_args = config_all['args']


        if os.path.isdir(os.path.join(config_args[1] + 'processed_data_cosql_removefrom')):
            shutil.rmtree(os.path.join(config_args[1] + 'processed_data_cosql_removefrom'))
        
        editsql_parser = interpret_parser()
        self.args = editsql_parser.parse_args(config_args[:-2])
        self.args_infer = editsql_parser.parse_args(config_args)
        self.modify(self.args)
        self.modify(self.args_infer)
        schema_tokens, column_names, database_schemas = preprocess_promax.ready_for_preprocessing(self.args)

        self.schema_tokens = schema_tokens
        self.column_name = column_names
        self.database_schemas = database_schemas

        self.model = run_promax.get_model(params=self.args)
        print("INIT in ", time() - time_st)
    
    def modify(self, params):
        params.raw_train_filename = os.path.join(params.model_folder_path, params.raw_train_filename)
        params.raw_validation_filename = os.path.join(params.model_folder_path, params.raw_validation_filename)
        params.data_directory = os.path.join(params.model_folder_path, params.data_directory)
        params.logdir = os.path.join(params.model_folder_path, params.logdir)
        params.save_file = os.path.join(params.model_folder_path, params.save_file)
        params.database_schema_filename = os.path.join(params.model_folder_path, params.database_schema_filename)
        

    def init_data_infer(self, utterances, db_id):
        data = {}
        data['final'] = {'utterance': "", 'query':""}
        data['database_id'] = db_id
        data['interaction'] = [
            {
                'utterance': x,
                'utterance_toks': x.split(' '),
                "query": "select",
                "query_toks_no_value": ["select"] # random word make it not empty
            } for x in utterances
        ]
        return [data]

    def infer(self, db_id, utterances):
        print("STARTING INFER")
        print("="*100)
        time_st = time()

        if os.path.isdir(os.path.join(self.args.model_folder_path, 'processed_data_cosql_removefrom')):
            shutil.rmtree(os.path.join(self.args.model_folder_path, 'processed_data_cosql_removefrom'))

        # utterances: [ sent1, sent2, ...]
        json_like_data = self.init_data_infer(utterances, db_id)

        # this preprocess and save to pickle shiet in data/cosql_data_removefrom/dev.pkl
        preprocess_promax.preprocess(self.args, json_like_data, self.schema_tokens, self.column_name, self.database_schemas)

        # inference: this create a step 1 prediction (edit-sql strategy)
        run_promax.infer_with_model(self.args_infer, self.model)

        # inference: step 2: edit previos predictions to get human-readable sentences
        print("INFER IN:", time() - time_st)
        return postprocess_eval_promax.get_inference(self.args, self.database_schemas)


def main():
    infer_model = InferModel()
    result = []
    result.append(infer_model.infer('product_catalog', ['What are the names of catalog with number 8']))
    result.append(infer_model.infer('car_1', ['hello blahskdjasdk, what are the maximum price of all car models ?']))
    result.append(infer_model.infer('news_report', ['show me the journalists from England | Do you want their names? | Yes, I do']))
    for i in result:
        print(i, end = '\n\n')

if __name__ == '__main__':
    main()