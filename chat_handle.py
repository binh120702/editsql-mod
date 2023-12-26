from infer_pipeline_promax import InferModel
from gemini_ai import Gemini
import sqlparse

class ChatHandler:
    def __init__(self):
        print("Init model 1")
        self.model_state_tracking = InferModel('/home/dapici/editsql/infer_promax_config.json')

        print("Init model 2")
        self.model_llm = Gemini()

    def predict_user_intent(self, chat):
        pass 

    def generate_state_tracking(self, chat, db_id):
        pass 

    def system_generation_response(self, chat):
        pass 

    def convert_tables_to_text(self, tables):
        cols = {}
        for i in range(len(tables['table_names'])):
            cols[i] = []
        for i in tables["column_names_original"]:
            if i[0] >= 0:
                cols[i[0]].append(i[1])
        
        text = '| ' + tables['db_id'] + ' |'

        for id, name in enumerate(tables['table_names']):
            text = text + name + ' : ' 
            for i in cols[id]:
                text = text + i + ', '
            text = text[:-2] + '|'
        print(text)
        return text 

    def parse_sql(self, sql):
        if sql[0] == '"':
            sql = sql[1:]
        if sql[-1] == '"':
            sql = sql[:-1]
        
        if "sql" in sql and "```" in sql:
            sql = sql.replace("```", "")
            sql = sql.replace("sql", "", 1).strip()

        sql = sqlparse.format(sql, reindent=True)

        return f" ```sql \n {sql} \n ```"

    def new_conversation(self):
        self.model_llm.reset_history()

    def handle_chat(self, context_chat, pre_tables, chat):
        tables = self.convert_tables_to_text(pre_tables)
        ALL_LABELS = [  "INFORM SQL",
                        "INFER SQL",
                        "AMBIGUOUS",
                        "AFFIRM" ,
                        "NEGATE" ,
                        "NOT RELATED",
                        "CANNOT UNDERSTAND",
                        "CANNOT ANSWER", 
                        "GREETING", 
                        "GOOD BYE",
                        "THANK YOU"
                        ]
        label = self.model_llm.predict_user_intent(tables, context_chat + ' | (user): ' + chat)

        print(label)

        if label not in ALL_LABELS:
            return "Something wrong, lease try again!"
        
        if label == "INFORM SQL" or label == "INFER SQL":
            if label == "INFORM SQL":
                context = chat
            else:
                context = context_chat + ' | ' + chat

            raw_sql = self.model_state_tracking.infer(db_id=pre_tables['db_id'], utterances=[context])
            gud_sql = self.model_llm.fine_tune_sql(context, raw_sql[0])
            
            print("GUD SQL", gud_sql)
            if not "select" in gud_sql.lower():
                return gud_sql
            
            return self.model_llm.response_generation(chat) + '\n' + self.parse_sql(gud_sql)
        elif label == "AMBIGUOUS":
            return "The sentence is ambigous, please try again"
        else:
            return label