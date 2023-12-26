import pathlib
import textwrap

import google.generativeai as genai
import os



class Gemini:
    def __init__(self) -> None:

        GOOGLE_API_KEY=os.environ['GOOGLE_API_KEY']

        genai.configure(api_key=GOOGLE_API_KEY)

        self.model = genai.GenerativeModel('gemini-pro')
        self.reset_history()

        # response = model.generate_content("Convert this utterance to sql: what is the minimum, average, and maximum age of all singers from France? given database scheme: | concert_singer | stadium : stadium_id , location , name , capacity , highest , lowest , average | singer : singer_id , name , country ( France ) , song_name , song_release_year , age , is_male | concert : concert_id , concert_name , theme , stadium_id , year | singer_in_concert : concert_id , singer_id")
        # print(response.text)

    def reset_history(self):
        self.chat = self.model.start_chat(history=[])

    def ask(self, ques):
        return self.model.generate_content(ques).text
    
    def fine_tune_sql(self, ques, res, history_context=True):
        schema = f"""
        ### Given the SQL query:
        "{res}"

        ### Replace the placeholder "1" in the query with the corresponding value mentioned by the user in their utterance.

        ### User's Utterance: 
        "{ques}"

        ### Modified SQL query (sql only, dont add description):
        """
        print("FINETUNING: ", schema)

        if history_context:
            print("WORKING WITH CHAT HISTORY")
            return self.chat.send_message(schema).text
        
        return self.model.generate_content(schema).text


    def predict_user_intent(self, database, chat):
        schema = """INFORM SQL The user informs his/her request
                if the users question can be answered by SQL. The
                system needs to write SQL. 
                \n
                INFER SQL If the users question must be an-
                swered by SQL+human inference. For example,
                users questions are are they..? (yes/no question) or
                the 3rd oldest.... SQL cannot directly (or unneces-
                sarily complicated) return the answer, but we can
                infer the answer based on the SQL results.
                \n
                AMBIGUOUS The users question is ambiguous (not specific like old, young, small, big, ...),
                the system needs to double check the user’s intent
                (e.g. what/did you mean by...?) or ask for which
                columns to return.
                \n
                AFFIRM Affirm something said by the system
                (user says yes/agree).
                \n
                NEGATE : Negate something said by the system
                (user says no/deny).
                \n
                NOT RELATED The users question is not related
                to the database, the system reminds the user.
                \n
                CANNOT UNDERSTAND The users question
                cannot be understood by the system, the system
                asks the user to rephrase or paraphrase question.
                \n
                CANNOT ANSWER The users question cannot be
                easily answered by SQL, the system tells the user
                its limitation.
                \n
                GREETING Greet the system.
                \n
                GOOD BYE Say goodbye to the system.
                \n
                THANK YOU Thank the system.
                """
        input = f"In a conversation of database querying, classify this user intent: [{chat}], database: [{database}], based on these labels: [{schema}]"

        print("calling: ", input)
        return self.model.generate_content(input).text
    
    def response_generation(self, chat):
        schema = f"Generate response as following (example: question: 'what is the arstist id?' -> your generation: 'the artist id is: '): \n {chat}"
        return self.ask(schema)

    def clarify(chat):
        pass
