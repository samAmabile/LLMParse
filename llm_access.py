import requests
import json
import csv
import os

from google import genai
from google.genai import types

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset, concatenate_datasets
from encorporator import Encorporator

#for storing corpora created from each save:
TESTS_FOLDER = "tests"

#for use in parsing files to store prompts/llm response separately:
parser = Encorporator()

class GeminiChat:
    def __init__(self, API, sys_instruct="you are a helpful assistant",gemini_model="gemini-2.5-flash"):
        self.API = API
        self.client = self.connect_to_gemini()
        self.sys_instruct = sys_instruct
        self.gemini_model = gemini_model

        self.config = types.GenerateContentConfig(
            system_instruction = self.sys_instruct
        )

    def connect_to_gemini(self):
        try:
            gemini_client = genai.Client(api_key=self.API)
            return gemini_client
        except Exception as e:
            print(f"Error, unable to connect to Gemini: {e}")
            return None
    
    #make folders if they do not yet exist:
    def verify_path(self, folder):
        os.makedirs(folder, exist_ok=True)


    def write_to_csv(self,history,live,tag):
        
        chatname = input("\nEnter a filename to save chat: ")
        filename = chatname+".csv" if not chatname.endswith(".csv") else chatname

        #added path to keep individual corpora in a subfolder:
        #masters stay in main folder for now
        #check for path,else make path:
        self.verify_path(TESTS_FOLDER)

        #add path to filename "tests/":
        path = os.path.join(TESTS_FOLDER, filename)
        #was crashing with no data so added this since my master csv was empty
        #should not affect anything else:
        try:
            chat_num = list(pd.read_csv('master_corpus.csv')["Chat"])[-1] + 1
        except (KeyError, IndexError):
            chat_num = 1
        
        #added utf-8 encoding to all file saving b/c some llm content was using weird chars:
        with open(path, 'w', newline='', encoding='utf-8') as file, open("master_corpus.csv", 'a', encoding='utf-8') as master:
            writer = csv.writer(file)
            writer.writerow(['Role','Content','Live?','Tag?'])
            master_writer = csv.writer(master)
            #moved up to try/except
            #chat_num = list(pd.read_csv('master_corpus.csv')["Chat"])[-1] + 1
            for message in history:
                role = message.role.capitalize()
                content = message.parts[0].text
                writer.writerow([role, content, live, tag])
                master_writer.writerow([role, content, live, tag, chat_num])
        
        #returns the path of the file now: tests/filename:
        return path 
    
    def chat_loop(self):
        if not self.client:
            print("Not connected to API")
            return
        
        try:
            chat = self.client.chats.create(
                model=self.gemini_model,
                config=self.config
            )
        except Exception as e:
            print(f"Error, could not initiate chat: {e}")
            return
        
        print("\nConnected to Gemini-2.5-flash")
        print("\nType 'Exit' to terminate chat and save to file")

        while True:
            prompt = input("\nYou: ").strip()
            if prompt == "Exit":
                break
            
            try:
                response = chat.send_message(prompt)
                gemini_reply = response.text
                print(f"\nGemini: {gemini_reply}")
            except Exception as e:
                print("API connection timed out: {e}")
                break
        
        history = chat.get_history()

        if history:
            csvname = self.write_to_csv(history,1,'')
            txtfile = csvname.replace('.csv', '.txt')
            

            #both filenames now have path preppended from write_to_csv function(tests/)
            

            with open(txtfile, 'w', encoding='utf-8') as file, open("master_chat_log.txt", 'a', encoding='utf-8') as master:
                for message in history:
                    role = message.role.capitalize()
                    content = message.parts[0].text
                    file.write(f"[{role}]: {content}\n---\n")
                    master.write(f"[{role}]: {content}\n---\n")
                master.write("\n****END OF CHAT****\n")

            #making standard names for promts vs llm_content:
            #remove path to adjust filenames:
            txtfile_base = os.path.basename(txtfile)
            promptfile = f"prompts_{txtfile_base}"
            modelfile = f"llm_content_{txtfile_base}"
                
            #tests/ path gets added back from parse_chat
            prompts, llm_content = parser.parse_chat(txtfile, promptfile, modelfile)
        else:
            print("No content to save")
            return None    
        print(f"chat saved to: {txtfile}")
        print(f"promtps saved to: {prompts}")
        print(f"LLM responses saved to {llm_content}")
        print(f"Aggregate prompt/response saved to: {csvname}")

        #return csv and llm_content for analysis, offer user choice:
        return csvname, llm_content
    
    def get_responses(self, questions, chat):
        chat_responses = []
        for q in questions:
            try:
                response = chat.send_message(q)
                chat_responses.append(response.text)
            except Exception as e:
                print("API connection timed out: {e}")
                break
        return chat_responses


    def train_classifier(self, chat_responses, data_responses):
        chat_df = pd.DataFrame({
            "text": chat_responses,
            "label": 0
        })
        data_df = pd.DataFrame({
            "text": data_responses,
            "label": 1
        })
        labeled_dataset = pd.concat([chat_df, data_df], ignore_index=True)
        shuffled_df = labeled_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df, test_df = train_test_split(shuffled_df, test_size=0.2, random_state=42)

        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000,
                ngram_range=(1,2), # unigrams and bigrams
                min_df=2
            )),
            ("logreg", LinearSVC(
                max_iter=200,
            ))])

        model.fit(train_df["text"],train_df["label"])
        preds = model.predict(test_df["text"])
        golds = test_df["label"]
        return accuracy_score(golds,preds), f1_score(golds,preds)
         
    def generate_larger_corpus(self):
        if not self.client:
            print("Not connected to API")
            return
        
        try:
            chat = self.client.chats.create(
                model=self.gemini_model,
                config=self.config
            )
        except Exception as e:
            print(f"Error, could not initiate chat: {e}")
            return
        
        while True:
            print("\nConnected to Gemini-2.5-flash")
            print("Generate a larger corpus from one of the following question/answer corpora: ")
            print("1. Hugging Face Natural Questions")
            print("2. MS Marco Dataset")
            print("3. Exit")
            

            corpus_choice = input("Choose a number to get corpus details ").strip()

            if corpus_choice == '3':
                break


            while True:
                if corpus_choice == '1': # wiki responses
                    df = pd.read_parquet("hf://datasets/sentence-transformers/natural-questions/pair/train-00000-of-00001.parquet")
                    df = df.sample(frac=1).reset_index(drop=True) # shuffle
                    print("\nNATURAL QUESTIONS DATASET")
                    print(f"{len(df)} rows of question/answer data (will only use max 500 to make corpus)")
                    print(f"Questions are human generated and look like: \n{df.iloc[0]["query"]}")
                    print(f"Answers are from wikipedia and look like: \n{".".join(df.iloc[0]["answer"].split('.')[:2])}...") # only first two sentences
                elif corpus_choice == '2': # human responses
                    ds = load_dataset("microsoft/ms_marco", "v1.1")
                    full_ds = concatenate_datasets([ds["train"],ds["test"],ds["validation"]])
                    df = full_ds.to_pandas()
                    df = df.sample(frac=1).reset_index(drop=True) # shuffle
                    df = df[df["answers"].apply(lambda x: len(x) > 0 and len(x[0].split()) > 3)] # only keep answers with more than 3 words
                    df["answer"] = df["answers"].apply(lambda x: x[0]) # get first answer, because ms marco gives a list
                    print("\nMS MARCO DATASET")
                    print(f"{len(df)} rows of question/answer data (will only use max 500 to make corpus)")
                    print(f"Questions are human generated and look like: \n{df.iloc[0]["query"]}")
                    print(f"Answers are human generated and look like: \n{df.iloc[0]["answer"]}")
            
                continue_button = input("\nType 'back' to go back, or anything else to continue ")
                if continue_button == 'back':
                    break

                questions = list(df["query"])[:500] # max 500
                #tag_yes_or_no = input("\nWould you like to append a tag onto all your queries? ie. 'put in an academic tone' or 'explain it like I'm 5' - type 'yes' for yes and anything else for no ")
                tag_choice = input("Type 0 to proceed, 1 to add a tag, or 2 to compare 2 or more tags ")
                
                if tag_choice == '1':
                    tag = input("Type your tag here: ")
                    questions_tagged = [q + "? " + tag for q in questions]
                    chat_responses = self.get_responses(questions_tagged, chat)
                    if len(chat_responses) == 0:
                        return None
                    data_responses = list(df["answer"])[:len(chat_responses)]
                    acc = self.train_classifier(chat_responses,data_responses)
                    print("\nAbility to differentiate between chat generated responses and text from data: ")
                    print(f"Accuracy: {acc}")

                    

                elif tag_choice == '2':
                    tags = input("Type the tags you want to use, separated by a comma ").strip().split(',')
                    base_responses = self.get_responses(questions, chat)
                    if len(base_responses) == 0:
                        return None
                    data_responses = list(df["answer"])[:len(base_responses)]
                    accuracies = {"base": self.train_classifier(base_responses,data_responses)}
                    for x in range(len(tags)):
                        qs = [q + "?" + tags[x] for q in questions]
                        responses = self.get_responses(qs, chat)
                        if len(responses) == 0:
                            return None
                        accuracies[tags[x]] = self.train_classifier(responses,data_responses)

                    print("\nAbility to differentiate between chat generated responses and text from data: ")
                    for x in accuracies:
                        print(f"Accuracy for {x}: {accuracies[x]}")
                    
                    

                else:
                    tag = ''


                history = chat.get_history()

                if history:

                    #keeping both options, leading with csv:
                    csvfile = self.write_to_csv(history,0,tag)
                    txtname = csvfile.replace('.csv', '.txt')

        
                    #writing to txt (this should be in a separate function):
                    with open(txtname, 'w', encoding='utf-8') as file, open("master_chat_log.txt", 'a', encoding='utf-8') as master:
                        for message in history:
                            role = message.role.capitalize()
                            content = message.parts[0].text
                            file.write(f"[{role}]: {content}\n---\n")
                            master.write(f"[{role}]: {content}\n---\n")
                        master.write("\n****END OF CHAT****\n")

                    #using the parser to parse the full chat into prompt and llm txt files:
                    #remove path to adjust filenames:
                    txtname_base = os.path.basename(txtname)
                    promptfile = f"prompts_{txtname_base}"
                    modelfile = f"llm_content_{txtname_base}"
                    
                    # tests\ path gets appended by parse_chat, so these have path affixed:
                    prompts, llm_content = parser.parse_chat(txtname, promptfile, modelfile)

                    #now we are saving all three, the full log, the prompts, and the llm responses, 
                    #additionally, prompts are appended to master_prompts.txt, llm to master_corpus.txt, 
                    # and csv to master_corpus.csv:
                    print(f"chat saved to: {txtname}")
                    print(f"prompts saved to: {prompts}")
                    print(f"LLM responses saved to: {llm_content}")
                    print(f"Aggregate of prompts,chat,tag saved to {csvfile}")

                    #returns a tuple of csv, txt where csv has all data, txt has llm only:
                    return csvfile, llm_content
                    #Note: the AppManager will check for tuple and ask user which to load
                    #if no tuple is found (some options return string) it just checks suffix and proceeds accordingly
                else:
                    print("No content to save")
                    return None    
                
                
                
class GeminiChatCloud:
    def __init__(self, cloud_uri):
        self.url = cloud_uri
        self.history = []

    def verify_path_cloud(self, folder):
        os.makedirs(folder, exist_ok=True)

    #updated write to csv to handle how cloud access parses chats:
    def write_to_csv_cloud(self,live,tag):
        chatname = input("\nEnter a filename to save chat: ")
        filename = chatname+".csv" if not chatname.endswith(".csv") else chatname
        
        #verify tests/ path, or make:
        self.verify_path_cloud(TESTS_FOLDER)

        #add path to filename: 
        path = os.path.join(TESTS_FOLDER, filename)
        #verify master_corpus.csv exists before saving:
        if os.path.exists('master_corpus.csv'):
            try:
                chat_num = list(pd.read_csv('master_corpus.csv')["Chat"])[-1] + 1
            except (KeyError, IndexError):
                chat_num = 1
        else:
            chat_num = 1

        #corpus saves to tests/ path, master still in main:
        with open(path, 'w', newline='', encoding='utf-8') as file, open("master_corpus.csv", 'a', encoding='utf-8') as master:
            writer = csv.writer(file)
            writer.writerow(['Role','Content','Live?','Tag?'])
            master_writer = csv.writer(master)
            #chat_num = list(pd.read_csv('master_corpus.csv')["Chat"])[-1] + 1
            #write the first line if no header, (for me, since no csv yet)
            for message in self.history:
                role = message['role'].capitalize()
                content = message['content']
                writer.writerow([role, content, live, tag])
                master_writer.writerow([role, content, live, tag, chat_num])
        
        #returns full path where file is saved:
        return path
    
    def chat_loop(self):
        if not self.url:
            print("error: could not connect to cloud api")
            return
        
        gemini_model = "gemini-2.5-flash"
        
        print("\nConnected to Gemini-2.5-flash")
        print("\nType 'Exit' to terminate chat and save to file")

        while True:
            prompt = input("\nYou: ").strip()
            if prompt == "Exit":
                break
            
            try:
                payload = {"prompt": prompt}

                response = requests.post(
                    self.url, 
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=60
                )

                response.raise_for_status()

                response_data = response.json()
                gemini_reply = response_data.get('response', 'Error: Proxy returned invalid format.')

                print(f"\nGemini: {gemini_reply}")

                self.history.append({"role": "user", "content": prompt})
                self.history.append({"role": "assistant", "content": gemini_reply})
            
            except requests.exceptions.HTTPError as errh:
                print(f"HTTP Error: {errh}")
                print(f"Proxy Response is: {response.text}")
                break
            except requests.exceptions.ConnectionError as errc:
                print(f"Connection Error: {errc}")
                break
            except requests.exceptions.Timeout as errt:
                print(f"Error, timeout: {errt}")
                break
            except requests.exceptions.RequestException as err:
                print(f"Unknown Error Happened: {err}")
                break
            except json.JSONDecodeError:
                print(f"could not decode content of JSON response from the proxy. Response was: {response.text}")
                break
            except Exception as e:
                print("API connection timed out: {e}")
                break
        
        if self.history:
            
            #write to csv file:
            csvfile = self.write_to_csv_cloud(1, '')
            txtfile = csvfile.replace('.csv', '.txt')
            
            #both filenames already have path preppended from the write_to_csv function
            
            #writing to txt file:
            with open(txtfile, 'w', encoding='utf-8') as file, open("master_chat_log.txt", 'a', encoding='utf-8') as master:
                for message in self.history:
                    role = message['role'].capitalize()
                    content = message['content']
                    file.write(f"[{role}]: {content}\n---\n")
                    master.write(f"[{role}]: {content}\n---\n")
                master.write("\n****END OF CHAT****\n")

                #remove tests\ path
                txtfile_base = os.path.basename(txtfile)
                promptfile = f"prompts_{txtfile_base}"
                modelfile = f"llm_content_{txtfile_base}"
                
                #parse chat adds tests\ path: 
                prompts, llm_content = parser.parse_chat(txtfile, promptfile, modelfile)
        else:
            print("No content to save")
            return None    
        print(f"chat saved to: {txtfile}")
        print(f"prompts saved to: {prompts}")
        print(f"LLM responses saved to: {llm_content}")
        print(f"Aggregate of prompt,response saved to: {csvfile}")

        return csvfile, llm_content
