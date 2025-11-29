import requests
import json
from google import genai
from google.genai import types
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score



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
            chatname = input("\nEnter a filename to save chat: ")
            filename = chatname+".txt" if not chatname.endswith(".txt") else chatname

            with open(filename, 'w') as file, open("master_corpus.txt", 'a') as master:
                for message in history:
                    role = message.role.capitalize()
                    content = message.parts[0].text
                    file.write(f"[{role}]: {content}\n---\n")
                    master.write(f"[{role}]: {content}\n---\n")
                master.write("\n****END OF CHAT****\n")
        else:
            print("No content to save")
            return None    
        print(f"chat saved to: {filename}")
        return filename
    
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
            ("tfidf", TfidfVectorizer()),
            ("reg", LogisticRegression(max_iter=1000))
        ])

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
            print("4. Exit")

            corpus_choice = input("Choose a number to get corpus details ").strip()

            if corpus_choice == '1':
                df = pd.read_parquet("hf://datasets/sentence-transformers/natural-questions/pair/train-00000-of-00001.parquet")
            elif corpus_choice == '4':
                break
            else:
                print("Invalid Choice")
                break

            while True:
                print(f"{len(df)} rows of question/answer data")
                print(f"Questions are human generated and look like: \n{df.iloc[0]["query"]}")
                print(f"Answers are from wikipedia and look like: \n{".".join(df.iloc[0]["answer"].split('.')[:2])}...") #only first two sentences
            
                continue_button = input("\nType 'back' to go back, or anything else to continue ")
                if continue_button == 'back':
                    break

                questions = list(df["query"])
                tag_yes_or_no = input("\nWould you like to append a tag onto all your queries? ie. 'put in an academic tone' or 'explain it like I'm 5' - type 'yes' for yes and anything else for no ")
                if tag_yes_or_no == 'yes':
                    tag = input("Type your tag here: ")
                    questions = [q + "? " + tag for q in questions]

                chat_responses = []
                for q in questions:
                    try:
                        response = chat.send_message(q)
                        chat_responses.append(response.text)
                    except Exception as e:
                        print("API connection timed out: {e}")
                        break

                data_responses = list(df["answer"])[:len(chat_responses)]
                acc,f1 = self.train_classifier(chat_responses,data_responses)
                print("\nAbility to differentiate between chat generated responses and text from data: ")
                print(f"Accuracy: {acc}")

                history = chat.get_history()

                if history:
                    chatname = input("\nEnter a filename to save chat: ")
                    filename = chatname+".txt" if not chatname.endswith(".txt") else chatname

                    with open(filename, 'w') as file, open("master_corpus.txt", 'a') as master:
                        for message in history:
                            role = message.role.capitalize()
                            content = message.parts[0].text
                            file.write(f"[{role}]: {content}\n---\n")
                            master.write(f"[{role}]: {content}\n---\n")
                        master.write("\n****END OF CHAT****\n")
                else:
                    print("No content to save")
                    return None    
                
                print(f"chat saved to: {filename}")
                return filename
class GeminiChatCloud:
    def __init__(self, cloud_uri):
        self.url = cloud_uri
        self.history = []
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
            chatname = input("\nEnter a filename to save chat: ")
            filename = chatname+".txt" if not chatname.endswith(".txt") else chatname

            with open(filename, 'w') as file, open("master_corpus.txt", 'a') as master:
                for message in self.history:
                    role = message['role'].capitalize()
                    content = message['content']
                    file.write(f"[{role}]: {content}\n---\n")
                    master.write(f"[{role}]: {content}\n---\n")
                master.write("\n****END OF CHAT****\n")
        else:
            print("No content to save")
            return None    
        print(f"chat saved to: {filename}")
        return filename
