import glob 
from google import genai
from google.genai import types
from openai import OpenAI
import nltk
from nltk.corpus import wordnet
from nltk.corpus import genesis
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.text import Text
from nltk.tag import pos_tag
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')


class Encorporator:
    #class that holds all the corpora related operations that can be performed on text
    
    #improved POS tagging:
    def get_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        if tag.startswith('V'):
            return wordnet.VERB
        if tag.startswith('N'):
            return wordnet.NOUN
        if tag.startswith('R'):
            return wordnet.ADV
        
        return wordnet.NOUN
    
    #load .txt file:
    def load_TXT(self, filename):
        with open(filename, 'r') as f:
            return f.read() 
    
    #split rawtext by sentence:
    def parse_sentence(self, rawtext)->list:
        sentences = rawtext.split('. ')
        return sentences

    #split rawtext by tokens:
    def parse_tokens(self, rawtext)->list:
        tokens = word_tokenize(rawtext)
        return tokens

    #run nltk FreqDist
    def get_fdist(self, tokens):
        fdist = FreqDist(tokens)
        return fdist
    
    #print entire FreqDist:
    def pretty_print_fdist(self, tokens):
        fdist = self.get_fdist(tokens)
        lines = []
        items = fdist.items()
        sorted_fdist = sorted(items, key=lambda item: item[1], reverse=True)
        longest = max(len(word) for word in fdist.keys())
        spacer = longest+5
        print("Frequency Distribution: ")
        for word, freq in sorted_fdist:
            line = f"{word:<{spacer}}: {freq:>2}{'|'*freq}" 
            print(line)   
            lines.append(line)
        return "\n".join(lines)
    
    #get sentiment analysis (returns a dict of categories and scores):
    def analyze_sentiment(self, rawtext)->dict:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(rawtext)
        return scores
    
    #primary function of class, uses all other functions and returns a tuple of all the parsed types in corpus:
    def encorporate(self ,rawtext):
        sentences = self.parse_sentence(rawtext)
        tokens = self.parse_tokens(rawtext)
        lower_tokens = [w.lower() for w in tokens]
        lemmatizer = WordNetLemmatizer()
        tagged = nltk.pos_tag(lower_tokens)
        lemmas = [lemmatizer.lemmatize(word, self.get_pos(tag)) for word, tag in tagged]
        return tagged, lemmas, tokens, sentences
    
    #allows call to encorporate directly on a filename:
    def encorporate_file(self, filename):
        rawtext = self.load_TXT(filename)
        return self.encorporate(rawtext)
    
    #saves chat to a master file of rawtext that is the master corpus
    #can be used for more broad questions about LLM language 
    #**this function may be obsolete, 
    #**I decided it was simpler to just append "master_corpus.txt" every time a chat is saved to file
    def master_corpus(self, rawtext):
        corpname = 'master_corpus.txt'
        with open(corpname, 'a') as file:
            file.write(rawtext + "\n****END CHAT****\n")



###################################DEEPSEEK API CONNECT############################################
class DeepSeekChat:
    def __init__(self, API):
        self.client = self.connect_to_deepseek(API)
    def connect_to_deepseek(self, API):
        deepseek_client = OpenAI(
            api_key=API,
            base_url="https://api.deepseek.com/v1")
        return deepseek_client
    def chat_loop(self, client):
        history = []
        print("Connected to DeepSeek LLM")
        print("Type 'Exit' to exit")
        while True:
            prompt = input("\nYou: ").strip()

            if prompt == "Exit":
                break

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages = history + [{"role": "user", "content": prompt}]
            )

            deepseek_reply = response.choices[0].message.content
            print(f"DeepSeek: {deepseek_reply}")

            history.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": deepseek_reply}
            ])
        
        #save the chat to both a new file, and append it to the master_corpus file:
        chatname = input("Name Chat: ")
        filename = chatname+".txt" if not chatname.endswith(".txt") else chatname
        with open(filename, 'w') as file, open("master_corpus.txt", 'a') as master:
            for message in history:
                role = message['role'].capitalize()
                content = message['content']
                file.write(f"[{role}]: {content}\n---\n")
                master.write(f"[{role}]: {content}\n---\n")
            master.write("\n****END OF CHAT****\n")

        print(f"Chat saved to {chatname}.txt")

        return filename

######################################GPT API CONNECT#############################################################
class GPTChat:
    def __init__(self, API):
        self.client = self.connect_to_gpt(API)
    def connect_to_gpt(self, API):
        try:
            gpt_client = OpenAI(
                api_key=API, 
                base_url="https://api.openai.com/v1" 
            )
            return gpt_client
        except Exception as e:
            print(f"Error, could not connect to gpt: {e}")
            return None
    def chat_loop(self, client):
        if not client:
            print("No connection established with GPT")
            return
        history = [
            {"role": "system", "content": "you are a helpful advisor"}
        ]
        
        gptmodel = "gpt-3.5-turbo"
        print(f"\nConnected to GPT 3.5 Turbo")
        print("Type 'Exit' to terminate chat and save to file")

        while True:
            prompt = input("\nYou: ").strip()
            if prompt == 'Exit':
                break
            try:
                response = client.chat.completions.create(
                    model=gptmodel,
                    messages=history + [{"role": "user", "content": prompt}]
                )
                gpt_reply = response.choices[0].message.content
                print(f"\nGPT: {gpt_reply}")

                history.extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": gpt_reply}
                ])
            except Exception as e:
                print("API Call Malfunction: {e}")
                break
        #save to file here
        if (len(history) > 1):
            chatname = input("\nenter name for chat: ")
            filename = chatname+"txt" if not chatname.lower().endswith(".txt") else chatname
            with open(filename, 'w') as file, open("master_corpus.txt", 'a') as master:
                for message in history:
                    role = message['role'].capitalize()
                    content = message['content']
                    file.write(f"[{role}]: {content}\n---\n")
                    master.write(f"[{role}]: {content}\n---\n")
                master.write("\n****END OF CHAT****\n")
            return filename
        else:
            ("no content to save to file")
            return None

################################################GEMINI API CONNECT########################################################
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


#############################################APP MANAGER CLASS#########################################################
###############################brings together all other classes with user interface###################################

class AppManager:
    def __init__(self):
        self.main_loop()
    
    #this menu selects the LLM model, or loading already saved files. Returns the filename of the chat or saved file:
    def MainMenu(self):
        print("LLM LANGUAGE ANALYZER")
        print("Select an LLM model to begin a chat, then after completing a chat the chat text will save to a file for analysis")
        print("To run analysis on all captured chats, load the Master Corpus")
        print("Main Menu: ")
        print("1. Chat with DeepSeek LLM")
        print("2. Chat with GPT 3.5 Turbo")
        print("3. Chat with Gemini 2.5 Flash **use this one**")
        print("4. Load file to analyze")
        print("5. Load Master Corpus")
        choice = input("Enter selection: ")

        #maybe return choice here and have functions for each option? 

        if choice == '1':
            API = input("Enter API Key to connect with DeepSeek: ")
            dschat = DeepSeekChat(API)
            dsclient = dschat.connect_to_deepseek(API)
            filename = dschat.chat_loop(dsclient)
            return filename
        if choice == '2':
            API = input("Enter API Key to connect with GPT: ")
            gptchat = GPTChat(API)
            gptclient = gptchat.connect_to_gpt(API)
            filename = gptchat.chat_loop(gptclient)
            return filename
        if choice == '3':
            API = input("Enter API Key to connect to Gemini: ")
            geminichat = GeminiChat(API)
            print("1. Live chat with the model")
            print("2. Generate larger corpus from question/answer data")
            live_or_generated = input("Enter selection: ").strip()
            if live_or_generated == '1':
                filename = geminichat.chat_loop()
            elif live_or_generated == '2':
                filename = geminichat.generate_larger_corpus()
            return filename
        if choice == '4':
            filename = self.load_file("txt")
            return filename
        if choice == '5':
            return "master_corpus.txt"
        
    def load_file(self, extension):
        #using regex style pattern matching to search the current directory for any .txt files
            ##**i want to add the ability to search in other places, load files not in current directory
        pattern = f"*.{extension}"
        savedfiles = glob.glob(pattern)

        if not savedfiles:
            print("no files with extension '.txt' found in current directory")
            return None
        
        print("\nselect file to load: ")
        filenames = {}
        for i, name in enumerate (savedfiles, 1):
            filenames[str(i)] = name
            print(f"{i}: {name}")

        #loop to select from the list of files in directory by their index, loops until 'x' or valid file choice:
        while True:
            num = input("\nEnter the number of the file you want to load for analysis, or enter 'x' to exit: ").strip()

            if num.lower() == 'x':
                return None
            
            if num in filenames:
                chosen_file = filenames[num]
                print(f"Loading file: {chosen_file}")
                return chosen_file
            else:
                print("invalid input, please enter just the number of the file to load: ")
        
####################################~~~MAIN APP INTERFACE~~~################################################
############################~~all direct user inputs flow from here~~#######################################

    def main_loop(self):
        filename = self.MainMenu()
        if not filename:
            print("No filename returned from main/no file created from chat")
            return
        
        #instance of Encorporator object to access those functions:
        encorporator_a = Encorporator()

        #rawtext:
        try:
            with open(filename, 'r') as file:
                rawtext = file.read()
        except FileNotFoundError:
            print(f"could not locate file to load")
        except IOError as e:
            print(f"Error: could not read data from file: {filename}; error: {e}")

        #basic corpus created from raw text, broken down into its parts:
        corpus = encorporator_a.encorporate(rawtext)
        pos_tag_tokens = corpus[0]
        lemmas = corpus[1]
        tokens = corpus[2]
        sentences = corpus[3]
        print(f"What would you like to do with <{filename}>?")

        #here is the main analysis function of the 
        while True:
            print("LANGUAGE ANALYSIS MENU")
            print("choose from the following options: ")
            print("1. Get a Frequency Distribution")
            print("2. Lemmatize text")
            print("3. Run Senitment Analysis")
            print("4. Print file as corpus (Lemmas, POS Tagged Tokens, List of Sentences)")
            print("5. Compare to another Corpus")
            print("6. Exit")

            choice = input("Enter your choice: ")

            if choice == '1':
                formatted_fdist = encorporator_a.pretty_print_fdist(tokens)
                action = input("enter filename to save to .txt file, enter 'back' to return to corpus analysis menu ").strip()

                if action.lower() == 'back':
                    continue
                
                filename = action+'.txt' if not action.endswith('.txt') else action
                with open(filename, 'w') as file:
                    file.write(formatted_fdist)
            
            if choice == '2':
                print("Lemmatized tokens :")
                for index, lemma in enumerate(lemmas):
                    if (index % 10 == 0):
                        print("\n")
                    print(lemma, end=' ')
                
                action = input("enter filename to save lemmas to .txt file, enter 'back' to return to corpus analysis menu ").strip()
                
                if action.lower() == 'back':
                    continue

                filename = action+".txt" if not action.endswith(".txt") else action

                with open(filename, 'w') as file:
                    for index, lemma in enumerate(lemmas):
                        if index % 10 == 0:
                            file.write("\n")
                        file.write(lemma)
            
            if choice == '3':
                print("Sentiment Analysis")
                scores = encorporator_a.analyze_sentiment(rawtext)
                for category, score in scores.items():
                    print(f"{category}: {score}")

                action = input("enter filename to save to .txt file, enter 'back' to return to corpus analysis menu").strip()

                if action.lower() == 'back':
                    continue

                filename = action+".txt" if not action.endswith(".txt") else action

                with open(filename, 'w') as file:
                    for category, score in scores.items():
                        file.write(f"\n{category}: {score}")
            
            if choice == '4':
                print("Lemmas: ")
                for index, lemma in enumerate(lemmas):
                    if (index % 10 == 0):
                        print("\n")
                    print(lemma, end=' ')

                print("tagged tokens: ")
                num_per_line = 10
                formatted_tagged_tokens = [f"{word}[{tag}]" for word, tag in pos_tag_tokens]

                for i in range(0, len(formatted_tagged_tokens), num_per_line):
                    line = formatted_tagged_tokens[i:i+num_per_line]
                    print('|'.join(line))

                print("Sentences: ")
                for index, s in enumerate(sentences):
                    print(f"\n{index}::{s}")

            if choice == '5':
                print("COMPARE CORPORA")
                print("choose from the following options:")
                print("1. Use the entire brown corpus")
                print("2. Choose specific subcorpora from the brown corpus")
                corpus_choice = input("Enter selection: ").strip()
                brown = nltk.corpus.brown
                if corpus_choice == '1':
                    brown_raw = brown.raw()
                elif corpus_choice == '2':
                    print("choose from the following options, separated by a space:")
                    subcorpora = {}
                    for n, cat in enumerate(brown.categories()):
                        print(f"{n}: {cat}")
                        subcorpora[n] = cat
    
                    subcorpus_choices = input("Enter selection: ").strip().split()
                    categories = [subcorpora[int(c)] for c in set(subcorpus_choices)]
                    brown_raw = brown.raw(categories=categories)

                tagged_br, lemmas_br, tokens_br, sentences_br = encorporator_a.encorporate(brown_raw)
                tagged, lemmas, tokens, sentences = encorporator_a.encorporate(rawtext)

                print(f"Type to Token ratio brown: {len(set(tokens_br)/len(tokens_br))}")
                print(f"Type to Token ratio file: {len(set(tokens)/len(tokens))}")

                fdist_br = encorporator_a.get_fdist(tokens_br)
                fdist = encorporator_a.get_fdist(tokens)
                
                # do some other comparisons here


            if choice == '6':
                print("exiting...")
                return

################################MAIN FUNCTION CALL (DUNDER MAIN)###############################################
# triggers the whole program:       
if __name__ == "__main__":
    llmapp = AppManager(); 









    



        

  
