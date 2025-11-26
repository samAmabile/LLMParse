import glob 
import requests
import json
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


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('brown')


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
    def pretty_print_fdist(self, tokens)->str:
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
    
    def pretty_print_sorted_fdist(self, hashmap)->str:
        lines = []
        for key, value in hashmap:
            line = f"{key:<10}: {value:>2}{'|'*value}"
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
    def __init__(self, API):
        self.client = self.connect_to_gemini(API)
    def connect_to_gemini(self, API):
        try:
            gemini_client = genai.Client(api_key=API)
            return gemini_client
        except Exception as e:
            print(f"Error, unable to connect to Gemini: {e}")
            return None
    def chat_loop(self, client):
        if not client:
            print("Not connected to API")
            return
        
        sys_instruct = "you are a helpful assistant"
        gemini_model = "gemini-2.5-flash"
        config = types.GenerateContentConfig(
            system_instruction = sys_instruct
        )
        try:
            chat = client.chats.create(
                model=gemini_model,
                config=config
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

#######################################GEMINI CLOUD CONNECT################################################
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
            print("Enter [1] to use API key")
            print("Enter [2] to connect via Cloud URL")
            connection = input("Enter your choice: ")
            if connection == '1':
                API = input("Enter API Key to connect to Gemini: ")
                geminichat = GeminiChat(API)
                geminiclient = geminichat.connect_to_gemini(API)
                filename = geminichat.chat_loop(geminiclient)
                return filename
            if connection == '2':
                uri = "https://us-central1-parsellm.cloudfunctions.net/gemini-proxy"
                geminichatcloud = GeminiChatCloud(uri)
                filename = geminichatcloud.chat_loop()
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
            print("5. Compare to another corpus")
            print("6. Exit")

            choice = input("Enter your choice: ")

            if choice == '1':
                formatted_fdist = encorporator_a.pretty_print_fdist(tokens)
                action = input("enter filename to save to .txt file, enter 'back' to return to corpus analysis menu").strip()

                if action.lower() == 'back':
                    continue
                
                filename = action+'.txt' if not action.endswith('.txt') else action
                with open(filename, 'w') as file:
                    file.write(formatted_fdist)
            
            elif choice == '2':
                print("Lemmatized tokens :")
                for index, lemma in enumerate(lemmas):
                    if (index % 10 == 0):
                        print("\n")
                    print(lemma, end=' ')
                
                action = input("enter filename to save lemmas to .txt file, enter 'back' to return to corpus analysis menu").strip()
                
                if action.lower() == 'back':
                    continue

                filename = action+".txt" if not action.endswith(".txt") else action

                with open(filename, 'w') as file:
                    for index, lemma in enumerate(lemmas):
                        if index % 10 == 0:
                            file.write("\n")
                        file.write(lemma)
            
            elif choice == '3':
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

            elif choice == '4':
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

            elif choice == '5':
                print("COMPARE CORPORA")
                print("choose from the following options:")
                while True:
                    print("1. Use the entire brown corpus")
                    print("2. Choose specific subcorpora from the brown corpus")
                    corpus_choice = input("Enter selection: ").strip()
                    brown = nltk.corpus.brown
                    if corpus_choice == '1':
                        print("Using sample of 10,0000 tokens from Brown")
                        brown_words = brown.words()
                        sampled = brown_words[:10000]
                        brown_raw = " ".join(sampled)
                        break
                    elif corpus_choice == '2':
                        print("choose from the following options, separated by a space:")
                        subcorpora = {}
                        for n, cat in enumerate(brown.categories()):
                            print(f"{n}: {cat}")
                            subcorpora[n] = cat
        
                        subcorpus_choices = input("Enter selection: ").strip().split()
                        categories = [subcorpora[int(c)] for c in set(subcorpus_choices)]
                        brown_raw = brown.raw(categories=categories)
                        break
                    else:
                        print("invalid input, enter '1' or '2':")
                        continue
                if brown_raw is None:
                    continue
                tagged_br, lemmas_br, tokens_br, sentences_br = encorporator_a.encorporate(brown_raw)
                tagged, lemmas, tokens, sentences = encorporator_a.encorporate(rawtext)
                len_set_br = len(set(tokens_br))
                len_br = len(tokens_br)
                len_set_cur = len(set(tokens))
                len_cur = len(tokens)
                ttr_br = len_set_br / len_br
                ttr_cur = len_set_cur / len_cur
                print(f"Type to Token ratio brown: {ttr_br}")
                print(f"Type to Token ratio file: {ttr_cur}")

                ################compare fdists:##############
                fdist_br = encorporator_a.get_fdist(tokens_br)
                sorted_fdist_br = sorted(fdist_br.items(), key=lambda item: item[1], reverse=True)

                fdist = encorporator_a.get_fdist(tokens)
                sorted_fdist = sorted(fdist.items(), key=lambda item: item[1], reverse=True)


                print("\n--------------------------------------------------------------")
                print()
                print("Brown Frequency Distribution: ")
                plot_br = encorporator_a.pretty_print_sorted_fdist(sorted_fdist_br)
                print(f"\n{plot_br}")
                print("\n--------------------------------------------------------------")
                print("Current Corpus Frequency Distribution: ")
                plot_cur = encorporator_a.pretty_print_sorted_fdist(sorted_fdist)
                print(f"\n{plot_cur}")
                # do some other comparisons here
                ################compare tag frequency########################
                tag_count_br = {}
                for word, tag in tagged_br:
                    if tag in tag_count_br:
                        tag_count_br[tag] += 1
                    else:
                        tag_count_br[tag] = 1
                tag_count_cur = {}
                for word, tag in tagged:
                    if tag in tag_count_cur:
                        tag_count_cur[tag] += 1
                    else:
                        tag_count_cur[tag] = 1
                sorted_tags_br = sorted(tag_count_br.items(), key=lambda item: item[1], reverse=True)
                sorted_tags_cur = sorted(tag_count_cur.items(), key=lambda item: item[1], reverse=True)
                print("\n----------------------------------------------------------------")
                print("Brown Part of Speech Tags: ")
                for tag, count in sorted_tags_br:
                    print(f"{tag}: {count}")
                print("------------------------------------------------------------------")
                print("Current Corpus Part of Speech Tags")
                for tag, count in sorted_tags_cur:
                    print(f"{tag}: {count}")
                ############top lemmas comparison################
                lemmas_dist_br = encorporator_a.pretty_print_fdist(lemmas_br)
                lemmas_dist_cur = encorporator_a.pretty_print_fdist(lemmas)
                print("\n---------------------------------------------------------------")
                print("Lemma Frequency Brown: ")
                print(lemmas_dist_br)
                print("------------------------------------------------------------------")
                print("Lemmas Frequency Current Corpus")
                print(lemmas_dist_cur)
                
                #for sentences, maybe we need a concordance/kwic or regex search of some kind

                action = input("Enter filename to save comparisons to file, enter 'back' to return to analysis menu: ").strip()
                if action == 'back':
                    break
                comparison_file = action+".txt" if not action.endswith(".txt") else action

                #bunch of strings to just hold all the above stuff for file saving.
                br_plot = f"Brown FreqDist: {plot_br}"
                cur_plot = f"{filename} FreqDist: {plot_cur}"
                formatted_tags_br = "Brown POS Tags: "+"\n".join((f"{tag}: {count}" for tag, count in sorted_tags_br))
                formatted_tags_cur = f"{filename} POS Tags: "+"\n".join((f"{tag}: {count}" for tag, count in sorted_tags_cur))
                brown_ttr = f"Type to Token ratio brown: {ttr_br}"
                cur_ttr = f"Type to Token ratio {filename}: {ttr_cur}"
                brown_lemmas = f"Brown Lemmas: {lemmas_dist_br}"
                cur_lemmas = f"{filename} Lemmas: {lemmas_dist_cur}"

                file_output = [
                    f"{filename} vs Brown Analysis:",
                    "\n"+"-"*25+"\n",
                    brown_ttr,
                    cur_ttr,
                    "\n"+"-"*25+"\n",
                    br_plot,
                    cur_plot,
                    "\n"+"-"*25+"\n",
                    formatted_tags_br,
                    formatted_tags_cur,
                    "\n"+"-"*25+"\n",
                    brown_lemmas,
                    cur_lemmas,
                    "\n"+"-"*25+"\n"
                ]
                with open(comparison_file, 'w') as file:
                    file.write("\n".join(file_output))

                print(f"Comparison Analysis Saved to: {comparison_file}") 
                    
            elif choice in ['6', 'Exit', 'exit']:
                print("exiting...")
                return
            else:
                print("Invalid input, please enter a number from the list, (enter '6' or 'Exit' to exit): ")
                continue
################################MAIN FUNCTION CALL (DUNDER MAIN)###############################################
# triggers the whole program:       
if __name__ == "__main__":
    llmapp = AppManager(); 









    



        

  
