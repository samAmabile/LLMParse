from llm_access import GeminiChat, GeminiChatCloud
from llm_access_surplus import GPTChat, DeepSeekChat
from encorporator import Encorporator

import glob
import os
import nltk
import random
import pandas as pd
from nltk.corpus import brown

nltk.download('brown')

#folders for paths:
TESTS_FOLDER = "tests"
ANALYSIS_FOLDER = "analysis"

class AppManager:
    def __init__(self):
        self.main_loop()
    
    #this menu selects the LLM model, or loading already saved files. Returns the filename of the chat or saved file:
    def MainMenu(self):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~LLM LANGUAGE ANALYZER~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Select an LLM model to begin a chat, then after completion the prompts/responses will save to separate files")
        print("The LLM content from your chat will automatically load to the program for analysis.")
        print("To analyze a previously saved chat, select: '5. Load file to analyze'")
        print("To run analysis on all captured LLM language to-date, load the Master Corpus")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Main Menu: ")
        print("1. Chat with DeepSeek LLM")
        print("2. Chat with GPT 3.5 Turbo")
        print("3. Chat with Gemini 2.5 Flash **use this one**")
        print("4. Chat with Gemini via Cloud (no API)")
        print("5. Load file to analyze") #adapt to have .csv or .txt
        print("6. Load Master Corpus")
        print("7. Parse Chat Logs into separate prompt/response files")
        choice = input("Enter selection: ")

        #maybe return choice here and have functions for each option? 

        if choice == '1':
            API = input("Enter API Key to connect with DeepSeek: ")
            dschat = DeepSeekChat(API)
            dsclient = dschat.connect_to_deepseek(API)
            filename = dschat.chat_loop(dsclient)
            return filename
        elif choice == '2':
            API = input("Enter API Key to connect with GPT: ")
            gptchat = GPTChat(API)
            gptclient = gptchat.connect_to_gpt(API)
            filename = gptchat.chat_loop(gptclient)
            return filename
        elif choice == '3':
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
        
        elif choice == '4':
            uri = "https://us-central1-parsellm.cloudfunctions.net/gemini-proxy"
            geminichatcloud = GeminiChatCloud(uri)
            filename = geminichatcloud.chat_loop()
            return filename
        
        #updated to search for both txt and csv: 
        elif choice == '5':
            folder = input("\nenter subfolder to search, or '.' for main: ")
            filename = self.load_file(["txt", "csv"], folder)
            return filename
        
        elif choice == '6':
            master_csv = "master_corpus.csv"
            master_txt = "master_corpus.txt"
            return master_csv, master_txt
        
        #this only calls up txt files, as it was meant to parse old whole chat files:
        elif choice == '7':
            parser = Encorporator()
            choose = input(
                "Enter [1] to enter filename, " \
                "\nEnter [2] to choose from list of .txt files in directory" \
                "\nEnter [3] to search in a subfolder")
            while True:
                if choose == '1':
                    infile = input("Enter filename: ")
                    break
                elif choose == '2':
                    infile = self.load_file("txt")
                    break
                elif choose == '3':
                    folder = input("\nEnter subfolder name: ")
                    infile = self.load_file("txt", folder)
                    break
                else:
                    print("invalid input, please enter '1' or '2': ")
            
            #alternatively could hardcode prefix to the chosen filename for each of these
            # i.e. "prompts"+infile or "llm_responses"+infile:
            promptfile = input("Enter Filename for prompts: ")
            modelfile = input("Enter Filename for model responses: ")

            #parse_chat returns both filenames:
            prompts, llm_responses = parser.parse_chat(infile, promptfile, modelfile)

            print(f"prompts saved to: {promptfile}")
            print(f"LLM responses saved to: {modelfile}, and appended to Master Corpus")

            #returns just the llm language file for analysis
            #no csv because this is parsing from old chats, we could update to accomodate though:
            return llm_responses

        else:
            print("invalid input, please enter a number from the list of options ")
              
    #function for loading files:
    def load_file(self, extensions, directory='.'):
        #using regex style pattern matching to search the current directory for any .txt files
            ##**i want to add the ability to search in other places, load files not in current directory
        if isinstance(extensions, str):
            extensions = [extensions]
        
        savedfiles = []
        for suffix in extensions:
            pattern = os.path.join(directory, f"*.{suffix}")
            savedfiles.extend(glob.glob(pattern))

        if not savedfiles:
            print(f"no files with extension '.txt' or '.csv' found in directory: [{directory}]")
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
    
    #checks for path, makes folder if path nonexistent:
    def verify_path(self, folder):
        os.makedirs(folder, exist_ok=True)

    def main_loop(self):
        filenames = self.MainMenu()
        if not filenames:
            print("No filename returned from main/no file created from chat")
            return
        #if the type of chat returns both csv and txt filenames as tuple:
        if isinstance(filenames, tuple):
            choice = input(
                "Enter [1] to analyze .csv containing [prompt,chat,live?,tag?]" \
              "\nEnter [2] to analyze .txt containing LLM responses only")
            filename = filenames[0] if choice=='1' else filenames[1]
        #otherwise it returns just a string (like in case of load saved file)
        else:
            filename = filenames
        #instance of Encorporator object to access those functions:
        encorporator_a = Encorporator()
        #whatever chat_loop returns and user selects, rawtext will be extracted from appropriate filetype:  
        # rawtext extract from .csv:
        if filename.endswith('.csv'):
            #give the user more variability, they could experiment with many different things
            df = pd.read_csv(filename)

            print("What would you like to look at?")
            while(True):
                print("1. LLM Rawtext")
                print("2. Prompts Rawtext")
                if filename == 'master_corpus.csv':
                    print("3. Live Chat Rawtext")
                    print("4. Generated Chat Rawtext")
                    print("5. Specific Tag")
                    print("6. Specific Chat #")
                selection = input("Select an option...")
            
                if selection == '1':
                    rawtext = '. '.join(list(df[df["Role"]=="Model"]["Content"]))
                    break
                elif selection == '2':
                    rawtext = '. '.join(list(df[df["Role"]=="User"]["Content"]))
                    break 
                elif selection == '3':
                    rawtext = '. '.join(list(df[df["Live?"]==1]["Content"]))
                    break
                elif selection == '4':
                    rawtext = '. '.join(list(df[df["Live?"]==0]["Content"]))
                    break
                elif selection == '5':
                    tag_choice = input("Which tag?")
                    rawtext = '. '.join(list(df[df["Tag"]==tag_choice]["Content"]))
                    break
                elif selection == '6':
                    chat_choice = input("Which chat?")
                    rawtext = '. '.join(list(df[df["Chat"]==int(chat_choice)]["Content"]))
                    break
                else:
                    print("invalid input, please enter a number from the list: ")
        else:
        #rawtext from .txt:
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

        #here is the main analysis 'function' of the program:
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
                action = input("\nenter filename to save to .txt file, enter 'back' to return to corpus analysis menu\n").strip()

                if action.lower() == 'back':
                    continue
                
                self.verify_path(ANALYSIS_FOLDER)

                filename = action+'.txt' if not action.endswith('.txt') else action
                path = os.path.join(ANALYSIS_FOLDER, filename)

                with open(path, 'w', encoding='utf-8') as file:
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
                
                self.verify_path(ANALYSIS_FOLDER)

                filename = action+".txt" if not action.endswith(".txt") else action
                path = os.path.join(ANALYSIS_FOLDER, filename)

                with open(path, 'w', encoding='utf-8') as file:
                    for index, lemma in enumerate(lemmas):
                        if index % 10 == 0:
                            file.write("\n")
                        file.write(lemma)
            
            elif choice == '3':
                print("Sentiment Analysis: ")
                scores = encorporator_a.analyze_sentiment(rawtext)
                for category, score in scores.items():
                    print(f"{category}: {score}")

                action = input("enter filename to save to .txt file, enter 'back' to return to corpus analysis menu").strip()

                if action.lower() == 'back':
                    continue
                
                self.verify_path(ANALYSIS_FOLDER)

                filename = action+".txt" if not action.endswith(".txt") else action
                path = os.path.join(ANALYSIS_FOLDER, filename)

                with open(path, 'w', encoding='utf-8') as file:
                    for category, score in scores.items():
                        file.write(f"\n{category}: {score}")

            elif choice == '4':
                #lemmas:
                lemma_list = []
                #here it prints to the terminal, formatting is correct:
                print("Lemmas: ")
                for index, lemma in enumerate(lemmas):
                    if (index % 10 == 0):
                        print("\n")
                    print(lemma, end=' ')
                    lemma_list.append(lemma)
                #assigning the list of lemmas to a list of lists of 10 lemmas each:
                lemmas_grouped = [lemma_list[i:i + 10] for i in range(0, len(lemma_list), 10)]
                #turning the list into a string:
                lemmas_lines = "\n".join(' '.join(word) for word in lemmas_grouped)
                #tagged tokens:    
                print("Tagged Tokens: ")
                num_per_line = 10
                formatted_tagged_tokens = [f"{word}[{tag.upper()}]" for word, tag in pos_tag_tokens]
                tagged_list = []
                for i in range(0, len(formatted_tagged_tokens), num_per_line):
                    line = formatted_tagged_tokens[i:i+num_per_line]
                    print('|'.join(line))
                    tagged_list.append(' '.join(line))

                print("Sentences: ")
                sentence_list = {}
                for index, s in enumerate(sentences):
                    print(f"\n{index}::{s}")
                    sentence_list[index] = s

                action = input("Enter filename to save to file, enter 'back' to return to analysis menu: ")

                if action.lower() == 'back':
                    continue

                self.verify_path(ANALYSIS_FOLDER)
                
                #set filename and subfolder for saving to:
                outfile = action+".txt" if not action.endswith(".txt") else action
                path = os.path.join(ANALYSIS_FOLDER, outfile)

                #formatting each element for file output:
                sentences_out = "\n".join(f"{index}: {sentence}" for index, sentence in sentence_list.items())
                tagged_out = "\n".join(tagged_list)
                lemmas_out = lemmas_lines
                annotated_corpus = [
                    f"Annotated {filename} Corpus: ", 
                    "-------------------------------------------------------------------------------------------",
                    "POS Tagged Tokens: ",
                    tagged_out, 
                    "-------------------------------------------------------------------------------------------",
                    "Lemmatized Tokens: ",
                    lemmas_out,
                    "-------------------------------------------------------------------------------------------",
                    "Indexed Sentences: ",
                    sentences_out   
                ]
                with open(path, 'w', encoding='utf-8') as of:
                    of.write("\n".join(annotated_corpus))

                print(f"Annotation Saved to {path}")

                
            elif choice == '5':
                print("COMPARE CORPORA")
                print("choose from the following options:")
                while True:
                    print("1. Use the entire brown corpus")
                    print("2. Choose specific subcorpora from the brown corpus")
                    corpus_choice = input("Enter selection: ").strip()
                    #brown = nltk.corpus.brown
                    if corpus_choice == '1':
                        print("Using sample of 50,000 tokens from Brown")
                        brown_words = list(brown.words())
                        sampled = random.sample(brown_words, 50000)
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
                #variables to hold the values:
                len_set_br = len(set(tokens_br))
                len_br = len(tokens_br)
                len_set_cur = len(set(tokens))
                len_cur = len(tokens)
                #variables to hold the float from division:
                ttr_br = len_set_br / len_br
                ttr_cur = len_set_cur / len_cur
                #print ttrs:
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
                plot_br = encorporator_a.pretty_print_sorted_fdist(sorted_fdist_br[:50])
                print(f"\n{plot_br}")
                print("\n--------------------------------------------------------------")
                print("Current Corpus Frequency Distribution: ")
                plot_cur = encorporator_a.pretty_print_sorted_fdist(sorted_fdist[:50])
                print(f"\n{plot_cur}")
                # do some other comparisons here
                ################compare tag frequency########################
                #making a dictionary of just tag::count :
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

                #sort descending:
                sorted_tags_br = sorted(tag_count_br.items(), key=lambda item: item[1], reverse=True)
                sorted_tags_cur = sorted(tag_count_cur.items(), key=lambda item: item[1], reverse=True)
                print("\n----------------------------------------------------------------")
                print("Brown Part of Speech Tags: ")
                for tag, count in sorted_tags_br:
                    print(f"{tag.upper()}: {count}")
                print("------------------------------------------------------------------")
                print("Current Corpus Part of Speech Tags")
                for tag, count in sorted_tags_cur:
                    print(f"{tag.upper()}: {count}")

                ############top lemmas comparison################
                #did this the least efficient way had to get fdist first:
                lemmas_dist_br = encorporator_a.get_fdist(lemmas_br)
                lemmas_dist_cur = encorporator_a.get_fdist(lemmas)

                #then sort fdists in reverse:
                sorted_lemmas_br = sorted(lemmas_dist_br.items(), key=lambda item: item[1], reverse=True)
                sorted_lemmas_cur = sorted(lemmas_dist_cur.items(), key=lambda item: item[1], reverse=True)

                #then print using the plot function for **sorted** fdists:
                print("\n---------------------------------------------------------------")
                print("Lemma Frequency Brown: ")
                lemmas_plot_br = encorporator_a.pretty_print_sorted_fdist(sorted_lemmas_br[:50])
                print(lemmas_plot_br)
                print("------------------------------------------------------------------")
                print("Lemmas Frequency Current Corpus")
                lemmas_plot_cur = encorporator_a.pretty_print_sorted_fdist(sorted_lemmas_cur[:50])
                print(lemmas_plot_cur)
                
                #for sentences, maybe we need a concordance/kwic or regex search of some kind

                action = input("Enter filename to save comparisons to file, enter 'back' to return to analysis menu: \n").strip()
                if action == 'back':
                    break
                
                #make folder if it doesnt exist:
                self.verify_path(ANALYSIS_FOLDER)

                #make filename and add path:
                comparison_file = action+".txt" if not action.endswith(".txt") else action
                path = os.path.join(ANALYSIS_FOLDER, comparison_file)

                #bunch of strings to just hold all the above stuff for file saving.
                br_plot = f"Brown FreqDist: {plot_br}"
                cur_plot = f"{filename} FreqDist: {plot_cur}"
                formatted_tags_br = "Brown POS Tags: "+"\n".join((f"{tag.upper()}: {count}" for tag, count in sorted_tags_br))
                formatted_tags_cur = f"{filename} POS Tags: "+"\n".join((f"{tag.upper()}: {count}" for tag, count in sorted_tags_cur))
                brown_ttr = f"Type to Token ratio brown: {ttr_br}"
                cur_ttr = f"Type to Token ratio {filename}: {ttr_cur}"
                brown_lemmas = f"Brown Lemmas: {lemmas_plot_br}"
                cur_lemmas = f"{filename} Lemmas: {lemmas_plot_cur}"

                #formatted output for file saving:
                file_output = [
                    f"{filename} vs Brown Analysis:",
                    "*Disclaimer: all graphs are of top 50 only*",
                    "\n"+"-"*100+"\n",
                    brown_ttr,
                    "\n"+"-"*100+"\n",
                    cur_ttr,
                    "\n"+"-"*100+"\n",
                    br_plot,
                    "\n"+"-"*100+"\n",
                    cur_plot,
                    "\n"+"-"*100+"\n",
                    formatted_tags_br,
                    "\n"+"-"*100+"\n",
                    formatted_tags_cur,
                    "\n"+"-"*100+"\n",
                    brown_lemmas,
                    "\n"+"-"*100+"\n",
                    cur_lemmas,
                    "\n"+"-"*100+"\n"
                ]
                with open(path, 'w', encoding='utf-8') as file:
                    file.write("\n".join(file_output))

                print(f"Comparison Analysis Saved to: {path}") 
                    
            elif choice in ['6', 'Exit', 'exit']:
                print("exiting...")
                return
            else:
                print("Invalid input, please enter a number from the list, (enter '6' or 'Exit' to exit): ")
                continue










    