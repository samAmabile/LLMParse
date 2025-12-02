
import os
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

#pats to save smaller corpora:
TESTS_FOLDER = "tests"

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
        sentences_raw = rawtext.split('. ')
        #clean all the '*' noise:
        clean = [s.replace('*', '') for s in sentences_raw]
        sentences = [s for s in clean if s != '']
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
    
    #check for folder, make if needed:
    def verify_path(self, folder):
        os.makedirs(folder, exist_ok=True)

    #this function is to retroactively parse files with both prompt and response into 
    #   two separate files, also removes [USER] and [MODEL] tags from those files:
    def parse_chat(self, infilename, promptinfile, modelinfile):
        
        
        prompts = []
        llm_responses = []
        prompt_tag = "[USER]:"
        llm_tags = ["[MODEL]:", "[ASSISTANT]:"]

        is_llm = False

        #verify tests/ path for saving:
        self.verify_path(TESTS_FOLDER)

        
        with open(infilename, 'r', encoding='utf-8') as infile:
            for line in infile:
                text = line.strip()
                if not text or text == '---':
                    continue
                if text.upper().startswith(tuple(llm_tags)):
                    is_llm = True
                if text.upper().startswith(prompt_tag):
                    is_llm = False
                if is_llm:
                    if text.upper().startswith(tuple(llm_tags)):
                        content = text.split(':', 1)
                        response = content[1].strip()
                    else:
                        response = text.strip()
                    llm_responses.append(response)

                if not is_llm:
                    
                    if text.upper().startswith(prompt_tag):
                        content = text.split(':', 1)
                        prompt = content[1].strip()
                    else:
                        prompt = text.strip()
                    prompts.append(prompt)
        
        
        #then save each to its own file
        promptname = promptinfile+".txt" if not promptinfile.endswith(".txt") else promptinfile
        modelname = modelinfile+".txt" if not modelinfile.endswith(".txt") else modelinfile

        #everything gets the path for tests:
        promptfile = os.path.join(TESTS_FOLDER, promptname)
        modelfile = os.path.join(TESTS_FOLDER, modelname)

        with open(promptfile, 'w', encoding='utf-8') as f1, open("master_prompts.txt", 'a', encoding='utf-8') as promptlog:
            f1.write("\n".join(prompts))
            promptlog.write("\n".join(prompts))
        with open(modelfile, 'w') as f2, open("master_corpus.txt", 'a') as master:
            f2.write("\n".join(llm_responses))
            master.write("\n".join(llm_responses))
        
        #returns file paths now:
        return promptfile, modelfile
        
        


    #primary function of class, uses all other functions and returns a tuple of all the parsed types in corpus:
    def encorporate(self ,rawtext):
        sentences = self.parse_sentence(rawtext)
        raw_tokens = self.parse_tokens(rawtext)
        #to remove '*', which LLM's seem to love:
        tokens = []
        for t in raw_tokens:
            #remove standalone '*'
            if t == '*':
                continue
            
            #remove '*' attached to words:
            clean = t.strip('*')

            #if anything left after cleaning, append:
            if clean:
                tokens.append(clean)

        lower_tokens = [w.lower() for w in tokens]
        lemmatizer = WordNetLemmatizer()
        tagged = nltk.pos_tag(lower_tokens)
        lemmas = [lemmatizer.lemmatize(word.strip(), self.get_pos(tag)) for word, tag in tagged]
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