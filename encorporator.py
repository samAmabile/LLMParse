
import csv
import os
import nltk
import re

import pandas as pd
import numpy as np
from nltk.corpus import wordnet
from nltk.corpus import genesis
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.text import Text
from nltk.text import ContextIndex
from nltk.tag import pos_tag
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('brown')
nltk.download('stopwords')

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
        #fixed now, was incorrectly parsing sentences before:
        sentences_raw = sent_tokenize(rawtext)
        #clean all the '*' noise:
        #clean = [s.replace('*', '') for s in sentences_raw]
        #sentences = [s for s in clean if s != '']
        return sentences_raw
    
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
        
        infile = ""

        #verify tests/ path for saving:
        self.verify_path(TESTS_FOLDER)

        encodings = [('utf-8', 'replace'), ('cp1254', 'replace'), ('latin-1', 'strict')]
        
        for encoding, error in encodings:
            try:
                with open(infilename, 'r', encoding=encoding, errors=error) as f:
                    infile = f.read()
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                print(f"Could not open file {infilename}")
                return "", ""
        
        if infile is None:
            print("All encodings failed to interpret {infilename}")
            return "", ""

        
        prompts = []
        llm_responses = []
        prompt_tag = "[USER]:"
        llm_tags = ["[MODEL]:", "[ASSISTANT]:"]

        is_llm = False
        for line in infile.splitlines():
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
        with open(modelfile, 'w', encoding='utf-8') as f2, open("master_corpus.txt", 'a', encoding='utf-8') as master:
            f2.write("\n".join(llm_responses))
            master.write("\n".join(llm_responses))
        
        #returns file paths now:
        return promptfile, modelfile
        
    #function to parse pos, lemma, and sentence from an annotated corpus and put it in a folder with each element in its own txt file,
    # plus a csv of the metadata that has [token_index, sentence_index, sentence, pos_token, lemma] 
    #note: tried very hard to get the csv to line up perfectly all the way through master_annotated_corpus, but still gets off-by-one at some point
    def parse_annotated(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError as e:
            print(f"Could not load file {filename}, error: {e}")
        
        sections = content.split("-------------------------------------------------------------------------------------------")
        title = sections[0]
        pos_tagged = sections[1].split('POS Tagged Tokens:')[1].strip()
        lemmas = sections[2].split('Lemmatized Tokens:')[1].strip()
        sentences_lines = sections[3].split('Indexed Sentences:')[1].strip().split('\n')

        sentences = []
        index = 0
        for line in sentences_lines:
            if line.strip() and ':' in line and line.strip().split(':')[0].isdigit() and int(line.strip().split(':')[0])==index:
                sentence = line.split(':', 1)[1].strip()
                index += 1

            else:
                sentence = line.strip()
            
            if sentence:
                sentences.append(sentence)

        sentences_out = '\n'.join(sentences)

        outfilename = os.path.basename(filename)

        pos_file = f"pos_{outfilename}"
        lemma_file = f"lemmas_{outfilename}"
        sentence_file = f"sentences_{outfilename}"

        pathname = outfilename.replace(".txt", "")
        new_path = f"{pathname}_files"
        master = "master_annotated_corpus_files"

        os.makedirs(new_path, exist_ok=True)

        pos_path = os.path.join(new_path, pos_file)
        lemma_path = os.path.join(new_path, lemma_file)
        sentence_path = os.path.join(new_path, sentence_file)

        with open(pos_path, 'a', encoding='utf-8') as p:
            p.write(pos_tagged)
        with open(lemma_path, 'a', encoding='utf-8') as l:
            l.write(lemmas)
        with open(sentence_path, 'a', encoding='utf-8') as s:
            s.write(sentences_out)

        if new_path != master:
            os.makedirs(master, exist_ok=True)
            master_basename = "master_annotated_corpus.txt"
            master_pos_file = f"pos_{master_basename}"
            master_lemma_file = f"lemmas_{master_basename}"
            master_sentence_file = f"sentences_{master_basename}"
            master_pos_path = os.path.join(master, master_pos_file)
            master_lemma_path = os.path.join(master, master_lemma_file)
            master_sentence_path = os.path.join(master, master_sentence_file)

            with open(master_pos_path, 'a', encoding='utf-8') as mp:
                mp.write(pos_tagged)
            with open(master_lemma_path, 'a', encoding='utf-8') as ml:
                ml.write(lemmas)
            with open(master_sentence_path, 'a', encoding='utf-8') as ms:
                ms.write(sentences_out)

        
        pos_tokens = pos_tagged.split()
        lemma_tokens = lemmas.split()

        csv_header = [['Index', 'Sentence_index', 'Sentence', 'POS_Tagged_Tokens', 'Lemmas']]

        sentence_index = 1
        sentences_left = list(sentences)
        sentence_text = sentences_left.pop(0) if sentences_left else ""
        sentence_tokens = word_tokenize(sentence_text)
        is_new_sentence = True

        sentence_id = sentence_index
        sentence_token_index = 0
        token_stream = zip(pos_tokens, lemma_tokens)
        for i, (pos, lemma) in enumerate(token_stream):
            
            sentence_content = ""

            if not sentence_tokens:
                sentence_index += 1
                sentence_id = sentence_index

                if sentences_left:
                    sentence_text = sentences_left.pop(0)
                    sentence_tokens = word_tokenize(sentence_text)
                    
                else:
                    sentence_text = ""

                sentence_content = sentence_text
            
            if sentence_tokens:
                sentence_tokens.pop(0)
            else:
                sentence_content = ""
            csv_header.append([
                i+1,
                sentence_id,    
                sentence_content,
                pos,
                lemma
            ])
           

        csv_filename = f"{pathname}_metadata.csv"
        csv_path = os.path.join(new_path, csv_filename)
        with open(csv_path, 'w', newline='', encoding='utf-8') as c:

            writer = csv.writer(c)
            writer.writerows(csv_header)

        print(f"Saved all annotated files to {new_path}")
        print(f"Saved metadata to {csv_path}")
            
        return pos_path, lemma_path, sentence_path, csv_path

    #def parse_annotated_string(self, content):
    def fix_csv_indexing(self, filename):
        filepath = filename

        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"could not open/find file {filename}")
            return pd.DataFrame()

        tagged_tokens = df['POS_Tagged_Tokens']
        sentences = df['Sentence']
        #index = df['Index']
        sentence_start_indices = df[sentences.str.strip() != ''].index
        #sentences_list = sentences.dropna()

        tokens = tagged_tokens.str.split('[', expand=True)[0].str.lower().fillna('')

        aligned_id = []
        misaligned_id = []

        for i in sentence_start_indices:

            current_sentence = str(sentences.loc[i]).strip()
            try:
                first_token_raw = current_sentence.split(maxsplit=1)[0].lower()

                if "'" in first_token_raw:
                    first_token_raw = first_token_raw.split("'")[0]

                if first_token_raw.endswith('.'):
                    has_period = '.' in first_token_raw[:-1]

                    if has_period:
                        first_token = first_token_raw
                    else:
                        first_token = first_token_raw.rstrip('.')
                elif first_token_raw.endswith(','):
                    first_token = first_token_raw.rstrip(',')
                elif first_token_raw.endswith(':'):
                    first_token = first_token_raw.rstrip(':')
                else:
                    first_token = first_token_raw
                
            except IndexError:
                continue

            actual_token = tokens.loc[i]

            if first_token != actual_token:
                misaligned_id.append(i)
            else:
                aligned_id.append(i)

        corrected_column_name = 'corrected_sentece_TEMP'
        df[corrected_column_name] = df['Sentence'].copy()

        min_idx = df.index.min()
        max_idx = df.index.max()

        for j in misaligned_id:

            cur_sentence = str(df.loc[j, 'Sentence']).strip()

            try:
                first_token_raw = cur_sentence.split(maxsplit=1)[0].lower()

                if "'" in first_token_raw:
                    first_token_raw = first_token_raw.split("'")[0]

                if first_token_raw.endswith('.'):
                    has_period = '.' in first_token_raw[:-1]

                    if has_period:
                        first_token = first_token_raw
                    else:
                        first_token = first_token_raw.rstrip('.')
                elif first_token_raw.endswith(','):
                    first_token = first_token_raw.rstrip(',')
                elif first_token_raw.endswith(':'):
                    first_token = first_token_raw.rstrip(':')
                else:
                    first_token = first_token_raw
                
            except IndexError:
                continue

            if j > min_idx:
                previous_token = tokens.loc[j-1]
                if first_token == previous_token:
                    df.loc[j-1, corrected_column_name] = cur_sentence
                    df.loc[j, corrected_column_name] = ''
                    continue
            
            if j < max_idx:
                next_token = tokens.loc[j+1]
                if first_token == next_token:
                    df.loc[j+1, corrected_column_name] = cur_sentence
                    df.loc[j, corrected_column_name] = ''
                    continue
            
        df = df.drop(columns=['Sentence'])
        df = df.rename(columns={corrected_column_name: 'Sentence'})

        columns_order = ['Index', 'Sentence_index', 'Sentence', 'POS_Tagged_Tokens', 'Lemmas']

        df = df[columns_order]

        outfilename = "./master_annotated_corpus_files/mamaf.csv"

        df.to_csv(outfilename, index=False)

        print(f"\nCorrection complete: {len(misaligned_id)} lines corrected. Saved as: {outfilename}")

        return df


    #primary function of class, uses all other functions and returns a tuple of all the parsed types in corpus:
    def encorporate(self ,fullrawtext):

        rawtext = fullrawtext.replace('*', '').strip()

        sentences = self.parse_sentence(rawtext)
        raw_tokens = self.parse_tokens(rawtext)
        #to remove '*', which LLM's seem to love:
        tokens = list(raw_tokens)
        #for t in raw_tokens:
            #remove standalone '*'
        #    if t == '*':
        #        continue
            
            #remove '*' attached to words:
        #    clean = t.strip('*')

            #if anything left after cleaning, append:
        #    if clean:
        #        tokens.append(clean)

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

    #search keyword, returns word before and word after for every match in the list of tokens:
    def get_keyword(self, keyword, tokens)->list:
        kwics = []
        for i, word in enumerate(tokens):
            if word == keyword:
                kwic = f"{tokens[i-1] if i > 0 else ''} [ {keyword} ] {tokens[i+1] if len(tokens)-1 > i else ''}"
                kwics.append(kwic)
        
        return kwics

    #searches for keyword by regex, gives one word before one word after:
    def get_regex(self, pattern, tokens):
        regex = pattern
        matches = []
        for i, token in enumerate(tokens):
            if re.search(regex, token):
                match = f"{tokens[i-1] if i > 0 else ''} [ {token} ] {tokens[i+1] if len(tokens)-1 > i else ''}"
                matches.append(match)

        return matches

    #searches for the lemma of the provided keyword in the lemmatized list of tokens, and returns the non-lemmatized version as result:
    def get_lemma(self, target, tokens):

        lower_tokens = [token.lower() for token in tokens]
        tagged = pos_tag(lower_tokens)
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(word.strip(), self.get_pos(tag)) for word, tag in tagged]

        target_lower = [target.lower()]
        target_tagged = pos_tag(target_lower)
        target_lemma = lemmatizer.lemmatize(target_tagged[0][0], self.get_pos(target_tagged[0][1]))

        zipped = zip(tokens, lemmas)
        matches = []

        for i, (token, lemma) in enumerate(zipped):
            if target_lemma == lemma:
                match = f"{tokens[i-1] if i>0 else ''} [ {tokens[i]} ] {tokens[i+1] if len(tokens)-1 > i else ''}"
                matches.append(match)
        
        return matches
    
    #combines all keyword search types into one function call:
    def get_kwic(self, pattern, tokens, regex=False, lemma=False):
        if regex:
            return self.get_regex(pattern, tokens)
        elif lemma:
            return self.get_lemma(pattern, tokens)
        else:
            return self.get_keyword(pattern, tokens)

    #search for phrase in list of sentences:
    def get_keyphrase(self, target, sentences):
        phrases = []
        for i, sentence in enumerate(sentences):
            if target in sentence:
                phrase = f"{i}: {sentence}"
                phrases.append(phrase)

        return phrases
    
    #uses regex to search the sentence list:
    def get_regexphrase(self, pattern, sentences):
        phrases = []
        for i, sentence in enumerate(sentences):
            if re.search(pattern, sentence):
                phrase = f"{i}: {sentence}"
                phrases.append(phrase)
        
        return phrases
    
    #lemmatizes the phrase searched for, and then matches it against lemmatized sentences in the list being searched:
    def get_lemmasphrase(self, target, sentences):
        phrases = []
        lemmatizer = WordNetLemmatizer()
        target_list = target.split(' ')
        punctuation = ".?!"
        target_cleaned = [word.strip(punctuation) for word in target_list]
        target_lower = [word.lower() for word in target_cleaned]
        target_tagged = pos_tag(target_lower)
        target_lemmas = [lemmatizer.lemmatize(word.strip(), self.get_pos(tag)) for word, tag in target_tagged]
        header = f"Lemmatization of '{target}': {' '.join(target_lemmas)}"
        phrases.append(header)
        for i, sentence in enumerate(sentences):
            sentence_list = sentence.split(' ')
            sentence_cleaned = [word.strip(punctuation) for word in sentence_list]
            sentence_lower = [word.lower() for word in sentence_cleaned]
            sentence_tagged = pos_tag(sentence_lower)
            sentence_lemmas = [lemmatizer.lemmatize(word.strip(), self.get_pos(tag)) for word, tag in sentence_tagged]
            if sentence_lemmas == target_lemmas:
                phrase = f"{i}: {sentence}"
                phrases.append(phrase)
        
        return phrases
     
    #combines all search types to one function call:
    def search_phrase(self, pattern, sentences, regex=False, lemmas=False):
        if regex:
            return self.get_regexphrase(pattern, sentences)
        elif lemmas:
            return self.get_lemmasphrase(pattern, sentences)
        else:
            return self.get_keyphrase(pattern, sentences)

    
    def get_collocations(self, word, tokens, N, minimum=0):

        finder = BigramCollocationFinder.from_words(tokens)

        finder.apply_freq_filter(minimum)

        finder.apply_word_filter(lambda w: w in stopwords.words('english'))
        finder.apply_ngram_filter(lambda word1, word2: word1 != word and word2 != word)

        collocations = finder.nbest(BigramAssocMeasures.pmi, N)

        return collocations
    
    def get_similar(self, word, tokens):

        index = ContextIndex(tokens)
        scores = index.word_similarity_dict(word)
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_10 = sorted_scores[1:6]
        similar = index.similar_words(word)

        return similar, top_10



        



                
            

            

        