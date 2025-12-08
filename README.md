
# LLMParse
LLM Language Analysis and Corpus Creation Tool

## Overview
**LLM Parse** is a language analysis and corpus creation tool for LLM language analysis. It captures and parses the language of chats with LLM models for linguistic analysis of words, tokens, lemmas, word frequency, and sentiment using the NLTK library.

## File Breakdown 
Some files are too large to view on Github. Refer to 
**Files:**
* **main.py** main file, calls to appmanager and initiates program
* **encorporator.py** holds just the Encorporator class now
* **llm_access.py** holds access to Gemini via GeminiChat and GeminiChatCloud. Has all the additional ML functions
* **llm_access_surplus.py** holds access to DeepSeek and GPT
* **appmanager.py** holds the AppManager class with all user interface, main menus
* **requirements.py** has all requirements/dependencies, updated to include scikit-learn, panda
* **/sample files and /tests** have sample files generated from the program
* **/master_annotated_corpus_files** has annotated corpus containing a file of each for POS tagged tokens, lemmas, and sentences, plus a .csv of the three. 
* **master_corpus.txt** now has only LLM generated content, and is updated with every chat
* **master_prompts.txt** holds a log of all prompts. 
* **master_chat_log.txt** what was previously master_corpus.txt, now has the entire running log of chats, with both prompt and response and the identifier tags "USER" and "MODEL"/"ASSISTANT"
## Features
* **Terminal-Based Chat:** chat with LLM models from command line
* **Generate Large Data Sets:** generate sets of questions for automated prompt-response to generate larger sets of data quickly
* **NLTK Tools:** analyze your chat based using NLTK functions: FreqDist, Lemmatize, Sentiment Analysis, comparison with the Brown corpus 
* **Detect LLM Content** train model to discern Human vs LLM generated content
* **Master LLM Corpus:** generates a master corpus of LLM language for broader study


## To Install

To install, clone the repo "https://github.com/samAmabile/LLMParse" 
run with command: py main.py (details below)

### Requirements

* **Python:** version 3.10 or higher
* **API Key:** your api key to access each of the LLM models
* **Plus:** NLTK, google.genai, openai, scikit-learn, panda. See requirements.txt for all imports

### Setup: 
1. **Clone Repository:**
    ```bash
    git clone https://github.com/samAmabile/LLMParse
    cd LLMParse
    ```

2. **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3. **To Run Program:**
    ```bash
    py main.py
    ```

## Note about API Key

Currently to use the program requires inputting API Keys for each model.
The cheapest option I have found is Gemini. API keys can be generated through Google AI Studio with your Google account. For students, there is a free year of Gemini Pro available right now. 




    


