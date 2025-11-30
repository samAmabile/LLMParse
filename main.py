import llm_access, llm_access_surplus, encorporator
from llm_access import GeminiChat, GeminiChatCloud
from encorporator import Encorporator
from appmanager import AppManager

import time
import sys



#stupid little loading animation: 
def fake_loading_animation(message, response, duration=2):

    start = time.time()

    animation = ["", ".", "..", "...", "....", "....."]
    
    print(message, end="", flush=True)

    while (time.time() - start) < duration:

        for dots in animation:
            sys.stdout.write(f"\r{message}{dots}     ")
            sys.stdout.flush()
            time.sleep(0.4)
        
        sys.stdout.write("\r" + " "*25 + "\r")
        sys.stdout.flush()

        print(response)

if __name__=="__main__":
    print("Welcome, **ParseLLM** is a tool for analyzing language from LLM's and building corpora of LLM language")
    print("Developed by Jeremy Miesch and Sam Whitney")
    print("\n")
    fake_loading_animation("Loading", "Complete")
    parseLLM = AppManager()
    fake_loading_animation("Closing", "Complete")

