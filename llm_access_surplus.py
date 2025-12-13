import openai
from openai import OpenAI

#features focused on Gemini, but these options will still work for live chatting:

class DeepSeekChat:
    def __init__(self, API):
        self.client = self.connect_to_deepseek(API)
        if self.client:
            self.chat_loop(self.client)
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
            prompt = input("\nYou: ")

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
        chatname = input("\nName Chat: ")
        filename = chatname+".txt" if not chatname.endswith(".txt") else chatname
        with open(filename, 'w') as file, open("master_chat_log.txt", 'a') as master:
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
        if self.client:
            self.chat_loop(self.client)
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
            prompt = input("\nYou: ")
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
            with open(filename, 'w') as file, open("master_chat_log.txt", 'a') as master:
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