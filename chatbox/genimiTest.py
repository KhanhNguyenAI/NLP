import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # Khởi tạo model
api_key = os.getenv("GEMINI_API_KEY")  # Đặt key vào file .env với tên GEMINI_API_KEY

class GeminiAPI():
    def __init__(self, model_name='gemini-pro'):
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.messages = [{'role': 'system', 'content': 'welcome to chatbox API'}]
        print('model: {}'.format(self.model_name))

    def response(self, user_prompt):
        user_input = str(user_prompt).lower()
        if user_input in ['goodbye', 'bye', 'quit', 'break']:
            return 'see you later'
        self.messages.append({'role': 'user', 'content': user_prompt})

        try:
            # Gemini không lưu lịch sử hội thoại như OpenAI, bạn cần tự quản lý nếu muốn
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in self.messages])
            response = self.model.generate_content(prompt)
            chatbot_response = response.text
            self.messages.append({'role': 'assistant', 'content': chatbot_response})
            return chatbot_response
        except Exception as e:
            print('Error:', e)
            return "Error: Could not connect to Gemini API."

if __name__ == '__main__':
    bot = GeminiAPI()
    while True:
        User_mess = input('you:')
        response = bot.response(user_prompt=User_mess)
        print('bot: ', response)
        if User_mess.lower() in ['goodbye', 'bye', 'quit', 'break']:
            break