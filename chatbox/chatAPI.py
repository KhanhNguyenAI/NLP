import openai
from dotenv import  load_dotenv
load_dotenv() # khoi tao model
api_key = 'AIzaSyDj-LEe_g3CrIX_AbPLY-IHVjIa0et3VxE'
class openAPI(): 
    def __init__(self,model_name='gpt-4.1-nano'):
        super().__init__()
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.messages = [{'role':'system','content':'wellcome to chatbox API'}]
        '''
        role:
        -> system : chatbox
        -> user : user
        -> assitant : history of chat with systems
        '''
        print('model: {}'.format(self.model_name))
        
    def response(self,user_prompt): 
        user_input = str(user_prompt).lower()
        if user_input in ['goodbye','bye','quit','break'] : 
            return 'see you later'
        self.messages.append({'role' : 'user' ,'content' :user_prompt})

        try : 
            '''
            create a respone from your question
            '''
            response = self.client.chat.completions.create(
                model= self.model_name,
                messages=self.messages,
                temperature=0.5
                #maxtoken.........(unnessesary)
            )          
            chatbot_response = response.choices[0].message.content
            self.messages.append({'role': 'assistant','content': chatbot_response})
            return chatbot_response
        except openai.APIConnectionError:
            print('error')
        except Exception:
            print('connect again after a few minutes')
if __name__ =='__main__':
    bot = openAPI(model_name='gemini-2.5-flash')
    while True:
        User_mess = input('you:')
        response = bot.response(user_prompt=User_mess)
        print('bot: ',response)
        if User_mess.lower() in  ['goodbye','bye','quit','break']:
            break
