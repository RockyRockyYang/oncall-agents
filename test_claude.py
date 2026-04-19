from anthropic import Anthropic
from app.config import settings                                                                   
                
client = Anthropic(api_key=settings.anthropic_api_key)                                            

response = client.messages.create(                                                                
    model=settings.rag_model,
    max_tokens=100,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)                                                                                                 

print(response.content[0].text)