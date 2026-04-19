import voyageai                             
from app.config import settings                                                                   

client = voyageai.Client(api_key=settings.voyage_api_key)  # type: ignore[attr-defined]                                         
                                            
texts = [
    "The server CPU usage is critically high at 95%.",
    "Service is responding slowly, latency above 5 seconds.",                                     
    "I enjoy hiking on weekends.",                                                                
]                                                                                                 
                                                                                                
result = client.embed(texts, model="voyage-3-lite", input_type="document")                        
                                            
for text, vector in zip(texts, result.embeddings):                                                
    print(f"Text: {text[:50]}")             
    print(f"Vector length: {len(vector)}")                                                        
    print(f"First 5 values: {vector[:5]}")
    print()