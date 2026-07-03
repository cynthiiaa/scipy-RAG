"""Check environment setup."""                                                                                                                                                                  
                                                                                                                                                                                                
import os                                                                                                                                                                                       
from dotenv import load_dotenv                                                                                                                                                                  
load_dotenv()                                                                                                                                                                                   
                                                                                                                                                                                                
# Python version                                                                                                                                                                                
import sys                                                                                                                                                                                      
assert sys.version_info >= (3, 11), "Need Python 3.11+"                                                                                                                                         
print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")                                                                                                                            
                                                                                                                                                                                                
# Core packages                                                                                                                                                                                 
import openai, chromadb, bs4, gradio                                                                                                                                                            
print("✓ Packages installed")                                                                                                                                                                   
                                                                                                                                                                                                
# API key                                                                                                                                                                                       
assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in .env"                                                                                                                                
print("✓ OpenAI API key set")                                                                                                                                                                   
                                                                                                                                                                                                
# Ollama (optional)                                                                                                                                                                             
try:                                                                                                                                                                                            
    import requests                                                                                                                                                                             
    requests.get("http://localhost:11434/api/tags", timeout=2)                                                                                                                                  
    print("✓ Ollama running")                                                                                                                                                                   
except:                                                                                                                                                                                         
    print("○ Ollama not running (optional)")                                                                                                                                                    
                                                                                                                                                                                                
print("\nLGTM! 👍")