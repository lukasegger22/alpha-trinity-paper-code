import google.generativeai as genai
import os

# --- MAC FIX ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

# !!! HIER DEINEN KEY EINFÜGEN !!!
GEMINI_API_KEY = "AIzaSyBDugTdEvFLt31k245kzxMnWnzNRhburM0"

def list_available_models():
    print(f"--- 🕵️‍♀️ CHECKING AVAILABLE GEMINI MODELS ---")
    
    if "DEIN_GEMINI" in GEMINI_API_KEY:
        print("❌ Bitte API Key einfügen!")
        return

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        print(f"Key configured. Asking Google...")
        
        # Wir listen alle Modelle auf
        count = 0
        for m in genai.list_models():
            # Wir interessieren uns nur für Modelle, die Text generieren können
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ FOUND: {m.name}")
                print(f"   Name for code: {m.name.replace('models/', '')}")
                print(f"   Description: {m.description[:60]}...")
                print("-" * 40)
                count += 1
                
        if count == 0:
            print("⚠️ Keine Modelle gefunden. Ist der API Key aktiv?")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    list_available_models()