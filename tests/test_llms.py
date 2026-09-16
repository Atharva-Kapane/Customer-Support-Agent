import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def safe_content(response):
    content = response.choices[0].message.content
    return (content or "").strip() or "(empty response)"


def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Gemini: API key missing")
        return

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Say 'Gemini is working' and nothing else."}],
            max_tokens=40
        )
        print("✅ Gemini          :", safe_content(response))
    except Exception as e:
        print("❌ Gemini          :", str(e)[:120])


def test_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Groq: API key missing")
        return

    models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    ]

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'Groq is working' and nothing else."}],
                max_tokens=40
            )
            print(f"✅ Groq ({model}):", safe_content(response))
            return
        except Exception as e:
            print(f"   Groq failed {model}: {str(e)[:90]}")
    print("❌ Groq: all models failed")


def test_openrouter():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OpenRouter: API key missing")
        return

    models = [
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3.5-lightning:free",
        "openrouter/free",
        "google/gemma-4-26b-a4b-it:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ]

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/Atharva-Kapane/Customer-Support-Agent",
            "X-Title": "Customer-Support-Agent"
        }
    )

    success = False
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'OpenRouter is working' and nothing else."}],
                max_tokens=40
            )
            print(f"✅ OpenRouter ({model}):", safe_content(response))
            success = True
        except Exception as e:
            print(f"   OpenRouter failed {model}: {str(e)[:90]}")
    
    if not success:
        print("❌ OpenRouter: all models failed")


if __name__ == "__main__":
    print("Testing free LLM providers...\n")
    test_gemini()
    print()
    test_groq()
    print()
    test_openrouter()