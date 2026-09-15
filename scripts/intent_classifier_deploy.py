from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Configuration - Replace with your actual Hugging Face profile name
hf_username = "Atharva-Kapane" 
repo_name = f"{hf_username}/twcs-distilbert-intent-classifier"
local_model_path = r"D:\Projects\Customer-Support-Agent\models\intent_classifier_distilbert"

print(f"Deploying local weights to: https://huggingface.co{repo_name}")

# 2. Load the exact frozen offline assets from your disk
model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 3. Upload to the cloud simultaneously (creates the repository automatically)
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print("\n--- PROD DEPLOYMENT COMPLETE ---")
print(f"Hosted URL: https://huggingface.co{repo_name}")
