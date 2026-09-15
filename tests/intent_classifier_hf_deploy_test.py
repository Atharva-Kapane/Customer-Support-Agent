from gradio_client import Client

client = Client("Atharva-Kapane/Intent_Classifier")
result = client.predict(
	text="Hello!!",
	api_name="/predict_intent",
)
print(result)


# Result Should be 'label' : 'other'