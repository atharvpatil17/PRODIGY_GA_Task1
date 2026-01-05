from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load fine-tuned model
model_path = "./gpt2-finetuned"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Input prompt
prompt = "Artificial intelligence is"

inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    inputs["input_ids"],
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,
    temperature=0.7
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n📝 Generated Text:\n")
print(generated_text)

