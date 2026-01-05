Text Generation with GPT-2
📌 Internship Task – Prodigy Infotech (Generative AI Track)
This project demonstrates fine-tuning GPT-2, a transformer-based language model developed by OpenAI, to generate coherent and contextually relevant text based on a given prompt.
The model is trained on a custom dataset to learn the style, structure, and context of the text provided.

🎯 Objective
Fine-tune a pre-trained GPT-2 model on a custom text dataset
Generate meaningful and context-aware text from prompts
Gain hands-on experience with Hugging Face Transformers and PyTorch
Understand the practical workflow of Generative AI tasks
🧠 Concepts Covered
Generative AI & Language Models
Transformers architecture
GPT-2 pre-training and fine-tuning
Tokenization and attention masks
Text generation and sampling techniques
🛠️ Technologies & Libraries Used
Python 3.x
Hugging Face Transformers (transformers)
Datasets library (datasets)
PyTorch (torch)
GitHub for version control and project submission
📂 Project Structure
PRODIGY_GA_02/ │── custom_data.txt # Custom training dataset │── fine_tune_gpt2.py # Script for fine-tuning GPT-2 │── generate_text.py # Script to generate text from trained model │── requirements.txt # Python dependencies │── README.md # Project documentation │── gpt2-finetuned/ # Fine-tuned GPT-2 model folder

📊 Dataset
File: custom_data.txt
Contains 15+ lines of text related to Artificial Intelligence, Machine Learning, and Generative AI
Used to fine-tune GPT-2 so it can generate similar style and context-aware text
⚙️ Installation & Setup
1️⃣ Clone the repository
git clone <your-github-repo-link>
cd PRODIGY_GA_02
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python fine_tune_gpt2.py
python generate_text.py

Example Output:
Once upon a time, the world was a place of peace and harmony. But now, it is a world of violence and chaos.
The world is filled with people who are afraid of each other, who fear each others' lives, and who hate each another...



