# Medical-Chatbot-using-Llama2
Medical-Chatbot-using-Llama2/
├── .venv/                        # Virtual environment folder
├── data/                         # Data folder
│   └── Medical_book.pdf          # PDF file (15,750 KB)
├── Medical_Chatbot.egg-info/     # Package info folder
├── model/                        # Model-related files
│   ├── instruction.txt           # Instruction file (1 KB)
│   └── llama-2-7b-chat.ggmlv3.q4_0.bin  # Model file (37,028,857 KB)
├── research/                     # Research-related files
│   ├── .ipynb_checkpoints/       # Checkpoints for Jupyter notebooks
│   └── trails.ipynb              # Jupyter notebook file (27 KB)
├── src/                          # Source code folder
│   ├── __pycache__/              # Compiled Python files
│   ├── __init__.py               # Package initialization file (0 KB)
│   ├── helper.py                 # Helper functions (2 KB)
│   └── prompt.py                 # Prompt logic (1 KB)
├── static/                       # Static assets like images or CSS
│   └── style.css                 # CSS file (5 KB)
├── templates/                    # HTML templates for the project
│   ├── chat.html                 # Chat interface template (4 KB)
│   └── pure.html                 # Basic HTML template (4 KB)
├── .env                          # Environment variables file
├── .gitignore                    # Git ignore configuration
├── app.py                        # Main application file
├── LICENSE                       # Licensing information
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── setup.py                      # Setup script
├── store_index.py                # Indexing logic
└── template.py                   # Template script

# How to run?
### STEPS:

```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone