## Running this app

### run with shell script (mac/linux)
run `./start_app.sh`
the script expects an API key from openAI. Generate from here: 
https://platform.openai.com/settings/organization/api-keys

### run manually if shell script fails:

#### python 
This is written in Python 3.14.0

#### generate python virtual environment
`python3 -m venv .venv`
`source .venv/bin/activate`

#### installrequired libraries 
`pip install -r requirements.txt`

#### set local variable for open ai api key
`export OPENAI_API_KEY="\<key from open ai website\>"`

### run app 
`python3 app.py`

