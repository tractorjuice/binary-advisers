# Have an AI Chat with Binary Advisers
Chat with Videos on AI

Chat with Binary Advisers is an AI-based application that allows you to interact with the content of videos about AI using AI.\
It utilizes the Streamlit framework to present an interactive user interface, making it easy for users to ask questions and get answers.
\
\
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://binaryadvisers.streamlit.app/)

## Features
- Querying the videos using natural language and AI.
- Incorporates OpenAI's GPT-4 model for generating answers.
- Uses Streamlit for a user-friendly UI.

## How to Run
1. Clone the repository.\
2. Set the OpenAI API key in the Streamlit secrets manager.\
3. Ensure you have the index.faiss and index.pkl files in the data_store directory.\
4. Run the streamlit app using the command streamlit run main.py.\

## Dependencies
To run this code, you need the following Python packages:

- openai
- streamlit

### API Keys
The application uses the OpenAI API. You will need to obtain an API key from OpenAI and set it in the Streamlit secrets manager.

### Data Store
The application uses a local datastore located in the data_store directory. It needs the index.faiss and index.pkl files to be present in this directory. If they are missing, you will get an error message.

## Using the Application
Once the application is running, you can use the input box labeled "Question for the book?" to ask your question. After entering your question, the application will generate an answer and display it on the screen.

## Developer Info
This application is developed by Mark Craddock. You can follow him on Twitter at https://twitter.com/mcraddock.

## Version Info
The current version of this application is 0.1.4.

## Disclaimer
This application is using the OpenAI beta service, so may experience issues.
Please use responsibly and in accordance with OpenAI's use-case policy.

## License
This project is licensed under MIT.
