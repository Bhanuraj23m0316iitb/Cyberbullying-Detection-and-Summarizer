FROM python:3.10-slim

WORKDIR /app
COPY . /app.py , ./definations.py , ./twitter_data

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install spaCy language model
RUN python -m spacy download en_core_web_sm

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]

