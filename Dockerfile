# Docker file adapted from this tutorial https://github.com/bennzhang/docker-demo-with-simple-python-app
FROM python:3.9.7

# Creating Application Source Code Directory
RUN mkdir -p /usr/src/app

# Setting Home Directory for containers
WORKDIR /usr/src/app

# Installing python dependencies
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("all-mpnet-base-v2")'
# Copying src code to Container
COPY . /usr/src/app

# Application Environment variables
#ENV APP_ENV development
ENV PORT 4000

# Exposing Ports
EXPOSE $PORT

# Setting Persistent data
VOLUME ["/app-data"]

# Running Python Application
#CMD gunicorn -b :$PORT -c gunicorn.conf.py main:app
CMD python -m gunicorn --timeout 0 -b :$PORT flask_app:app
