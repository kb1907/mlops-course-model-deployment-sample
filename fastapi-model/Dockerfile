FROM python:3.9-slim

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY . app.py /app/

# Install packages from requirements.txt

RUN pip install --upgrade pip &&\
    pip install -r requirements.txt
CMD python app.py
