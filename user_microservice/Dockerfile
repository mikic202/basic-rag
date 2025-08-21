FROM python:3.13.1-bookworm
COPY requirements.txt /
CMD pip3 install -t /requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 5002
CMD [ "python3", "main.py" ]
