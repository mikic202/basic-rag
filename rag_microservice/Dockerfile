FROM python:3.13.1-bookworm
COPY requirements.txt /
RUN pip3 install -r /requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 5001
CMD [ "python3", "main.py" ]