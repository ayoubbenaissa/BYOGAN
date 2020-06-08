FROM python:3.7.1

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

COPY . /app

#set working directory:
WORKDIR /app

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["flask", "run"]
#CMD ["flask", "run", "--host", "172.18.0.3"]