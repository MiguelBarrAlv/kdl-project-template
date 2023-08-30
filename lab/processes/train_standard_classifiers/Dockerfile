FROM python:3.9-slim

WORKDIR /app

COPY Pipfile Pipfile.lock /app/

RUN pip install pipenv && pipenv install --deploy --ignore-pipfile

COPY . /app/

EXPOSE 5000

CMD ["pipenv", "run", "python", "main.py"]
