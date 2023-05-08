
#get python
FROM python:3.8-slim-buster

# create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /opt/spamML/

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/spamML"

COPY docker-requirements.txt .
RUN pip3 install -r docker-requirements.txt

COPY . .

CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0"]