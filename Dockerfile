FROM python:3.10-slim

# Install dependencies



COPY app /app

ENV VIRTUAL_ENV=/usr/local
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install uv

COPY requirements.txt /
COPY run.py /
COPY gunicorn.sh /

RUN uv pip install -r requirements.txt

EXPOSE 80

RUN ["chmod", "+x", "./gunicorn.sh"]

ENTRYPOINT ["./gunicorn.sh"]
