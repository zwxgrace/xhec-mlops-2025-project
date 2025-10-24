FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

COPY bin/run_services.sh /app/bin/run_services.sh
COPY abalone.csv /app/abalone.csv

# use dos2unix to convert line endings from Windows to Unix format
RUN apt-get update && \
    apt-get install -y dos2unix && \
    dos2unix /app/bin/run_services.sh && \
    apt-get purge -y --auto-remove dos2unix && \
    rm -rf /var/lib/apt/lists/*

RUN chmod +x /app/bin/run_services.sh

EXPOSE 8001
EXPOSE 4201

CMD ["/app/bin/run_services.sh"]
