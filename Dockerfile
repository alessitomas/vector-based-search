FROM python:3.10-slim

WORKDIR /project

COPY requirements.txt /project/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /project/

    
EXPOSE 8888

# uvicorn app.main:app --host 0.0.0.0 --port 8888
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888"]