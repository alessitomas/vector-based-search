from fastapi import FastAPI, Query, Response
import os
import uvicorn
import json as json_lib
import pandas as pd
import torch
from .finder import FinderModel  


app = FastAPI()
app.predictor = FinderModel()  

@app.get("/hello")
def read_hello():
    return {"message": "hello world"}

@app.get("/query")
def query(query: str = Query(..., description="Input text for prediction")):
    if app.predictor is None:
        return Response(json_lib.dumps({"error": "Model not loaded"}), media_type="application/json")

    results = app.predictor.retrieve_publications(query)  
    pretty_data = json_lib.dumps({'results': results, 'message': 'OK'}, indent=4)
    return Response(pretty_data, media_type="application/json")


def run():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run()
