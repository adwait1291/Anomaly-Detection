import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
import serve_model


app_desc = """<h2>Anomaly Detector `detect/image`</h2>"""
app = FastAPI(title='Anomaly Detector', description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
