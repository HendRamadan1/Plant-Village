from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import uvicorn
from PIL import Image
import csv
import os 
from fastapi.responses import FileResponse,StreamingResponse
from src.backend.service import predictor 
from src.backend.gemini import gemini_service
from src.backend.generete_dpf import pdf_service
import shutil
import markdown

current_dir = os.path.dirname(os.path.abspath(__file__))
templates_path = os.path.join(current_dir, "templates")

templates = Jinja2Templates(directory=templates_path)
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/index")
async def ai_engine(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit")
@app.post("/submit")
async def submit(request: Request, image: UploadFile = File(...)):
    upload_dir = os.path.join("static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, image.filename)

    img_bytes = await image.read()

    with open(file_path, "wb") as buffer:
        buffer.write(img_bytes)

    try:
        # فتح الصورة فقط للعرض و PDF
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ✅ الموديل يحتاج bytes
        res = predictor.predict(img_bytes)

        # ✅ إصلاح الخطأ الحقيقي
        raw_conf = res['confidence']
        confidence = float(raw_conf.replace('%', '').strip())

        # تقرير Gemini
        report_text_md = gemini_service.get_plant_report(res['class_name'], confidence)
        report_text_html = markdown.markdown(report_text_md)

        # اسم آمن للملف
        pdf_dir = os.path.join("static", "pdfs")
        os.makedirs(pdf_dir, exist_ok=True)
        safe_name = res['class_name'].replace(" ", "_").replace("/", "_")
        pdf_file_path = os.path.join(pdf_dir, f"{safe_name}.pdf")

        pdf_service.generate(
            disease_name=res['class_name'],
            confidence=confidence,
            report_text=report_text_html,
            image_pil=img_pil,
            file_path=pdf_file_path
        )

    except Exception as e:
        print("🔥 FULL ERROR:", e)
        return {"error": str(e)}

    image_url_for_html = f"/static/uploads/{image.filename}"

    return templates.TemplateResponse("submit.html", {
        "request": request,
        "title": res['class_name'],
        "desc": report_text_html,
        "image_url": image_url_for_html,
        "pred": confidence,
        "pdf_files": [os.path.basename(pdf_file_path)] 
    })


PDF_FOLDER = "static/pdfs"

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("static", "pdfs", filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(path=file_path, filename=filename, media_type="application/pdf")

@app.get("/market")
async def market(request: Request):
    products = []
    csv_file = r"D:\course\Courses\Plant Village\src\supplement_info (1).csv"
    
    with open(csv_file, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=',')  # صححت الـ delimiter
        
        # تنظيف أسماء الأعمدة
        reader.fieldnames = [name.strip().replace(" ", "_") for name in reader.fieldnames]
        
        for row in reader:
            row = {k.strip().replace(" ", "_"): v for k,v in row.items()}
            
            if row.get("supplement_image", "").strip():
                products.append({
                    "supplement_name": row.get("supplement_name", ""),
                    "disease_name": row.get("disease_name", ""),
                    "image": row.get("supplement_image", ""),
                    "link": row.get("buy_link", "#")
                })
    
    return templates.TemplateResponse("market.html", {"request": request, "products": products})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000,reload=True)