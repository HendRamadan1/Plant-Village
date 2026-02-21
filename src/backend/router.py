from fastapi import APIRouter,File,UploadFile,HTTPException,status
from .schema import PredictionResult
from .service import predictor
import traceback

api_router=APIRouter()

@api_router.post("/prediction",
                 response_model=PredictionResult,
                 status_code=status.HTTP_200_OK,
                 summary="Classify plant disease from image",
                 description="Upload an image of a plant leaf, and the model will identify the disease and the confidence level."
)


async def predict_plant_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail='Invalid file type,Please apload an image(PNG,JPG)')
    try:
        image=await file.read()
        result=await predictor.predict(image_byte=image)
        return result
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail=f"AN error eccurred during image processing: {str(e)}")



