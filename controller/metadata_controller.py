import os

from fastapi import APIRouter
from starlette.responses import JSONResponse

from util.db.faiss_handler import FaissHandler

# Paths to FAISS index and metadata
FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH')
METADATA_PATH = os.getenv('FAISS_METADATA_PATH')

# Create a router instance
router = APIRouter(
    prefix="/metadata",
    tags=["METADATA"],
    responses={404: {"description": "Not found"}},
)


@router.get("/first-value", summary="Get the first value", description="Retrieve the first embedding and metadata.")
def get_first_value():
    """
    API endpoint to retrieve the first embedding and metadata.
    """
    result = FaissHandler.get_first_value(FAISS_INDEX_PATH, METADATA_PATH)
    return {"response": result}
