from flask import Blueprint

api_bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")

from . import routes
