from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import jwt
from jwt import PyJWTError
from starlette.types import ASGIApp, Receive, Scope, Send
from cryptography.hazmat.primitives import serialization

from app.core.config import settings

app = FastAPI()

class ValidateTokenMiddleware:
    def __init__(self, app: ASGIApp, public_key: str, expected_audience: list):
        self.app = app
        self.public_key = self.load_public_key(public_key)
        self.expected_audience = expected_audience

    def load_public_key(self, public_key_str: str):
        try:
            return serialization.load_pem_public_key(public_key_str.encode())
        except ValueError as e:
            logging.error(f"Error loading public key: {e}")
            raise

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        if scope["path"] in ["/docs", "/openapi.json"]:
            await self.app(scope, receive, send)
            return

        if scope['method'] == 'OPTIONS':
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        token_string = request.headers.get("Authorization")

        logging.basicConfig(level=logging.INFO)

        if not token_string:
            response = JSONResponse(status_code=403, content={"message": "No token provided!"})
            await response(scope, receive, send)
            return

        args = token_string.split(" ")
        if args[0] != "Bearer":
            response = JSONResponse(status_code=401, content={"message": "Unauthorized!"})
            await response(scope, receive, send)
            return

        try:
            decoded_token = jwt.decode(args[1], self.public_key, algorithms=["RS256"], audience=self.expected_audience)
            scope['user'] = decoded_token["email"]
            scope['user_id'] = decoded_token["user_id"]

        except PyJWTError as e:
            logging.error(f"JWT decode error: {str(e)}")
            response = JSONResponse(status_code=401, content={"message": str(e)})
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)