import uvicorn
from app import app

if __name__ == "__main__":
    uvicorn.run(app, port=8000, ssl_keyfile="./private.key", ssl_certfile="./certificate.crt", ssl_ca_certs="./ca_bundle.crt")
