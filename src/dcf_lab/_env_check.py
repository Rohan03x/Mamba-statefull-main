import os

from dotenv import load_dotenv

load_dotenv()
print('PROVIDER=', os.getenv('PROVIDER'))
print('CIQ_BASE_URL=', os.getenv('CIQ_BASE_URL'))
print('Has ACCESS_TOKEN=', bool(os.getenv('CIQ_ACCESS_TOKEN')))
print('Has USERNAME=', bool(os.getenv('CIQ_USERNAME')))
print('Has PASSWORD=', bool(os.getenv('CIQ_PASSWORD')))
print('Has TOKEN_URL=', bool(os.getenv('CIQ_TOKEN_URL')))
