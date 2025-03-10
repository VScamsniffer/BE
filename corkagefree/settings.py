from pathlib import Path
import os
import requests
import dotenv
from decouple import config
from datetime import timedelta

# Build paths inside the project
import requests
import dotenv





# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Security Settings
SECRET_KEY = config('SECRET_KEY')
DEBUG = True # 개발 환경에서 True, 배포 환경에서는 False로 설정
ALLOWED_HOSTS = ['127.0.0.1', 'localhost', '40.82.157.231', 'vscamsniffer.work.gd','4.230.156.117:443']

# Application Definition

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
dotenv.load_dotenv()

SECRET_KEY = os.environ.get("SECRET_KEY")
API_KEY=os.environ.get("SECRET_KEY")

SECRET_KEY = config('SECRET_KEY')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['127.0.0.1', 'localhost']


# Application definition

INSTALLED_APPS = [
    'daphne',
    'corsheaders',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.sites',
    'rest_framework',
    'rest_framework.authtoken',
    'rest_framework_simplejwt',
    'users',
    'rp',
    'solution',

    #allauth
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.google',
    'allauth.socialaccount.providers.kakao',
    'allauth.socialaccount.providers.naver',
    'channels',
    "attach"
]

import sys

# 🚀 collectstatic 실행 시 bitsandbytes 관련 앱 제외
if 'collectstatic' in sys.argv:
    INSTALLED_APPS = [app for app in INSTALLED_APPS if app not in ['rp']]


# REST Framework Settings
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
}

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'AUTH_HEADER_TYPES': ('Bearer',),
}

# ASGI Configuration
ASGI_APPLICATION = 'corkagefree.asgi.application'

# Middleware
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    "allauth.account.middleware.AccountMiddleware",
]

ROOT_URLCONF = 'corkagefree.urls'
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [("redis", 6379) if os.environ.get("DOCKER_ENVIRONMENT") else ("localhost", 6379)],
        },
        "OPTIONS": {
            "capacity": 1500,
            "expiry": 10,
            "group_expiry": 86400,  # 그룹 만료 시간 (초)
            "ping_interval": 20,    # 20초마다 ping 메시지
            "ping_timeout": 10,     # 10초 내에 pong 응답 필요
        },
    },
}


#ping을 보내서 타임아웃 설정
# CHANNELS_DEFAULTS = {
#     'websocket': {
#         'ping_interval': 20,  # 20초마다 ping 메시지 보내기
#         'ping_timeout': 10,  # 10초 이내로 pong 메시지를 받아야 타임아웃 처리
#     }
# }

# Templates Configuration
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'corkagefree.wsgi.application'

# Database Configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password Validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static Files
STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [os.path.join(BASE_DIR, "static")]  # STATIC_ROOT와 겹치지 않도록 수정

# Default Primary Key Field Type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CORS Settings
CORS_ORIGIN_ALLOW_ALL = False
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://vscamsniffer.work.gd",
    "https://4.230.156.117:443",
]
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://vscamsniffer.work.gd",
    "https://4.230.156.117:443",
]
CORS_ALLOW_ALL_ORIGINS = True 

CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_HEADERS = [
    "Authorization",
    "Content-Type",
    "X-CSRFToken",
    "Cache-Control",
    "Pragma",
    "Accept",
]
CORS_ALLOW_METHODS = [
    "GET",
    "POST",
    "PUT",
    "DELETE",
    "OPTIONS",
    "PATCH",
]
CORS_PREFLIGHT_MAX_AGE = 0  # Disable Preflight Request Caching

# Authentication and Social Login Settings
AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
)
SITE_ID = 5
# LOGIN_REDIRECT_URL = "http://localhost:3000/"
LOGIN_REDIRECT_URL = "https://vscamsniffer.work.gd/"

SOCIALACCOUNT_ADAPTER = "users.adapters.MySocialAccountAdapter"
ACCOUNT_LOGIN_ON_GET = True
SOCIALACCOUNT_LOGIN_ON_GET = True

SOCIALACCOUNT_PROVIDERS = {
    'google': {
        'SCOPE': ['profile', 'email'],
        'AUTH_PARAMS': {'access_type': 'online'},
    },
    'kakao': {'SCOPE': ['profile_nickname']},
    'naver': {'SCOPE': ['profile_nickname']},
}

# Azure Blob Storage Settings
DEFAULT_FILE_STORAGE = "storages.backends.azure_storage.AzureBlobStorage"

AZURE_ACCOUNT_NAME = config('AZURE_ACCOUNT_NAME')
AZURE_ACCOUNT_KEY = config('AZURE_ACCOUNT_KEY')
AZURE_CONTAINER = config('AZURE_CONTAINER')
MEDIA_URL = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_CONTAINER}/"
AZURE_CUSTOM_DOMAIN = f"{AZURE_ACCOUNT_NAME}.blob.core.windows.net"


#디버깅을 위한 로깅 설정(daphne)
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'level': 'DEBUG',  # Daphne 관련 DEBUG 로그를 콘솔로 출력
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'ERROR',
            'class': 'logging.FileHandler',
            'filename': 'daphne_error.log',
        },
    },
    'loggers': {
        'daphne': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',  # DEBUG 수준으로 Daphne 로그 출력
            'propagate': False,
        },
    },
}

