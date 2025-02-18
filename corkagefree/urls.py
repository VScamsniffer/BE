from django.contrib import admin
from django.urls import path, include
from users.views import DataListView, AddDataView
from django.urls import path
from django.conf.urls import include
from rest_framework_simplejwt.views import TokenObtainPairView
from django.views.generic import TemplateView  # React 파일을 서빙하기 위해 사용
from dj_rest_auth.registration.views import SocialLoginView


from django.conf import settings
from django.conf.urls.static import static

from attach.views import AudioFileUploadView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path('api/user/', get_user_info, name='get_user_info'),
    path('api/logout/', user_logout, name='user_logout'),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    # path('accounts/google/login/callback/', google_login_callback, name='google_callback'),  # ✅ 추가된 경로
    path('accounts/naver/login/callback/', naver_login_callback, name='naver_callback'),  # ✅ 네이버 로그인 콜백
    path('', TemplateView.as_view(template_name="index.html"), name='home'),
    path("api/data-list/", DataListView.as_view(), name="data_list"),
    path("api/add-data/", AddDataView.as_view(), name="add_data"),
    path("upload-audio/", AudioFileUploadView.as_view(), name="audio-upload"),
    path("attach/", include("attach.urls")),
    path('rollplaying/', include('rp.urls')),
    path('solution/', include('solution.urls')),
    path('analyze-file/', analyze_file, name='analyze_file'),
]




#미디어 파일 저장
from django.conf.urls.static import static
from django.conf import settings

# MEDIA_URL로 들어오는 요청에 대해 MEDIA_ROOT 경로를 탐색한다.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
