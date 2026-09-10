# WonderSnap

بازسازی مستقل دموی لینکدین: تصویر زندهٔ وب‌کم، تشخیص ۲۱ نقطهٔ دست با MediaPipe و بناهای سه‌بعدی ساخته‌شده از ۲۵۰٬۰۰۰ ذره روی GPU.

این نسخه مدل آمادهٔ سه‌بعدی بارگذاری نمی‌کند؛ نقاط چهار بنای **Big Ben، Statue of Liberty، Temple Gate و Colosseum** با فرمول‌های NumPy ساخته می‌شوند. ModernGL محاسبات تغییرشکل، حرکت، glow و trail را روی GPU انجام می‌دهد.

## اجرا روی Windows

Python 3.10 یا 3.11 پیشنهاد می‌شود.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python wondersnap.py
```

در اولین اجرا مدل رسمی Hand Landmarker (حدود ۷٫۵ مگابایت) به‌طور خودکار در پوشهٔ `assets` دریافت می‌شود. دفعات بعد کاملاً محلی است.

اگر PowerShell اجازهٔ Activate نداد، بدون فعال‌سازی اجرا کنید:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe wondersnap.py
```

## کنترل‌ها

- **کف دست باز:** ذرات پخش می‌شوند.
- **مشت بسته:** ذرات دوباره روی فرم بنا جمع می‌شوند.
- **بشکن / رها کردن سریع شست از انگشت وسط:** بنای بعدی با trail ظاهر می‌شود.
- `Space`: بنای بعدی (fallback بدون تشخیص gesture)
- کلیدهای `1` تا `4`: انتخاب مستقیم بنا
- `O`: پخش ذرات
- `F`: جمع کردن ذرات
- `H`: نمایش/عدم نمایش نقاط دست
- `Esc`: خروج

## تنظیم کارایی

پیش‌فرض برنامه ۲۵۰٬۰۰۰ ذره و رزولوشن 1280×720 است. روی GPU ضعیف‌تر:

```powershell
python wondersnap.py --particles 100000 --width 960 --height 540
```

## دوربین RTSP شما

آدرس `10.41.41.254`، نام کاربری `admin`، رمز و مسیر main stream دوربین به‌صورت پیش‌فرض داخل `wondersnap.py` تنظیم شده‌اند، بنابراین برنامه بدون هیچ تنظیم اضافه‌ای به دوربین وصل می‌شود:

```powershell
.\run.ps1
```

> رمز داخل سورس ذخیره شده است. اگر این پروژه را جایی منتشر می‌کنید، مقدار `DEFAULT_RTSP_PASSWORD` را خالی کنید و رمز را از متغیر محیطی بدهید:

```powershell
$env:WONDERSNAP_RTSP_PASSWORD = 'YOUR_PASSWORD'
python wondersnap.py
```

ترتیب اولویت رمز: `--rtsp-password` سپس `WONDERSNAP_RTSP_PASSWORD` سپس مقدار پیش‌فرض داخل فایل.

### اگر تصویر کند بود

main stream این دوربین 2560×1440 است. برای CPU کمتر از sub stream استفاده کنید:

```powershell
python wondersnap.py --substream
```

اگر دوربین در دسترس نباشد برنامه خودکار سراغ وب‌کم محلی می‌رود و اگر آن هم نبود با پس‌زمینهٔ مصنوعی و کلیدهای میانبر بالا می‌آید. قطع‌شدن استریم هم خودکار دوباره وصل می‌شود. منبع فعلی همیشه پایین جعبهٔ HUD نوشته می‌شود.

### اجرای صحیح در PyCharm

در `Settings → Project → Python Interpreter` گزینهٔ `Add Interpreter → Existing` را بزنید و این فایل را انتخاب کنید:

```text
.venv\Scripts\python.exe
```

اگر بالای کنسول مسیر `...Python312\python.exe` دیده شود، هنوز interpreter سراسری انتخاب شده و خطای NumPy برمی‌گردد. مسیر صحیح باید شامل `WonderSnap\.venv\Scripts\python.exe` باشد.

نسخهٔ فعلی برای اجرای معمولی این اشتباه را خودکار تشخیص می‌دهد و برنامه را با `.venv` دوباره اجرا می‌کند. بااین‌حال برای استفاده از Debugger خود PyCharm همچنان interpreter پروژه را روی `.venv` قرار دهید.

برای مدل‌هایی که مسیر استریم متفاوت دارند:

```powershell
python wondersnap.py --rtsp-path '/cam/realmonitor?channel=1&subtype=1'
```

مسیر پیش‌فرض جریان اصلی (main stream) برابر `/cam/realmonitor?channel=1&subtype=0` است. برای تغییر کامل مشخصات:

```powershell
python wondersnap.py --rtsp-host 192.168.1.20 --rtsp-user admin --rtsp-path '/Streaming/Channels/101'
```

برای استفاده از وب‌کم محلی به‌جای RTSP:

```powershell
python wondersnap.py --webcam --camera 1
```

سوییچ‌های دیگر: `--no-webcam-fallback` (اگر RTSP وصل نشد سراغ وب‌کم نرود) و `--no-mirror` (تصویر آینه‌ای نشود).

برای تست ساخت مدل‌ها بدون باز کردن دوربین یا پنجره:

```powershell
python wondersnap.py --self-test --particles 10000
```

> تشخیص «بشکن» صرفاً بصری است؛ نور مناسب، دیده‌شدن کامل دست و قرار گرفتن کف دست روبه‌روی دوربین نتیجه را بهتر می‌کند. اگر دوربین یا ورود RTSP در دسترس نباشد برنامه همچنان با پس‌زمینهٔ مصنوعی و کلیدهای میانبر اجرا می‌شود. پاسخ `401 Unauthorized` یعنی نام کاربری/رمز دوربین باید بررسی شود؛ پاسخ `404` معمولاً یعنی `--rtsp-path` درست نیست.
