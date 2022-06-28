## Virtual Environment

### python 2.x
```
python -m venv env
```

### python 3.x
```
python3 -m venv env
```

---

## pip install packege

```
pip install -r requirements.txt
```

---

## File Tree

```
ML
├── api
│   ├── ai
│   │   ├── best_model.pth
│   │   ├── django_fn.py
│   │   └── model.py
│   ├── migrations
│   │   ├── __init__.py
│   │   └── 0001_initial.py
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── env
│   └── ...
├── ML
│   ├── __init__.py
│   ├── asgi.py
│   ├── urls.py
│   └── wsgi.py
├── .gitignore
├── manage.py
├── requirements.txt
└── secrets.json
``` 

---

## Venv Activate

### for Mac
```
source env/bin/activate
```

### for Windows
```
env\Scripts\activate.bat
```


---

## Start Server
```
$ ~/ML   python manage.py runserver
```