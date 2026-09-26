# Beer Classifier / Bira Sınıflandırıcı

## English

A clean FastAPI web application that uses a PyTorch neural network to classify
beer as **IPA**, **Light Lager**, or **Premium Lager** from four measurements:

- **OG:** Original gravity before fermentation
- **ABV:** Alcohol by volume percentage
- **pH:** Acidity level of the beer
- **IBU:** Bitterness level of the beer

### Features

- Responsive, minimal web interface
- PyTorch model inference on CPU
- Prediction confidence and class probabilities
- JSON prediction API and interactive API documentation
- Safe checkpoint loading with `weights_only=True`

### Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the application

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API documentation is available at
<http://127.0.0.1:8000/docs>.

The included model uses the following class mapping:

- `0`: IPA
- `1`: Light Lager
- `2`: Premium Lager

To use another checkpoint or change the displayed class names:

```bash
MODEL_PATH=models/model.pth \
CLASS_NAMES="IPA,Light Lager,Premium Lager" \
uvicorn app.main:app --reload
```

### API example

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"og":1.05,"abv":5.0,"ph":4.2,"ibu":35}'
```

---

## Türkçe

PyTorch sinir ağı kullanarak dört ölçüm üzerinden birayı **IPA**,
**Light Lager** veya **Premium Lager** olarak sınıflandıran sade bir FastAPI web
uygulamasıdır:

- **OG:** Fermantasyon öncesi başlangıç yoğunluğu
- **ABV:** Hacimce alkol yüzdesi
- **pH:** Biranın asitlik seviyesi
- **IBU:** Biranın acılık derecesi

### Özellikler

- Mobil uyumlu, sade web arayüzü
- CPU üzerinde PyTorch model tahmini
- Güven oranı ve sınıf olasılıkları
- JSON tahmin API'si ve etkileşimli API dokümantasyonu
- `weights_only=True` ile güvenli checkpoint yükleme

### Kurulum

Python 3.10 veya daha yeni bir sürüm önerilir.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Uygulamayı çalıştırma

```bash
uvicorn app.main:app --reload
```

<http://127.0.0.1:8000> adresini açın. Etkileşimli API dokümantasyonu
<http://127.0.0.1:8000/docs> adresindedir.

Projeye dahil edilen model şu sınıf eşlemesini kullanır:

- `0`: IPA
- `1`: Light Lager
- `2`: Premium Lager

Başka bir checkpoint kullanmak veya sınıf adlarını değiştirmek için:

```bash
MODEL_PATH=models/model.pth \
CLASS_NAMES="IPA,Light Lager,Premium Lager" \
uvicorn app.main:app --reload
```

### API örneği

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"og":1.05,"abv":5.0,"ph":4.2,"ibu":35}'
```
