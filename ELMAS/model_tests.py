import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# ==============================================================================
# JUNIOR SEVİYE İÇİN KRİTİK STANDARTLAR
# ==============================================================================

def main():
    # 1. Modeli ve yardımcı yapıları (scaler/encoder) diski okuma (rb) modunda yükleme
    # Pickle dosyaları binary (ikili) olduğu için mutlaka 'rb' modu kullanılır.
    with open("30-diamond_model_complete.pkl", "rb") as f:
        saved_data = pickle.load(f)

    # 2. Sözlük içinden eğitilmiş makine öğrenmesi modelini çekme
    model = saved_data["model"]

    # 3. Canlı veri veya test verisini okuyup ön işlemleri tamamlanmış halde modele verme
    X_test_scaled = pd.read_csv("30_testdatascaled.csv", index_col=0)

    # 4. Model üzerinden fiyat/hedef değişken tahminini alma
    predictions = model.predict(X_test_scaled)
    print(predictions)

# ==============================================================================
# BU YAPI NEDEN ÇOK ÖNEMLİ? (if __name__ == "__main__":)
# ==============================================================================
# Bu blok, dosya doğrudan çalıştırıldığında (script olarak) main() fonksiyonunu tetikler.
# Eğer bu dosya başka bir projeye 'import' edilirse, main() otomatik çalışmaz;
# böylece kod karmaşası ve istenmeyen işlemler engellenmiş olur.
# Production (Canlı) ortam kodlarında bu standart mutlaka kullanılır.
if __name__ == "__main__":
    main()