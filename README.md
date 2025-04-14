# Çok Etiketli Görüntü Sınıflandırma (Multi-Label Classification with CNN)

Bu proje, **YOLOv8 formatında etiketlenmiş bir veri setini** kullanarak, el ile oluşturulmuş bir CNN (Convolutional Neural Network) ile **çok etiketli sınıflandırma** yapmayı amaçlamaktadır. 
YOLO modeli doğrudan kullanılmamış, bunun yerine PyTorch ile sıfırdan oluşturulan katmanlarla temel bir yapay sinir ağı eğitilmiştir.

##  Amaç ve Kapsam

Bu çalışmada, öğrencilerin kendi donanımları üzerinde CUDA destekli derin öğrenme modelleri eğitmesi hedeflenmiştir.  
Özellikle:
- NVIDIA ekran kartı olan öğrenciler CUDA üzerinden model eğitmiştir.  
- YOLO yerine **elle tanımlanmış CNN mimarisi** tercih edilmiştir.  
- Model, **sekiz sınıfı** (örneğin: Ceylan, Scooter, Basketbol topu vb.) aynı anda tanıyacak şekilde tasarlanmıştır.  
- Bazı görsellerde birden fazla nesne yer almakta olup, bu nedenle model **multi-label** (çok etiketli) yapıdadır.  

##  Veri Seti Hazırlığı

 **Etiketleme:**  
   Görseller, [Roboflow](https://roboflow.com) platformu kullanılarak **YOLOv8 formatında** etiketlenmiştir. Her `.txt` dosyası, görseldeki nesnelerin sınıf kimlikleri ve koordinatlarını içerir.



##  Model Mimarisi

Model, PyTorch kullanılarak tanımlanmış **basit ve modüler bir CNN yapısı**dır.  
Toplamda 4 adet `Conv + ReLU + MaxPool` bloğu ve bir `Fully Connected` (Linear) katman içerir.  
Modelin sonunda 8 boyutlu bir çıktı vektörü yer alır, her bir boyut bir sınıfı temsil eder.

> **Loss Fonksiyonu:** `BCEWithLogitsLoss` (çoklu etiket için uygundur)  
> **Eğitim:** AMP (otomatik karma hassasiyet) ile hızlandırılmıştır  
> **Ağ Çıkışı:** Her sınıf için sigmoid aktivasyonu ile 0-1 arası olasılık değeri

##  Eğitim

Model sadece **CUDA destekli cihazlarda** çalıştırılmak üzere tasarlanmıştır.  
Eğitim süreci `train_1.py` dosyasında gerçekleştirilir:

```bash
python train_1.py
Eğitim sürecinde GPU sıcaklığı ve bellek kullanımı da izlenmektedir (nvidia-smi ile entegre).
Model belirlenen epoch sayısı boyunca eğitilir ve en son "fast_multilabel_cnn.pth" olarak kaydedilir.

Test
Gerçek zamanlı video akışında modelin tahmini görselleştirilir

Trackbar ile video üzerinde ileri/geri sarma yapılabilir

Tahminler her 3 karede bir alınır, aralarda son tahmin korunur.
