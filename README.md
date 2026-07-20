# EmotionViA v2

مشروع تخرج — تصنيف تعابير الوجه (Facial Emotion Recognition) على 3 داتاسيتس
مدموجة (EmoPOI + FER-2013 + RAF-DB)، مبني بالكامل بناءً على `implementation_plan.md`
(تحليل الداتا، مقارنة المعماريات، القرارات المعمارية).

> **ملاحظة تسمية:** هذا المشروع يُصنّف **تعبير الوجه/المشاعر**، وليس **هوية
> الشخص** (Face Identification). المصطلحان مختلفان أكاديميًا.

---

## 1) لماذا الكود متغيّر عن ما هو مذكور حرفيًا في implementation_plan.md؟

بُني هذا الكود في بيئة **بدون إنترنت أو GPU**، فتم اتخاذ قرارين هندسيين واعيين:

1. **إدارة الـ config:** الخطة الأصلية اقترحت Hydra (`+experiment=baseline_v1`
   CLI overrides). بما إن `hydra-core` مش متاح للتثبيت والاختبار هنا، ومنطق
   الـ `defaults:` composition بتاعته حسّاس لتفاصيل namespacing دقيقة، تم استبداله
   بمُحمّل config بسيط ومباشر (`emotionvia_v2/config.py`) بيدمج نفس ملفات YAML
   بمنطق واضح تقدر تتبّعه سطر بسطر. لو عندك `hydra-core` فعليًا في بيئتك ومرتاح
   بيه، ترجع لـ `hydra.main` بسهولة — التغيير محدود في `train.py`/`evaluate.py` فقط.
2. **التنفيذ الفعلي:** لم يتم تشغيل تدريب حقيقي هنا (لا بيانات ولا GPU). كل
   الملفات اتفحصت **نحويًا بالكامل** (`py_compile` على كل ملف) ومنطق الأجزاء
   المستقلة عن `torch` (dedup، الـ subject-wise split) **اتفحص فعليًا وشغّال
   صح** (شوف تفاصيل الاختبار في محادثتنا). لازم تشغّله بنفسك على Colab/جهاز
   فيه GPU للتأكد النهائي، ولو ظهر أي خطأ وقت التشغيل ابعتهولي وهصلحه فورًا.

## 2) هيكل المشروع

```
emotionvia_v2/
├── configs/
│   ├── dataset/{emopoi,fer2013,rafdb}.yaml   # إعدادات كل داتاسيت + مسارات env vars
│   ├── model/backbone.yaml                    # اختيار الـ backbone/attention/temporal
│   └── experiment/baseline_v1.yaml            # التجربة الكاملة (training hyperparams)
├── emotionvia_v2/
│   ├── config.py                              # تحميل ودمج كل ملفات الـ YAML
│   ├── train.py                                # سكريبت التدريب الرئيسي
│   ├── evaluate.py                              # تقييم + Confusion Matrix + Grad-CAM
│   ├── data/
│   │   ├── dedup.py                            # اكتشاف/حذف الصور المكررة (داخلي + عابر للـ splits)
│   │   ├── transforms.py                        # تحويلات موحّدة (grayscale->RGB، augmentation)
│   │   └── datasets/{fer2013,rafdb,emopoi}.py   # كلاسات كل داتاسيت
│   ├── models/
│   │   ├── backbone.py                          # factory لـ 7 backbones (resnet/effnet/convnext/swin/vit/mobilenet)
│   │   ├── attention.py                          # CBAM + Squeeze-Excitation
│   │   ├── temporal.py                           # BiLSTM / TCN / Transformer (لسلاسل EmoPOI)
│   │   ├── fusion.py                             # Concatenation fusion
│   │   └── emotion_model.py                      # الموديل الموحّد (يجمع كل حاجة فوق)
│   ├── losses/
│   │   ├── focal_loss.py
│   │   └── class_balanced_loss.py
│   └── engine/
│       ├── trainer.py                            # حلقة التدريب (MixUp, AMP, early stopping)
│       └── early_stopping.py
├── scripts/resolve_config.py                    # فحص المسارات قبل التدريب
└── tests/data/                                    # unit tests ببيانات صناعية (synthetic)
```

## 3) التركيب (Setup)

```bash
pip install -r requirements.txt
```

اضبط متغيرات البيئة بمسارات الداتاسيتس بعد فك الضغط (عدّل القيم حسب مكانك الفعلي):

```bash
export EMOTIONVIA_EMOPOI_ROOT="/content/emotionvia_v2/EmoPOI"
export EMOTIONVIA_FER2013_ROOT="/content/emotionvia_v2/FER-2013"
export EMOTIONVIA_RAFDB_ROOT="/content/emotionvia_v2/RAF-DB DATASET/DATASET"
export EMOTIONVIA_RAFDB_TRAIN_CSV="/content/emotionvia_v2/.../train_labels.csv"
export EMOTIONVIA_RAFDB_TEST_CSV="/content/emotionvia_v2/.../test_labels.csv"
export EMOTIONVIA_OUTPUT_DIR="/content/emotionvia_v2/outputs/baseline_v1"
```

## 4) التحقق قبل التدريب (Manual Verification)

```bash
python scripts/resolve_config.py --experiment baseline_v1
```

يطبع الـ config كامل بعد الحل، ويتأكد إن كل المسارات موجودة فعليًا قبل ما تبدأ
تدريب طويل وتكتشف غلطة مسار بعد نص ساعة تحميل.

## 5) التدريب

```bash
python -m emotionvia_v2.train --experiment baseline_v1
```

## 6) التقييم وتوليد صور المخرجات (Confusion Matrix, Grad-CAM, ...)

```bash
python -m emotionvia_v2.evaluate --experiment baseline_v1
```

النواتج بتتحفظ في `${EMOTIONVIA_OUTPUT_DIR}`:
`training_curves.png`, `confusion_matrix.png`, `classification_report.txt`,
`sample_predictions.png`, `gradcam.png`.

## 7) الاختبارات (Unit Tests)

```bash
pytest tests/ -v
```

الاختبارات بتبني بيانات صناعية صغيرة جدًا (بايثون + Pillow فقط) بنفس هيكل
الداتاسيتس الحقيقية — مفيش حاجة تنزيل داتا حقيقية عشان تشغّل الاختبارات دي.
بتغطّي: عدّ العينات، الـ dedup الداخلي، إزالة التسريب العابر بين splits، تحويل
labels لـ 0-based، وأهمها: **عدم تداخل نفس الـ person_id بين train/val/test في
EmoPOI**.

## 8) تبديل الـ backbone / تفعيل CBAM أو التدريب الزمني (temporal)

كل ده متاح كـ config فقط، من غير أي تعديل كود:

```yaml
# configs/model/backbone.yaml
model:
  backbone: efficientnet_v2_s   # أو resnet18 / resnet50 / convnext_tiny / swin_t / mobilenet_v3_large / vit_b_16
  use_attention: true
  attention_type: cbam          # أو se / none
  use_temporal: false           # true لو عايز BiLSTM/TCN فوق سلاسل EmoPOI (محتاج is_sequence: true في emopoi.yaml)
```

## 9) نقاط مهمة للمناقشة (من implementation_plan.md)

- **Cross-split leakage في FER-2013** كانت أخطر ثغرة في الداتا الأصلية —
  التعامل معاها موثّق ومُختبَر في `dedup.py` + `test_fer2013.py`.
- **Subject-wise split في EmoPOI** إلزامي (مش اختياري) لمنع تسريب فريمات نفس
  الشخص بين splits — موثّق ومُختبَر في `test_emopoi.py`.
- **Focal Loss + Class-Balanced weights** بيعالجوا imbalance وصل لـ 17× —
  اتنفذوا في `losses/`، قابلين للتبديل لـ `weighted_ce` عادي من الـ config لو
  حبيت تقارن الأداء بين الاتنين في تقرير المشروع (نقطة قوية للمناقشة).
- **CBAM** بيتفعّل بس مع الـ backbones اللي بترجّع spatial feature maps (كل
  الـ CNNs)، وبيتعطّل تلقائيًا مع تحذير واضح لـ Swin-T/ViT.
