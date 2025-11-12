import ddddocr

ocr = ddddocr.DdddOcr(
    det=False,
    ocr=False,
    show_ad=False,
    import_onnx_path=r"C:\Users\12460\Documents\project\dddd_trainer\projects\test_4\models\test_4_1.0_23_6000_2025-11-13-01-11-23.onnx",
    charsets_path=r"C:\Users\12460\Documents\project\dddd_trainer\projects\test_4\models\charsets.json",
)

with open(r"C:\Users\12460\Downloads\1112\new\PKKQ_1578462523867.jpg", "rb") as f:
    image = f.read()

res = ocr.classification(image)
print(res)
