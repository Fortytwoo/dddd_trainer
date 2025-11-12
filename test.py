import ddddocr

ocr = ddddocr.DdddOcr(
    det=False,
    ocr=False,
    show_ad=False,
    import_onnx_path=r"C:\Users\12460\Documents\project\dddd_trainer\projects\test_18\models\test_18_1.0_30_4000_2025-11-13-03-04-50.onnx",
    charsets_path=r"C:\Users\12460\Documents\project\dddd_trainer\projects\test_18\models\charsets.json",
)

with open(r"C:\Users\12460\Documents\WeChat Files\wxid_ch45oo067gju22\FileStorage\File\2025-11\练习平台18题验证码生成器源码&EXE\500\3-369_1762970532.png", "rb") as f:
    image = f.read()

res = ocr.classification(image)
print(res)
