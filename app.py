
import os
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "static" / "uploads"
OUTPUT_DIR = BASE / "static" / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = "image-processing-practicals"

PRACTICALS = {
    1: {"title":"RGB / Grayscale, Arithmetic & Bitwise Operations",
        "ops":["Grayscale","RGB Split","Image Addition","Image Subtraction","Bitwise AND","Bitwise OR","Bitwise XOR","Bitwise NOT"]},
    2: {"title":"2-D Geometric Transformations",
        "ops":["Translation","Rotation","Scaling","X Shearing","Y Shearing","Reflection","Cropping"]},
    3: {"title":"Histogram Equalization, Spatial Enhancement & Thresholding",
        "ops":["Histogram Equalization","Gamma Correction","Binary Threshold","Adaptive Threshold","Otsu Threshold"]},
    4: {"title":"Spatial Domain Filters",
        "ops":["Averaging Filter","Gaussian Filter","Median Filter","Bilateral Filter","Laplacian Filter"]},
    5: {"title":"Image Inpainting",
        "ops":["Navier-Stokes (NS) Method","Telea Method"]},
    6: {"title":"Lossless Image Compression",
        "ops":["PNG Lossless Compression","RLE Compression"]},
    7: {"title":"Morphological Operations",
        "ops":["Erosion","Dilation","Opening","Closing","Morphological Gradient","Top Hat","Black Hat"]},
    8: {"title":"Object Detection using Correlation",
        "ops":["Correlation Detection"]},
    9: {"title":"Top-Hat Transformation",
        "ops":["Top-Hat Transformation","Black-Hat Transformation"]},
    10: {"title":"Colour Space Conversion",
         "ops":["HSV","YCrCb","LAB","RGB"]},
    11: {"title":"Edge Detection",
         "ops":["Canny","Sobel","Laplacian","Scharr"]},
}

ALLOWED_EXT = {"png","jpg","jpeg","bmp","webp"}

def ext_ok(fn):
    return "." in fn and fn.rsplit(".",1)[1].lower() in ALLOWED_EXT

def stamp(img, roll="CS24107"):
    out = img.copy()
    h, w = out.shape[:2]
    if len(out.shape)==2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (max(0,w-125), max(0,h-28)), (w, h), (255,255,255), -1)
    cv2.putText(out, roll, (max(3,w-118), max(18,h-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
    return out

def load_image(file_storage, fallback_name):
    if file_storage and file_storage.filename:
        if not ext_ok(file_storage.filename):
            return None, "Unsupported image format."
        path = UPLOAD_DIR / Path(file_storage.filename).name
        file_storage.save(path)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return img, None
    img = cv2.imread(str(BASE/"static"/"sample_inputs"/fallback_name), cv2.IMREAD_COLOR)
    return img, None

def process(pno, op, img, img2=None, mask=None):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if pno == 1:
        if op == "Grayscale": return g
        if op == "RGB Split": return np.hstack([img[:,:,2], img[:,:,1], img[:,:,0]])
        if img2 is None: img2 = img.copy()
        img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
        if op == "Image Addition": return cv2.add(img, img2)
        if op == "Image Subtraction": return cv2.absdiff(img, img2)
        if op == "Bitwise AND": return cv2.bitwise_and(img, img2)
        if op == "Bitwise OR": return cv2.bitwise_or(img, img2)
        if op == "Bitwise XOR": return cv2.bitwise_xor(img, img2)
        if op == "Bitwise NOT": return cv2.bitwise_not(img)
    if pno == 2:
        h,w = g.shape
        if op == "Translation":
            M = np.float32([[1,0,80],[0,1,45]])
            return cv2.warpAffine(img,M,(w,h))
        if op == "Rotation":
            M = cv2.getRotationMatrix2D((w/2,h/2),30,1.0)
            return cv2.warpAffine(img,M,(w,h))
        if op == "Scaling":
            return cv2.resize(img,None,fx=0.75,fy=0.75,interpolation=cv2.INTER_AREA)
        if op == "X Shearing":
            M = np.float32([[1,0.35,0],[0,1,0],[0,0,1]])
            return cv2.warpPerspective(img,M,(int(w*1.35),h))
        if op == "Y Shearing":
            M = np.float32([[1,0,0],[0.35,1,0],[0,0,1]])
            return cv2.warpPerspective(img,M,(w,int(h*1.35)))
        if op == "Reflection":
            return cv2.flip(img,1)
        if op == "Cropping":
            return img[h//6:5*h//6,w//6:5*w//6]
    if pno == 3:
        if op == "Histogram Equalization": return cv2.equalizeHist(g)
        if op == "Gamma Correction":
            gamma=1.6
            table=np.array([((i/255.0)**gamma)*255 for i in np.arange(256)]).astype("uint8")
            return cv2.LUT(img,table)
        if op == "Binary Threshold":
            _,o=cv2.threshold(g,127,255,cv2.THRESH_BINARY); return o
        if op == "Adaptive Threshold":
            return cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        if op == "Otsu Threshold":
            _,o=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU); return o
    if pno == 4:
        if op == "Averaging Filter": return cv2.blur(img,(7,7))
        if op == "Gaussian Filter": return cv2.GaussianBlur(img,(7,7),0)
        if op == "Median Filter": return cv2.medianBlur(img,7)
        if op == "Bilateral Filter": return cv2.bilateralFilter(img,9,75,75)
        if op == "Laplacian Filter": return cv2.convertScaleAbs(cv2.Laplacian(g,cv2.CV_64F))
    if pno == 5:
        if mask is None: return img
        m = cv2.threshold(cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)[1]
        flags = cv2.INPAINT_NS if "NS" in op else cv2.INPAINT_TELEA
        return cv2.inpaint(img,m,3,flags)
    if pno == 6:
        if op == "PNG Lossless Compression": return img
        if op == "RLE Compression":
            g2=g
            return cv2.cvtColor(g2,cv2.COLOR_GRAY2BGR)
    if pno == 7:
        kernel=np.ones((5,5),np.uint8)
        if op == "Erosion": return cv2.erode(g,kernel,iterations=1)
        if op == "Dilation": return cv2.dilate(g,kernel,iterations=1)
        if op == "Opening": return cv2.morphologyEx(g,cv2.MORPH_OPEN,kernel)
        if op == "Closing": return cv2.morphologyEx(g,cv2.MORPH_CLOSE,kernel)
        if op == "Morphological Gradient": return cv2.morphologyEx(g,cv2.MORPH_GRADIENT,kernel)
        if op == "Top Hat": return cv2.morphologyEx(g,cv2.MORPH_TOPHAT,kernel)
        if op == "Black Hat": return cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,kernel)
    if pno == 8:
        if img2 is None: return img
        tmpl = img2
        result = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(result)
        th,tw=tmpl.shape[:2]
        out=img.copy()
        cv2.rectangle(out,maxloc,(maxloc[0]+tw,maxloc[1]+th),(0,255,0),3)
        cv2.putText(out,f"Correlation: {maxv:.4f}",(15,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        return out
    if pno == 9:
        kernel=np.ones((15,15),np.uint8)
        if op == "Top-Hat Transformation":
            return cv2.morphologyEx(g,cv2.MORPH_TOPHAT,kernel)
        return cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,kernel)
    if pno == 10:
        code={"HSV":cv2.COLOR_BGR2HSV,"YCrCb":cv2.COLOR_BGR2YCrCb,"LAB":cv2.COLOR_BGR2LAB,"RGB":cv2.COLOR_BGR2RGB}[op]
        return cv2.cvtColor(img,code)
    if pno == 11:
        if op == "Canny": return cv2.Canny(g,100,200)
        if op == "Sobel":
            sx=cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3); sy=cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3)
            return cv2.convertScaleAbs(cv2.magnitude(sx,sy))
        if op == "Laplacian": return cv2.convertScaleAbs(cv2.Laplacian(g,cv2.CV_64F))
        if op == "Scharr":
            sx=cv2.Scharr(g,cv2.CV_64F,1,0); sy=cv2.Scharr(g,cv2.CV_64F,0,1)
            return cv2.convertScaleAbs(cv2.magnitude(sx,sy))
    return img

@app.route("/")
def home():
    return render_template("index.html", practicals=PRACTICALS)

@app.route("/practical/<int:pno>", methods=["GET","POST"])
def practical(pno):
    if pno not in PRACTICALS:
        return "Practical not found", 404
    data=PRACTICALS[pno]
    result=None
    input_name=None
    output_name=None
    message=None
    if request.method=="POST":
        op=request.form.get("operation",data["ops"][0])
        img,_=load_image(request.files.get("image"), "sample_color.png")
        img2,_=load_image(request.files.get("image2"), "template.png")
        mask,_=load_image(request.files.get("mask"), "feather_mask.png")
        if img is None:
            flash("Please upload a valid image.")
            return redirect(request.url)
        try:
            out=process(pno,op,img,img2,mask)
            out=stamp(out, request.form.get("roll","CS24107"))
            uid=str(abs(hash((os.urandom(8),pno,op))))
            input_path=UPLOAD_DIR/f"input_{uid}.png"
            output_path=OUTPUT_DIR/f"output_{uid}.png"
            cv2.imwrite(str(input_path),img)
            cv2.imwrite(str(output_path),out)
            input_name=input_path.name
            output_name=output_path.name
            result={"operation":op}
        except Exception as e:
            flash(f"Processing error: {e}")
    return render_template("practical.html", pno=pno, data=data, input_name=input_name, output_name=output_name, result=result)

if __name__=="__main__":
    app.run(debug=True)
