from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
import io
import base64

app = Flask(__name__)

# ✅ Load model ONCE (important for performance)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("model.pt", map_location=device)
model.to(device)
model.eval()

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ✅ Transform
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ✅ Prediction function (reuse everywhere)
def model_predict(img_pil):
    with torch.inference_mode():
        img = transform(img_pil).unsqueeze(0).to(device)
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        
        entropy = -(probs * probs.clamp(min=1e-9).log()).sum().item()
        max_entropy = torch.log(torch.tensor(float(len(class_names))))

        if entropy > 0.7 * max_entropy:
            return "Not Trash / Unknown", 0.0

        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = class_names[pred_idx]
        confidence = probs.max().item() * 100

    return pred_class, confidence


# ================= ROUTES =================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/team")
def team():
    return render_template("team.html")


@app.route("/user-guide")
def guide():
    return render_template("guide.html")


@app.route("/testing")
def testing():
    return render_template("big_testing.html")


# ✅ OLD IMAGE UPLOAD (kept)
@app.route("/", methods=["POST"])
def predict_upload():
    file = request.files.get("image")

    if not file or file.filename == "":
        return render_template("index.html", error="No image selected")

    img_pil = Image.open(file).convert("RGB")

    pred_class, confidence = model_predict(img_pil)

    # Convert image to base64 (for preview)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return render_template(
        "index.html",
        prediction=pred_class,
        confidence=f"{confidence:.2f}",
        img_data=img_str
    )


# ✅ 🔥 NEW REAL-TIME API (THIS IS THE KEY)
@app.route("/predict", methods=["POST"])
def predict_realtime():
    file = request.files.get("image")

    if not file:
        return jsonify({"error": "No image"}), 400

    try:
        img_pil = Image.open(file.stream).convert("RGB")

        pred_class, confidence = model_predict(img_pil)

        return jsonify({
            "prediction": pred_class,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= RUN =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)