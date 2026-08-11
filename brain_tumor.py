    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    
    # ============================
    # Load Trained Model
    # ============================
    model = load_model("brain_tumor_model.keras")
    
    # ============================
    # Class Names
    # ============================
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # ============================
    # Select Image
    # ============================
    Tk().withdraw()
    
    image_path = askopenfilename(
        title="Select MRI Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    
    if image_path == "":
        print("No image selected.")
        exit()
    
    # ============================
    # Read Image
    # ============================
    img = cv2.imread(image_path)
    
    if img is None:
        print("Unable to read image.")
        exit()
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ============================
    # Preprocess Image
    # ============================
    img_resize = cv2.resize(img_rgb, (128, 128))
    img_resize = img_resize.astype("float32") / 255.0
    img_input = np.expand_dims(img_resize, axis=0)
    
    # ============================
    # Prediction
    # ============================
    prediction = model.predict(img_input)
    
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    print("Prediction:", classes[predicted_class])
    print("Confidence: {:.2f}%".format(confidence))
    
    # ============================
    # Show Image
    # ============================
    plt.figure(figsize=(6,6))
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {classes[predicted_class]}\nConfidence: {confidence:.2f}%")
    plt.axis("off")
    plt.show()