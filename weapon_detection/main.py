from tkinter import *
from tkinter import filedialog, messagebox
import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


main = Tk()
main.title("Improving Weapon Recognition in the Context of Skyrim Throught the Application of ML Techniques")
main.geometry("1300x1200")
main.config(bg="lightgray")

dataset_path = ""
X = None
Y = None
X_train = None
X_test = None
Y_train = None
Y_test = None
Y_train_labels = None
Y_test_labels = None
cnn_model = None

p = r = f = acc = 0
p1 = r1 = f1v = acc1 = 0

shapes = ['sword', 'Sniper', 'SMG', 'shotgun', 'Handgun', 'Grenadelauncher', 'Bazooka', 'AutomaticRifle']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR_DEFAULT = os.path.join(BASE_DIR, "Data")
TEST_DIR_DEFAULT = os.path.join(BASE_DIR, "testimages")

os.makedirs(MODEL_DIR, exist_ok=True)


def log(msg):
    text.insert(END, str(msg) + "\n")
    text.see(END)


def clear_text():
    text.delete('1.0', END)


def get_label_from_folder(folder_name):
    try:
        return shapes.index(folder_name)
    except ValueError:
        return -1


def preprocess_image(img_path, size=(64, 64)):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        return img
    except Exception:
        return None


def build_cnn_model(input_shape=(64, 64, 3), num_classes=8):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def uploadDataset():
    global dataset_path
    clear_text()
    selected = filedialog.askdirectory(initialdir=BASE_DIR, title="Select Dataset Folder")
    if not selected:
        log("Dataset selection cancelled.")
        return
    dataset_path = selected
    log(f"Dataset loaded from: {dataset_path}")
    log("Now click 'Image Processing & Normalization'")


def imageProcessing():
    global X, Y, X_train, X_test, Y_train, Y_test, Y_train_labels, Y_test_labels, dataset_path
    clear_text()

    if dataset_path == "":
        if os.path.exists(DATA_DIR_DEFAULT):
            dataset_path = DATA_DIR_DEFAULT
            log(f"No folder selected. Using default dataset folder: {dataset_path}")
        else:
            messagebox.showerror("Error", "Please upload/select dataset folder first.")
            return

    X_list = []
    Y_list = []
    total_images = 0

    log("Reading images from dataset...")

    for class_name in shapes:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            log(f"Warning: folder not found -> {class_dir}")
            continue

        class_count = 0
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)

            if not os.path.isfile(file_path):
                continue

            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                img = preprocess_image(file_path, (64, 64))
                if img is not None:
                    X_list.append(img)
                    Y_list.append(get_label_from_folder(class_name))
                    class_count += 1
                    total_images += 1

        log(f"{class_name}: {class_count} images loaded")

    if len(X_list) == 0:
        messagebox.showerror("Error", "No valid images found in dataset.")
        return

    X = np.array(X_list, dtype="float32")
    Y_labels = np.array(Y_list, dtype="int32")
    Y = to_categorical(Y_labels, num_classes=len(shapes))

    np.save(os.path.join(MODEL_DIR, "X_processed.npy"), X)
    np.save(os.path.join(MODEL_DIR, "Y_processed.npy"), Y)
    np.save(os.path.join(MODEL_DIR, "Y_labels.npy"), Y_labels)

    X_train, X_test, Y_train, Y_test, Y_train_labels, Y_test_labels = train_test_split(
        X, Y, Y_labels,
        test_size=0.20,
        random_state=42,
        stratify=Y_labels
    )

    log(f"\nTotal number of images found in dataset : {len(X)}")
    log(f"Total classes found in dataset          : {len(shapes)}")
    log(f"Training images                         : {len(X_train)}")
    log(f"Testing images                          : {len(X_test)}")
    log(f"Class names                             : {shapes}")
    log("\nImage processing completed successfully.")


def gnb():
    global p, r, f, acc
    clear_text()

    global X_train, X_test, Y_train_labels, Y_test_labels

    if X_train is None or X_test is None:
        if os.path.exists(os.path.join(MODEL_DIR, "X_processed.npy")) and os.path.exists(os.path.join(MODEL_DIR, "Y_labels.npy")):
            X_loaded = np.load(os.path.join(MODEL_DIR, "X_processed.npy"))
            Y_labels_loaded = np.load(os.path.join(MODEL_DIR, "Y_labels.npy"))

            Xtr, Xte, Ytr_lab, Yte_lab = train_test_split(
                X_loaded, Y_labels_loaded,
                test_size=0.20,
                random_state=42,
                stratify=Y_labels_loaded
            )

            X_train_local = Xtr
            X_test_local = Xte
            Y_train_local = Ytr_lab
            Y_test_local = Yte_lab
        else:
            messagebox.showerror("Error", "Please run Image Processing first.")
            return
    else:
        X_train_local = X_train
        X_test_local = X_test
        Y_train_local = Y_train_labels
        Y_test_local = Y_test_labels

    x_train = X_train_local.reshape((X_train_local.shape[0], -1))
    x_test = X_test_local.reshape((X_test_local.shape[0], -1))

    nb_classifier = GaussianNB()
    nb_classifier.fit(x_train, Y_train_local)
    y_pred = nb_classifier.predict(x_test)

    acc = accuracy_score(Y_test_local, y_pred) * 100
    p = precision_score(Y_test_local, y_pred, average='macro', zero_division=0) * 100
    r = recall_score(Y_test_local, y_pred, average='macro', zero_division=0) * 100
    f = f1_score(Y_test_local, y_pred, average='macro', zero_division=0) * 100

    cm = confusion_matrix(Y_test_local, y_pred)
    clr = classification_report(Y_test_local, y_pred, target_names=shapes, zero_division=0)

    with open(os.path.join(MODEL_DIR, "gnb_model.pkl"), "wb") as file:
        pickle.dump(nb_classifier, file)

    log("========== GNB CLASSIFIER RESULTS ==========")
    log(f"ML Model Accuracy  = {acc:.2f}%")
    log(f"Model Precision    = {p:.2f}%")
    log(f"ML Model Recall    = {r:.2f}%")
    log(f"ML Model F1-Score  = {f:.2f}%\n")

    log("Confusion Matrix:")
    log(cm)

    log("\nClassification Report:")
    log(clr)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix of GNB Classifier")
    plt.colorbar()
    tick_marks = np.arange(len(shapes))
    plt.xticks(tick_marks, shapes, rotation=45)
    plt.yticks(tick_marks, shapes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def cnnModel():
    global cnn_model, acc1, p1, r1, f1v
    clear_text()

    global X_train, X_test, Y_train, Y_test, Y_test_labels

    if X_train is None or X_test is None:
        if os.path.exists(os.path.join(MODEL_DIR, "X_processed.npy")) and \
           os.path.exists(os.path.join(MODEL_DIR, "Y_processed.npy")) and \
           os.path.exists(os.path.join(MODEL_DIR, "Y_labels.npy")):

            X_loaded = np.load(os.path.join(MODEL_DIR, "X_processed.npy"))
            Y_loaded = np.load(os.path.join(MODEL_DIR, "Y_processed.npy"))
            Y_labels_loaded = np.load(os.path.join(MODEL_DIR, "Y_labels.npy"))

            X_train_local, X_test_local, Y_train_local, Y_test_local, _, Y_test_lab_local = train_test_split(
                X_loaded, Y_loaded, Y_labels_loaded,
                test_size=0.20,
                random_state=42,
                stratify=Y_labels_loaded
            )
        else:
            messagebox.showerror("Error", "Please run Image Processing first.")
            return
    else:
        X_train_local = X_train
        X_test_local = X_test
        Y_train_local = Y_train
        Y_test_local = Y_test
        Y_test_lab_local = Y_test_labels

    cnn_model = build_cnn_model(input_shape=(64, 64, 3), num_classes=len(shapes))
    weights_path = os.path.join(MODEL_DIR, "model.weights.h5")
    history_path = os.path.join(MODEL_DIR, "history.pckl")

    loaded_weights = False

    if os.path.exists(weights_path):
        try:
            cnn_model.load_weights(weights_path)
            loaded_weights = True
            log("Existing CNN weights loaded successfully.")
        except Exception as e:
            log(f"Old weight loading failed: {str(e)}")
            log("Training CNN again with current TensorFlow/Keras version...")

    if not loaded_weights:
        log("Training CNN model...")
        history = cnn_model.fit(
            X_train_local,
            Y_train_local,
            batch_size=16,
            epochs=15,
            validation_data=(X_test_local, Y_test_local),
            verbose=1
        )

        cnn_model.save_weights(weights_path)

        with open(history_path, "wb") as f_hist:
            pickle.dump(history.history, f_hist)

        log("CNN model trained and weights saved.")

    y_pred_probs = cnn_model.predict(X_test_local)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(Y_test_local, axis=1)

    acc1 = accuracy_score(y_true, y_pred) * 100
    p1 = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    r1 = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1v = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100

    cm = confusion_matrix(y_true, y_pred)
    clr = classification_report(y_true, y_pred, target_names=shapes, zero_division=0)

    log("========== CNN CLASSIFIER RESULTS ==========")
    log(f"Deep Learning Model Accuracy  = {acc1:.2f}%")
    log(f"Deep Learning Model Precision = {p1:.2f}%")
    log(f"Deep Learning Model Recall    = {r1:.2f}%")
    log(f"Deep Learning Model F1-Score  = {f1v:.2f}%\n")

    log("Confusion Matrix:")
    log(cm)

    log("\nClassification Report:")
    log(clr)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix of CNN Classifier")
    plt.colorbar()
    tick_marks = np.arange(len(shapes))
    plt.xticks(tick_marks, shapes, rotation=45)
    plt.yticks(tick_marks, shapes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    if os.path.exists(history_path):
        try:
            with open(history_path, "rb") as f_hist:
                hist_data = pickle.load(f_hist)

            if 'accuracy' in hist_data and 'loss' in hist_data:
                plt.figure(figsize=(10, 6))
                plt.plot(hist_data['accuracy'], marker='o', label='Accuracy')
                plt.plot(hist_data['loss'], marker='o', label='Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Value')
                plt.title('CNN Performance Evaluation')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()
        except Exception as e:
            log(f"Could not display training history graph: {str(e)}")


def predict():
    global cnn_model

    if cnn_model is None:
        weights_path = os.path.join(MODEL_DIR, "model.weights.h5")
        if os.path.exists(weights_path):
            try:
                cnn_model = build_cnn_model(input_shape=(64, 64, 3), num_classes=len(shapes))
                cnn_model.load_weights(weights_path)
            except Exception:
                messagebox.showerror("Error", "Please click 'Build & Train CNN Model' first.")
                return
        else:
            messagebox.showerror("Error", "Please click 'Build & Train CNN Model' first.")
            return

    file_path = filedialog.askopenfilename(
        initialdir=TEST_DIR_DEFAULT if os.path.exists(TEST_DIR_DEFAULT) else BASE_DIR,
        title="Select Test Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )

    if not file_path:
        return

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Could not read selected image.")
        return

    resized = cv2.resize(img, (64, 64))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    test = np.array(rgb, dtype='float32') / 255.0
    test = np.expand_dims(test, axis=0)

    preds = cnn_model.predict(test)
    pred_index = np.argmax(preds)
    pred_label = shapes[pred_index]
    confidence = float(np.max(preds)) * 100

    clear_text()
    log("========== TEST IMAGE CLASSIFICATION ==========")
    log(f"Predicted Class = {pred_label}")
    log(f"Confidence      = {confidence:.2f}%")

    display_img = cv2.resize(img, (500, 500))
    cv2.putText(display_img, f'Class: {pred_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_img, f'Confidence: {confidence:.2f}%', (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Weapon Detection Result', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def graph():
    global p, r, f, acc, p1, r1, f1v, acc1

    if acc == 0 and acc1 == 0:
        messagebox.showerror("Error", "Run GNB and CNN models first.")
        return

    df = pd.DataFrame([
        ['GNB', 'Precision', p],
        ['GNB', 'Recall', r],
        ['GNB', 'F1 Score', f],
        ['GNB', 'Accuracy', acc],
        ['CNN', 'Precision', p1],
        ['CNN', 'Recall', r1],
        ['CNN', 'F1 Score', f1v],
        ['CNN', 'Accuracy', acc1],
    ], columns=['Algorithms', 'Metrics', 'Value'])

    pivot_df = df.pivot(index='Algorithms', columns='Metrics', values='Value')
    pivot_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Performance Metrics Comparison Between Algorithms')
    plt.ylabel('Score')
    plt.xlabel('Algorithms')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def close():
    main.destroy()


font = ('times', 15, 'bold')
title = Label(main, text='Improving Weapon Recognition in the Context of Skyrim Throught the Application of ML Techniques')
title.config(bg='powder blue', fg='olive drab')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20, y=100)
uploadButton.config(font=ff)

processButton = Button(main, text="Image Processing & Normalization", command=imageProcessing)
processButton.place(x=20, y=150)
processButton.config(font=ff)

mlButton = Button(main, text="GNB Classifier", command=gnb)
mlButton.place(x=20, y=200)
mlButton.config(font=ff)

modelButton = Button(main, text="Build & Train CNN Model", command=cnnModel)
modelButton.place(x=20, y=250)
modelButton.config(font=ff)

predictButton = Button(main, text="Upload Test Image & Classify", command=predict)
predictButton.place(x=20, y=300)
predictButton.config(font=ff)

graphButton = Button(main, text="Performance Graph", command=graph)
graphButton.place(x=20, y=350)
graphButton.config(font=ff)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=20, y=400)
exitButton.config(font=ff)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=85)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450, y=100)
text.config(font=font1)

main.mainloop()