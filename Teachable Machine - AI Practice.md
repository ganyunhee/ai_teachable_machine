## Purpose
---
Learn how to utilize Teachable Machine by Google in order to create basic predictive models (예측 모델). Create a basic predictive model that distinguishes between two commonly confused dog breeds: Siberian husky and Alaskan malamute. Additionally, learn how to manipulate source files form Teachable Machine in order to add functionality to the created predictive model.
## Process
---
#### Using the Teachable Machine

![[Pasted image 20230827181713.png]]

First, ten images for each of the two dog breeds were gathered as initial datasets and uploaded onto the Teachable Machine interface.

![[Pasted image 20230827181623.png | 320]]      ![[Pasted image 20230827181554.png | 320]]


Next, the model is trained(학습) and the results and the results are confirmed on the preview(미리 보기).

  ![[Pasted image 20230827182850.png | 200]]     ![[Pasted image 20230827182915.png | 200]]     ![[Pasted image 20230827182942.png | 200]]


![[Pasted image 20230827183616.png | 300]]     ![[Pasted image 20230827184103.png | 300]]



For every time the existing data(images) are modified or additional data(images) are uploaded to the classes, the model is trained again (재학습), and results are once again previewed on the preview panel.

**Additional.** The training process can be visualized in data graphs under Training > Advanced Panel > Under the Hood (학습 > 고급 > 고급 설정)

![[Pasted image 20230827182956.png | 150]]     ![[Pasted image 20230827183238.png | 200]]     ![[Pasted image 20230827183204.png | 200]]

Data regarding the accuracy for each class confusion matrix, as well as the accuracy and loss per epoch are provided to help in understanding how the model is working.
#### Generate and download source files

A variety of options are provided for generating the source files. When using Tensorflow on Python, the model can be constructed into a Keras H5 file or Tensorflow Savedmodel file and the code is created using functions from the Keras library or OpenCV Kera. When using Tensorflow.js, the model can be constructed in JavaScript or in p5.js.
#### Running code on Google Colab

Teachable Machine not only generates the model but also creates source code for running the model. After downloading the source files, the following generated Python code was used to run the model on Google Colab.

```python
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("ratgeber_hund_rasse_portraits_siberian-husky_1_1200x527..jpg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
```

For the code to run properly, `keras_model.h5` and `labels.txt`, as well as one other sample image was imported onto Google Colab.

![[Pasted image 20230828002522.png | 300]]

After running the code cell, the following results were shown.

![[Pasted image 20230828002459.png | 300]]

Then, `display(image)` was used to show image along with results.

![[Pasted image 20230828010722.png | 400]]

Google Colab Link. https://colab.research.google.com/drive/1_jlpEqn2wQ-yvEZqHAhWSnCLuIBvUODS?usp=sharing

**Additional.** For sharing purposes, files were uploaded to GitHub (via git clone) and imported on Google Colab.

```
!git clone https://github.com/ganyunhee/ai_teachable_machine
```
#### Running code on private server

Download Tensorflow.js files. 

Navigate to  `tm-my-image-model` folder. 

Within folder, create `index.html`.

Copy generated JavaScript code from Teachable Machine and paste within the `<body>` tag of `index.html`. 

Create `my_model` folder and move all JSON and bin files (i.e. `metadata.json`, `model.json`, `weights.bin`) to this folder. 

Then, run `index.html` via Live Server (VS Code plugin).


**ISSUE.** Camera restricted. Must allow private network

Google Chrome blocks camera access from Enable private network even after allowing permissions for camera usage.

To solve this, navigate to `chrome://flags/#unsafely-treat-insecure-origin-as-secure` on Google Chrome.

Find and enable `Insecure origins treated as secure`

![[Pasted image 20230828000930.png | 400]]
#### Results

![[Pasted image 20230828005734.png | 200]]     ![[Pasted image 20230828005838.png | 200]]
#### [Additional] Adding descriptions for each breed upon detection

Edit JavaScript code. 

Gather description for each dog breed.

```
// Siberian Husky

The Siberian Husky is a striking and energetic breed of dog known for its captivating blue or multi-colored eyes, distinctive markings, and remarkable endurance. Originating in Siberia, these dogs were bred by the Chukchi people for sledding and hauling heavy loads across long distances in cold climates. Their dense double coat helps them stay warm in freezing temperatures, and their erect triangular ears give them an alert and attentive appearance. Siberian Huskies are known for their friendly and outgoing nature, often forming strong bonds with people and other dogs. They possess an adventurous spirit and require regular exercise to channel their energy, making them suitable for active families who can provide them with the mental and physical stimulation they need.
```

```
// Alaskan Malamute

The Alaskan Malamute is a powerful and majestic breed of dog known for its strong build, endurance, and friendly demeanor. Originating from the Arctic regions, they were initially bred for heavy-duty tasks like hauling sleds and pulling heavy loads in harsh conditions. With a dense double coat that offers protection against extreme cold, their distinctive appearance includes a plumed tail, erect ears, and a robust frame. Alaskan Malamutes are social and affectionate animals that often form strong bonds with their families. They require regular exercise and mental stimulation to stay content, and their gentle nature makes them suitable companions for families that can provide the necessary care and attention they need.
```

Verify code and see that prediction results are shown by the `predict()` function.

```js
        // run the webcam image through the image model
        async function predict() {
            // predict can take in an image, video or canvas html element
            const prediction = await model.predict(webcam.canvas);
            for (let i = 0; i < maxPredictions; i++) {
                const classPrediction =
                    prediction[i].className + ": " + prediction[i].probability.toFixed(2);
                labelContainer.childNodes[i].innerHTML = classPrediction;
            }
        }
```

Add code to show description for each dog breed, depending on prediction. (NOTE. Here, prediction results are continuously displayed for reference.)

```js
if (prediction[0].className == "Siberian Husky" && prediction[0].probability.toFixed(2) == 1.00) {
	labelContainer.childNodes[2].innerHTML = "Husky 설명";
}

else if (prediction[1].className == "Siberian Husky" && prediction[1].probability.toFixed(2) == 1.00) {
	labelContainer.childNodes[2].innerHTML = "Malamute 설명";
}

else {
	labelContainer.childNodes[2].innerHTML = "알 수 없음";
}

```

To show explanation, add a new loop that creates an additional div in the `init()` function.

```js
            // Add an additional loop to create a third child node for explanations
            for (let i = 0; i < 1; i++) {
                labelContainer.appendChild(document.createElement("div"));
            }
```

## Model Behavior Analysis

#### Scenario: Ambiguous Image Classification

One photo of a Siberian Husky was misplaced as an Alaskan Malamute. This is an ambiguous case since the picture of the husky presented majority of the traits of a husky and a malamute. Even to the human eye, the husky can be perceived as a malamute at first glance. However, this model is expected to be more accurate than that. Repeated tweaking of the epochs, batch sizes, learning rates were done and the model was trained at every change until proper results were shown. Soon, the model became more accurate.

**BEFORE**

![[Pasted image 20230827183855.png | 300]]

**AFTER**

![[Pasted image 20230827193011.png | 300]]

#### Scenario: Indecisive or Biased Results

The model seems to provide more accuracy for Siberian huskies and less accuracy with Alaskan Malamutes. It is indecisive between the two whenever an image of an Alaskan Malamute is shown and tends to repeatedly add to the prediction values of the Siberian Husky (i.e. signs of varying numbers and rapidly flickering text - in favor of husky traits).

**Improved Approach I.** Compare class prediction values

With the previous code, if the prediction values were not exactly at 1.00 or 100%, the results vary between the two classes and the model is left indecisive continuously displaying "알 수 없음". To remedy this, prediction values were compared with each other so that when one class had greater prediction values, it's explanation would be displayed.

```js
            if (prediction[0].className == "Siberian Husky" && prediction[0].probability.toFixed(2) > prediction[1].probability.toFixed(2)) {
                labelContainer.childNodes[2].innerHTML = "Husky 설명";
            }

            else if (prediction[1].className == "Alaskan Malamute" && prediction[1].probability.toFixed(2) > prediction[0].probability.toFixed(2)) {
                labelContainer.childNodes[2].innerHTML = "Malamute 설명";
            }

            else {
                labelContainer.childNodes[2].innerHTML = "알 수 없음";
            }
```

**Improved Approach II.** Add a threshold of more than 0.60 

If the prediction value for a class is more than 60%, then the model shows the explanation for that class thereby providing a conclusion even if the values are somewhat indecisive. This ensures more consistency, confidence and interpretability.

```js

        // Add a prediction value threshold
        const PREDICTION_THRESHOLD = 0.6;

        // run the webcam image through the image model
        async function predict() {
            // predict can take in an image, video or canvas html element
            const prediction = await model.predict(webcam.canvas);
            for (let i = 0; i < maxPredictions; i++) {
                const classPrediction =
                    prediction[i].className + ": " + prediction[i].probability.toFixed(2);
                labelContainer.childNodes[i].innerHTML = classPrediction;
            }
            if (prediction[0].className == "Siberian Husky" && prediction[0].probability.toFixed(2) > PREDICTION_THRESHOLD) {
                labelContainer.childNodes[2].innerHTML = "Husky 설명";
            }

            else if (prediction[1].className == "Alaskan Malamute" && prediction[1].probability.toFixed(2) > PREDICTION_THRESHOLD) {
                labelContainer.childNodes[2].innerHTML = "Malamute 설명";
            }

            else {
                labelContainer.childNodes[2].innerHTML = "알 수 없음";
            }
        }
```

The results still vary at first but eventually add up to the proper prediction result i.e. the image is put into the proper class.
## Final Results
---

![[Pasted image 20230828035338.png]]

![[Pasted image 20230828035407.png]]
###### Source. https://github.com/ganyunhee/ai_teachable_machine

## Discussion
---
#### Inherent Ambiguity / Intra-class Variability

The varying results that are evident in the model's indecisiveness towards the Alaskan Malamute is definitely not due to class imbalance (since both classes were given the same data set), and is yet to be checked for algorithm bias.

However, upon more research and a closer look, instead of bias I conclude that the varying results is more caused by "inherent ambiguity" or "intra-class variability" as the model is struggling to distinguish between the two classes due to shared traits, in this case - a similarity in visual traits.

**To Improve.** Tweak values of epoch, batch size, and learning rate. Calculate accuracy for each tweak and verify thru code. Determine if there is algorithm bias. Expand data set and train model for better accuracy. Search ways on how to accomodate for ambiguity.

## References
---
Teachable Model. https://teachablemachine.withgoogle.com/train.
Malamute vs Husky: All 10 Differences Explained. https://www.marvelousdogs.com/malamute-vs-husky/.
Predictive Modeling: See the Future and Make More Profitable Decisions. https://www.appier.com/ko-kr/blog/predictive-modeling-see-the-future-and-make-more-profitable-decisions
How to allow Chrome to access my camera on localhost?. https://stackoverflow.com/questions/16835421/how-to-allow-chrome-to-access-my-camera-on-localhost

Photos acquired from Unsplash, Pexels, and Google Images.