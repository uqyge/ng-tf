import { Component, OnInit, Input } from "@angular/core";
import { TNodeFlags } from "@angular/core/src/render3/interfaces/node";

import * as tf from "@tensorflow/tfjs";
import { ControllerDataset } from "./controller_dataset";
import { WebcamImage } from "../modules/webcam/domain/webcam-image";
import { Webcam } from "./webcam";
// import * as ui from "./ui";

const IMAGE_SIZE = 224;
const NUM_CLASSES = 2;

const controllerDataset = new ControllerDataset(NUM_CLASSES);
@Component({
  selector: "app-mnet",
  templateUrl: "./mnet.component.html",
  styleUrls: ["./mnet.component.scss"]
})
export class MnetComponent implements OnInit {
  @Input() imgW: WebcamImage;
  constructor() {}
  truncatedMnet: tf.LayersModel;
  model: tf.LayersModel;
  showCat: boolean = false;
  face1Img = document.getElementById("face1") as HTMLImageElement;
  imgF1 = tf.browser
    .fromPixels(this.face1Img)
    .toFloat()
    .expandDims(0);

  face2Img = document.getElementById("face2") as HTMLImageElement;
  imgF2 = tf.browser
    .fromPixels(this.face2Img)
    .toFloat()
    .expandDims(0);

  stopImg = document.getElementById("stop1") as HTMLImageElement;
  imgS1 = tf.browser
    .fromPixels(this.stopImg)
    .toFloat()
    .expandDims(0);

  stop2Img = document.getElementById("stop2") as HTMLImageElement;
  imgS2 = tf.browser
    .fromPixels(this.stop2Img)
    .toFloat()
    .expandDims(0);

  predictions: any;
  ngOnInit() {
    this.loadTruncatedMobileNet();
    // console.log(this.m7);
  }

  async loadTruncatedMobileNet() {
    const mobilenet = await tf.loadLayersModel(
      "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
    );

    const layer = mobilenet.getLayer("conv_pw_13_relu");
    this.truncatedMnet = tf.model({
      inputs: mobilenet.inputs,
      outputs: layer.output
    });
    console.log("model loaded");
  }

  async train() {
    tf.tidy(() => {
      controllerDataset.addExample(this.truncatedMnet.predict(this.imgF1), 0);
      controllerDataset.addExample(this.truncatedMnet.predict(this.imgF2), 0);
      controllerDataset.addExample(this.truncatedMnet.predict(this.imgS1), 1);
      controllerDataset.addExample(this.truncatedMnet.predict(this.imgS2), 1);
    });
    if (controllerDataset.xs == null) {
      throw new Error("Add some examples before training!");
    }
    this.imgW;
    console.log(this.imgW);
    // Creates a 2-layer fully connected model. By creating a separate model,
    // rather than adding layers to the mobilenet model, we "freeze" the weights
    // of the mobilenet model, and only train weights from the new model.
    this.model = tf.sequential({
      layers: [
        // Flattens the input to a vector so we can use it in a dense layer. While
        // technically a layer, this only performs a reshape (and has no training
        // parameters).
        tf.layers.flatten({
          inputShape: this.truncatedMnet.outputs[0].shape.slice(1)
        }),
        // Layer 1.
        tf.layers.dense({
          units: 100,
          activation: "relu",
          kernelInitializer: "varianceScaling",
          useBias: true
        }),
        // Layer 2. The number of units of the last layer should correspond
        // to the number of classes we want to predict.
        tf.layers.dense({
          units: NUM_CLASSES,
          kernelInitializer: "varianceScaling",
          useBias: false,
          activation: "softmax"
        })
      ]
    });

    // Creates the optimizers which drives training of the model.
    const optimizer = tf.train.adam(1e-5);
    // We use categoricalCrossentropy which is the loss function we use for
    // categorical classification which measures the error between our predicted
    // probability distribution over classes (probability that an input is of each
    // class), versus the label (100% probability in the true class)>
    this.model.compile({
      optimizer: optimizer,
      loss: "categoricalCrossentropy"
    });

    // We parameterize batch size as a fraction of the entire dataset because the
    // number of examples that are collected depends on how many examples the user
    // collects. This allows us to have a flexible batch size.
    const batchSize = Math.floor(controllerDataset.xs.shape[0] * 1);
    if (!(batchSize > 0)) {
      throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`
      );
    }

    // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
    this.model.fit(controllerDataset.xs, controllerDataset.ys, {
      batchSize,
      epochs: 200
    });
  }

  async predict() {
    while (true) {
      // {
      const predictedClass = tf.tidy(() => {
        // Capture the frame from the webcam.
        // Make a prediction through mobilenet, getting the internal activation of
        // the mobilenet model, i.e., "embeddings" of the input images.
        // var img = document.createElement("img");
        // img.src = this.imgW.imageAsDataUrl;

        var img = document.getElementById("cam") as HTMLImageElement;
        console.log("input", img);
        var imgData = tf.browser.fromPixels(img).expandDims(0);
        const embeddings = this.truncatedMnet.predict(imgData);

        // const embeddings = this.truncatedMnet.predict(this.imgD);

        // Make a prediction through our newly-trained model using the embeddings
        // from mobilenet as input.
        const predictions = this.model.predict(embeddings) as tf.Tensor;
        // Returns the index with the maximum probability. This number corresponds
        // to the class the model thinks is the most probable given the input.
        return predictions.as1D().argMax();
      });

      const classId = (await predictedClass.data())[0];
      predictedClass.dispose();
      console.log(classId);
      if (classId > 0) {
        this.showCat = true;
      } else if (classId == 0) {
        this.showCat = false;
      }

      // ui.predictClass(classId);
      // await tf.nextFrame();
    }
  }
}
