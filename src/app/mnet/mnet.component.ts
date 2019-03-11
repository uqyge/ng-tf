import { Component, OnInit } from "@angular/core";
import { TNodeFlags } from "@angular/core/src/render3/interfaces/node";

import * as tf from "@tensorflow/tfjs";
import { loadGraphModel } from "@tensorflow/tfjs";
const IMAGE_SIZE = 224;
@Component({
  selector: "app-mnet",
  templateUrl: "./mnet.component.html",
  styleUrls: ["./mnet.component.scss"]
})
export class MnetComponent implements OnInit {
  constructor() {}
  mnet: tf.LayersModel;
  predictions: any;
  m7 = document.getElementById("mnist_6") as HTMLImageElement;
  ngOnInit() {
    this.loadModel();
    // document.body.appendChild(this.m7);
    // this.predict(this.m7);
    // console.log(this.m7);
  }
  // this.predict(this.m7);
  //// LOAD PRETRAINED KERAS MODEL ////

  async loadModel() {
    this.mnet = await tf.loadLayersModel(
      "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
    );
    console.log("model loaded");
    console.log(this.m7);
  }

  async predict(imgElement) {
    const logits = tf.tidy(() => {
      // tf.browser.fromPixels() returns a Tensor from an image element.
      const img = tf.browser.fromPixels(imgElement).toFloat();

      const offset = tf.scalar(127.5);
      // Normalize the image from [0, 255] to [-1, 1].
      const normalized = img.sub(offset).div(offset);

      // Reshape to a single-element batch so we can pass it to predict.
      const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

      // Make a prediction through mobilenet.
      return this.mnet.predict(batched);
    });
    console.log(logits);
  }
}
