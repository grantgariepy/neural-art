// neural network random art generator

// settings for networkSize, from html slider1

var slider1 = document.getElementById("networkSizeSlider");
var output1 = document.getElementById("output1");
var networkSize;
let update1 = () => (
  (output1.innerHTML = slider1.value), (networkSize = slider1.value)
);

slider1.addEventListener("input", update1);
update1();

// settings for networkSize, from html slider2

var slider2 = document.getElementById("nHiddenSlider");
var output2 = document.getElementById("output2");
var nHidden;
let update2 = () => (
  (output2.innerHTML = slider2.value), (nHidden = slider2.value)
);

slider2.addEventListener("input", update2);
update2();

// actual size of generated image
var sizeh = 32 * 5;
var sizew = sizeh;
var sizeImage = sizeh * sizew;

var nH, nW, nImage;
var mask;

// settings of nnet:
var nOut = 3; // r, g, b layers

// support variables:
var img;
var img2;
var G = new R.Graph(false);

var initModel = function() {
  "use strict";

  var model = [];
  var i;

  var randomSize = 1.0;

  // define the model below:
  model.w_in = R.RandMat(networkSize, 3, 0, randomSize); // x, y, and bias

  for (i = 0; i < nHidden; i++) {
    model["w_" + i] = R.RandMat(networkSize, networkSize, 0, randomSize);
  }

  model.w_out = R.RandMat(nOut, networkSize, 0, randomSize); // output layer

  return model;
};

var forwardNetwork = function(G, model, x_, y_) {
  // x_, y_ is a normal javascript float, will be converted to a mat object below
  // G is a graph to amend ops to
  var x = new R.Mat(3, 1); // input
  var i;
  x.set(0, 0, x_);
  x.set(1, 0, y_);
  x.set(2, 0, 1.0); // bias.
  var out;
  out = G.tanh(G.mul(model.w_in, x));
  for (i = 0; i < nHidden; i++) {
    out = G.tanh(G.mul(model["w_" + i], out));
  }
  out = G.sigmoid(G.mul(model.w_out, out));
  return out;
};

function getColorAt(model, x, y) {
  // function that returns a color given coordintes (x, y)
  // (x, y) are scaled to -0.5 -> 0.5 for image recognition later
  // but it can be behond the +/- 0.5 for generation above and beyond
  // recognition limits
  var r, g, b;
  var out = forwardNetwork(G, model, x, y);

  r = out.w[0] * 255.0;
  g = out.w[1] * 255.0;
  b = out.w[2] * 255.0;

  return color(r, g, b);
}

function genImage(img, model) {
  var i, j, m, n;
  img.loadPixels();
  for (i = 0, m = img.width; i < m; i++) {
    for (j = 0, n = img.height; j < n; j++) {
      img.set(i, j, getColorAt(model, i / sizeh - 0.5, j / sizew - 0.5));
    }
  }
  img.updatePixels();
}

function setup() {
  "use strict";
  var myCanvas;

  myCanvas = createCanvas(500, 500);

  myCanvas.parent("p5Container");

  nW = Math.max(Math.floor(windowWidth / sizew), 1);
  nH = Math.max(Math.floor(windowHeight / sizeh), 1);
  nImage = nH * nW;
  mask = R.zeros(nImage);

  //img.save('genart.png','png');

  noLoop();
  img = createImage(500, 500);
  // img.resize(1000*1.0, 1000*1.0);
  frameRate(30);
}

// function getRandomLocation() {
//   var i,
//     result = 0,
//     r;
//   for (i = 0; i < nImage; i++) {
//     result += mask[i];
//   }
//   if (result === nImage) {
//     mask = R.zeros(nImage);
//   }
//   do {
//     r = R.randi(0, nImage);
//   } while (mask[r] !== 0);
//   mask[r] = 1;
//   return r;
// }

function displayImage(n) {
  var row = Math.floor(n / nW);
  var col = n % nW;
  image(img, col * sizew, row * sizeh);
}

document.getElementById("generate").onclick = function draw() {
  model = initModel();
  genImage(img, model);
  displayImage();
  console.log(output1.value, output2.value);
};
