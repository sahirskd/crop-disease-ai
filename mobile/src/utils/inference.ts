import * as ort from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';
import * as jpeg from 'jpeg-js';
import { Buffer } from 'buffer';

// Polyfill for Buffer needed by jpeg-js in React Native
if (typeof (globalThis as any).Buffer === 'undefined') {
  (globalThis as any).Buffer = Buffer;
}

const MODEL_URI = require('../../assets/cropsense_fp32_v2.onnx');
const WEIGHTS_URI = require('../../assets/classifier_weights.json');
const CLASSES_URI = require('../../assets/class_names.json');

let session: ort.InferenceSession | null = null;
let classifierWeights: number[][] | null = null;
let classNames: string[] | null = null;

export async function initModel() {
  if (session) return;

  // Load model
  const modelAsset = Asset.fromModule(MODEL_URI);
  await modelAsset.downloadAsync();
  session = await ort.InferenceSession.create(
    (modelAsset.localUri || modelAsset.uri).replace('file://', '')
  );

  // Load weights (React Native parses required JSON directly)
  classifierWeights = WEIGHTS_URI;

  // Load class names
  classNames = CLASSES_URI;
}

// Preprocess image to Float32Array [1, 3, 300, 300]
async function preprocessImage(imageUri: string): Promise<Float32Array> {
  const base64 = await FileSystem.readAsStringAsync(imageUri, { encoding: FileSystem.EncodingType.Base64 });
  const rawImageData = Buffer.from(base64, 'base64');

  const decodedImage = jpeg.decode(rawImageData, { useTArray: true });

  // decodedImage.data is an array of [r,g,b,a, r,g,b,a...]
  // We need to extract RGB, normalize to [0,1], then standard normalization
  // Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]

  const width = decodedImage.width;
  const height = decodedImage.height;

  // The image must be exactly 300x300 for our model
  if (width !== 300 || height !== 300) {
    throw new Error(`Expected image to be 300x300, got ${width}x${height}. Please use expo-image-manipulator first.`);
  }

  const numPixels = width * height;
  const tensorData = new Float32Array(3 * numPixels);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < numPixels; i++) {
    const r = decodedImage.data[i * 4 + 0] / 255.0;
    const g = decodedImage.data[i * 4 + 1] / 255.0;
    const b = decodedImage.data[i * 4 + 2] / 255.0;

    tensorData[i] = (r - mean[0]) / std[0]; // R channel
    tensorData[numPixels + i] = (g - mean[1]) / std[1]; // G channel
    tensorData[2 * numPixels + i] = (b - mean[2]) / std[2]; // B channel
  }

  return tensorData;
}

export interface InferenceResult {
  className: string;
  confidence: number;
  heatmap: number[][]; // 10x10 normalized CAM
}

export async function runInference(imageUri: string): Promise<InferenceResult> {
  if (!session || !classifierWeights || !classNames) {
    await initModel();
  }

  // Preprocess
  const inputTensorData = await preprocessImage(imageUri);

  const tensor = new ort.Tensor('float32', inputTensorData, [1, 3, 300, 300]);

  // Run
  const inputName = session!.inputNames[0];
  const results = await session!.run({ [inputName]: tensor });

  const probsTensor = results.probabilities;
  const fmapsTensor = results.fmaps;

  const probs = probsTensor.data as Float32Array;
  const fmaps = fmapsTensor.data as Float32Array;

  // Find top class
  let maxProb = 0;
  let maxIdx = 0;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > maxProb) {
      maxProb = probs[i];
      maxIdx = i;
    }
  }

  // Calculate CAM
  const predWeights = classifierWeights![maxIdx];
  const H = 10;
  const W = 10;
  const CHANNELS = 1536;

  let heatmap = Array.from({ length: H }, () => Array(W).fill(0));
  let minCam = Infinity;
  let maxCam = -Infinity;

  // fmaps is shape [1, 1536, 10, 10]
  // stored sequentially: Channel 0 (100 values), Channel 1 (100 values), etc.
  for (let h = 0; h < H; h++) {
    for (let w = 0; w < W; w++) {
      let sum = 0;
      for (let c = 0; c < CHANNELS; c++) {
        // Index in flattened fmaps tensor
        const val = fmaps[c * H * W + h * W + w];
        sum += val * predWeights[c];
      }

      // ReLU
      sum = Math.max(0, sum);
      heatmap[h][w] = sum;

      if (sum < minCam) minCam = sum;
      if (sum > maxCam) maxCam = sum;
    }
  }

  // Normalize CAM
  for (let h = 0; h < H; h++) {
    for (let w = 0; w < W; w++) {
      heatmap[h][w] = (heatmap[h][w] - minCam) / (maxCam - minCam + 1e-8);
    }
  }

  return {
    className: classNames![maxIdx],
    confidence: maxProb,
    heatmap
  };
}
