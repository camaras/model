import * as ort from 'onnxruntime-web';

// MobileNet expects 224x224 RGB, normalized with ImageNet mean/std
const INPUT_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

export async function loadModel(file) {
  const buffer = await file.arrayBuffer();
  const session = await ort.InferenceSession.create(buffer);
  return session;
}

function preprocessImage(imageElement) {
  const canvas = document.createElement('canvas');
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Failed to acquire 2D canvas context for preprocessing.');
  ctx.drawImage(imageElement, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  // NCHW float32 tensor: [1, 3, 224, 224]
  const tensorData = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
    const r = data[i * 4]     / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    tensorData[i]                             = (r - MEAN[0]) / STD[0];
    tensorData[INPUT_SIZE * INPUT_SIZE + i]   = (g - MEAN[1]) / STD[1];
    tensorData[2 * INPUT_SIZE * INPUT_SIZE + i] = (b - MEAN[2]) / STD[2];
  }
  return new ort.Tensor('float32', tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function softmax(logits) {
  const max = logits.reduce((a, b) => (a > b ? a : b));
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

export async function runInference(session, imageElement) {
  const inputTensor = preprocessImage(imageElement);
  const inputName = session.inputNames[0];
  const feeds = { [inputName]: inputTensor };
  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const logits = Array.from(results[outputName].data);
  const probs = softmax(logits);
  // Return top-5
  const indexed = probs.map((p, i) => ({ index: i, prob: p }));
  indexed.sort((a, b) => b.prob - a.prob);
  return indexed.slice(0, 5);
}
