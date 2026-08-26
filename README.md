# MobileNet Image Classifier

A browser-only React app that lets you upload a MobileNet ONNX model and classify images — all inference runs locally in your browser via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).

## Usage

1. **Download the ONNX model** from the [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet) (e.g. `mobilenetv2-7.onnx`).
2. **Run the app** (see below) and click **Choose .onnx file** to load the model.
3. **Upload any image** (JPG / PNG).
4. Click **Classify Image** to see the top-5 ImageNet predictions with confidence scores.

No data leaves your browser.

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```