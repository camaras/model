import { useState, useRef } from 'react';
import {
  Box, Button, Card, CardMedia, Chip,
  CircularProgress, Container, LinearProgress, Paper,
  Step, StepLabel, Stepper, Typography, Alert
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ImageIcon from '@mui/icons-material/Image';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import { loadModel, runInference } from './inference';
import { IMAGENET_CLASSES } from './imageNetLabels';

const STEPS = ['Upload ONNX Model', 'Upload Image', 'Run Inference'];

export default function App() {
  const [activeStep, setActiveStep] = useState(0);
  const [model, setModel] = useState(null);
  const [modelName, setModelName] = useState('');
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState('');

  const [imageEl, setImageEl] = useState(null);
  const [imageSrc, setImageSrc] = useState('');
  const [imageName, setImageName] = useState('');

  const [results, setResults] = useState(null);
  const [inferring, setInferring] = useState(false);
  const [inferError, setInferError] = useState('');

  const modelInputRef = useRef();
  const imageInputRef = useRef();

  async function handleModelUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';
    setModelError('');
    setModelLoading(true);
    try {
      const session = await loadModel(file);
      setModel(session);
      setModelName(file.name);
      setActiveStep(1);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setModelError(`Failed to load model: ${message}`);
    } finally {
      setModelLoading(false);
    }
  }

  function handleImageUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (imageSrc) URL.revokeObjectURL(imageSrc);
    const url = URL.createObjectURL(file);
    setImageSrc(url);
    setImageName(file.name);
    setResults(null);
    setInferError('');
    const img = new Image();
    img.onload = () => {
      setImageEl(img);
      setActiveStep(2);
    };
    img.src = url;
  }

  async function handleInfer() {
    if (!model || !imageEl) return;
    setInferring(true);
    setInferError('');
    setResults(null);
    try {
      const top5 = await runInference(model, imageEl);
      setResults(top5);
    } catch (err) {
      setInferError(`Inference failed: ${err.message}`);
    } finally {
      setInferring(false);
    }
  }

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'grey.100', py: 4 }}>
      <Container maxWidth="md">
        {/* Header */}
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <ModelTrainingIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
          <Typography variant="h4" fontWeight={700}>MobileNet Image Classifier</Typography>
          <Typography variant="subtitle1" color="text.secondary" mt={1}>
            Upload a MobileNet ONNX model and an image to classify it in your browser — no server required.
          </Typography>
        </Box>

        {/* Stepper */}
        <Stepper activeStep={activeStep} alternativeLabel sx={{ mb: 4 }}>
          {STEPS.map(label => (
            <Step key={label}><StepLabel>{label}</StepLabel></Step>
          ))}
        </Stepper>

        {/* Step 1 – Model */}
        <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Step 1 — Upload ONNX Model
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={2}>
            Download the MobileNet v2 ONNX model from the{' '}
            <a href="https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet"
               target="_blank" rel="noreferrer">ONNX model zoo</a>{' '}
            and upload it here.
          </Typography>
          <input
            ref={modelInputRef}
            type="file"
            accept=".onnx"
            style={{ display: 'none' }}
            onChange={handleModelUpload}
          />
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              startIcon={modelLoading ? <CircularProgress size={18} color="inherit" /> : <CloudUploadIcon />}
              onClick={() => modelInputRef.current.click()}
              disabled={modelLoading}
            >
              {modelLoading ? 'Loading…' : 'Choose .onnx file'}
            </Button>
            {modelName && !modelLoading && (
              <Chip color="success" label={modelName} />
            )}
          </Box>
          {modelError && <Alert severity="error" sx={{ mt: 2 }}>{modelError}</Alert>}
        </Paper>

        {/* Step 2 – Image */}
        <Paper elevation={2} sx={{ p: 3, mb: 3, opacity: model ? 1 : 0.5 }}>
          <Typography variant="h6" gutterBottom>
            Step 2 — Upload Image
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={2}>
            Upload any JPG / PNG image. It will be resized to 224×224 for inference.
          </Typography>
          <input
            ref={imageInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleImageUpload}
            disabled={!model}
          />
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              color="secondary"
              startIcon={<ImageIcon />}
              onClick={() => imageInputRef.current.click()}
              disabled={!model}
            >
              Choose image
            </Button>
            {imageName && <Chip color="info" label={imageName} />}
          </Box>

          {imageSrc && (
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
              <Card sx={{ maxWidth: 320 }}>
                <CardMedia
                  component="img"
                  image={imageSrc}
                  alt="uploaded"
                  sx={{ maxHeight: 300, objectFit: 'contain', bgcolor: 'grey.200' }}
                />
              </Card>
            </Box>
          )}
        </Paper>

        {/* Step 3 – Inference */}
        <Paper elevation={2} sx={{ p: 3, mb: 3, opacity: imageEl ? 1 : 0.5 }}>
          <Typography variant="h6" gutterBottom>
            Step 3 — Run Inference
          </Typography>
          <Button
            variant="contained"
            color="success"
            size="large"
            startIcon={inferring ? <CircularProgress size={20} color="inherit" /> : <ModelTrainingIcon />}
            onClick={handleInfer}
            disabled={!model || !imageEl || inferring}
          >
            {inferring ? 'Classifying…' : 'Classify Image'}
          </Button>

          {inferError && <Alert severity="error" sx={{ mt: 2 }}>{inferError}</Alert>}

          {results && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                Top-5 Predictions
              </Typography>
              {results.map(({ index, prob }) => {
                const label = IMAGENET_CLASSES[index] ?? `class ${index}`;
                const pct = (prob * 100).toFixed(1);
                return (
                  <Box key={index} sx={{ mb: 1.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2" noWrap sx={{ maxWidth: '70%' }}>{label}</Typography>
                      <Typography variant="body2" fontWeight={600}>{pct}%</Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={prob * 100}
                      sx={{ height: 8, borderRadius: 4 }}
                      color={prob > 0.5 ? 'success' : prob > 0.1 ? 'primary' : 'inherit'}
                    />
                  </Box>
                );
              })}
            </Box>
          )}
        </Paper>

        <Typography variant="caption" color="text.secondary" textAlign="center" display="block">
          All processing happens locally in your browser. No data is sent to any server.
        </Typography>
      </Container>
    </Box>
  );
}
