import { useState } from 'react';
import ImageUploader from '../components/ImageUploader';
import ResultCard from '../components/ResultCard';
import GradCamViewer from '../components/GradCamViewer';
import { predictImage } from '../api/axiosClient';
 
export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
 
  const handleUpload = async (file) => {
    setLoading(true); setError(null); setResult(null);
    setPreview(URL.createObjectURL(file));
    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };
 
  return (
    <div className='min-h-screen bg-gray-50 py-10'>
      <div className='max-w-2xl mx-auto px-4'>
        <h1 className='text-3xl font-bold text-gray-900 text-center mb-2'>
          Chest X-Ray Analyzer
        </h1>
        <p className='text-center text-gray-500 mb-8'>AI-powered pneumonia detection</p>
        <ImageUploader onUpload={handleUpload} loading={loading} />
        {error && <div className='mt-4 p-4 bg-red-50 border border-red-200 rounded-lg
          text-red-700 text-sm'>{error}</div>}
        {result && (
          <div className='mt-6'>
            <ResultCard result={result} />
            <GradCamViewer originalPreview={preview} gradcamImage={result.gradcam_image} />
          </div>
        )}
      </div>
    </div>
  );
}
