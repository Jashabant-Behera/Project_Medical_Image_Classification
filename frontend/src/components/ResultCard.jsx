export default function ResultCard({ result }) {
  const isPneumonia = result.prediction === 'PNEUMONIA';
  return (
    <div className={`rounded-xl border-2 p-6 ${isPneumonia
      ? 'border-red-300 bg-red-50' : 'border-green-300 bg-green-50'}`}>
      <div className='flex items-center gap-3 mb-4'>
        <span className='text-3xl'>{isPneumonia ? '⚠️' : '✅'}</span>
        <div>
          <h2 className={`text-2xl font-bold ${isPneumonia ? 'text-red-700':'text-green-700'}`}>
            {result.prediction}
          </h2>
          <p className='text-sm text-gray-500'>Model Version: {result.model_version}</p>
        </div>
      </div>
      <div className='mb-3'>
        <div className='flex justify-between text-sm mb-1'>
          <span className='font-medium'>Confidence</span>
          <span className='font-bold'>{(result.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className='h-3 bg-gray-200 rounded-full overflow-hidden'>
          <div
            className={`h-full rounded-full ${isPneumonia ? 'bg-red-500' : 'bg-green-500'}`}
            style={{ width: `${result.confidence * 100}%` }}
          />
        </div>
      </div>
      <p className='text-xs text-gray-400'>Inference time: {result.inference_ms}ms</p>
    </div>
  );
}
