export default function GradCamViewer({ originalPreview, gradcamImage }) {
  return (
    <div className='mt-6'>
      <h3 className='font-semibold text-gray-700 mb-3'>Grad-CAM Explainability</h3>
      <div className='grid grid-cols-2 gap-4'>
        <div className='text-center'>
          <p className='text-xs text-gray-500 mb-2 font-medium'>Original X-Ray</p>
          <img src={originalPreview} alt='Original'
            className='w-full rounded-lg border border-gray-200' />
        </div>
        <div className='text-center'>
          <p className='text-xs text-gray-500 mb-2 font-medium'>Grad-CAM Heatmap</p>
          <img src={gradcamImage} alt='Grad-CAM'
            className='w-full rounded-lg border border-gray-200' />
        </div>
      </div>
      <p className='text-xs text-gray-400 mt-2'>
        Red/warm regions indicate areas most influential in the prediction.
      </p>
    </div>
  );
}
