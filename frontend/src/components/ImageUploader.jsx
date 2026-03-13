import { useDropzone } from 'react-dropzone';
 
export default function ImageUploader({ onUpload, loading }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'image/jpeg': ['.jpg', '.jpeg'], 'image/png': ['.png'] },
    maxSize: 16 * 1024 * 1024,
    onDrop: (acceptedFiles) => { if (acceptedFiles[0]) onUpload(acceptedFiles[0]); },
  });
  return (
    <div {...getRootProps()} className={`border-2 border-dashed rounded-xl p-12 text-center
      cursor-pointer transition-colors
      ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'}`}>
      <input {...getInputProps()} />
      <div className='text-4xl mb-4'>🫁</div>
      {loading ? (
        <p className='text-blue-600 font-semibold'>Analyzing X-Ray...</p>
      ) : (
        <>
          <p className='text-lg font-semibold text-gray-700'>
            {isDragActive ? 'Drop X-Ray here' : 'Drag & Drop Chest X-Ray'}
          </p>
          <p className='text-sm text-gray-400 mt-2'>JPEG or PNG, max 16MB</p>
          <button className='mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg
            hover:bg-blue-700 transition-colors'>Browse File</button>
        </>
      )}
    </div>
  );
}
