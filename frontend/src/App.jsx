import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import Home from './pages/Home';
 
export default function App() {
  return (
    <BrowserRouter>
      <nav className='bg-white shadow-sm px-6 py-3 flex gap-6'>
        <Link to='/' className='font-semibold text-blue-600'>🫁 X-Ray Classifier</Link>
      </nav>
      <Routes>
        <Route path='/' element={<Home />} />
      </Routes>
    </BrowserRouter>
  );
}
