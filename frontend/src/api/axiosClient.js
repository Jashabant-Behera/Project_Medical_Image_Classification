import axios from 'axios';
 
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
 
const client = axios.create({ baseURL: BASE_URL });
 
export const predictImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await client.post('/api/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};
 
export const fetchHistory = async (limit = 20) => {
  const response = await client.get(`/api/predictions?limit=${limit}`);
  return response.data;
};
