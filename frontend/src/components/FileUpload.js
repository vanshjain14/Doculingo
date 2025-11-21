import React, { useState } from 'react';
import './FileUpload.css';

function FileUpload({ onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('idle');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }
    setUploadStatus('uploading');
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('File upload failed');
      await response.json();
      setUploadStatus('success');
      onUploadSuccess();
    } catch (error) {
      console.error("Error uploading file:", error);
      setUploadStatus('error');
    }
  };

  return (
    <div className="upload-section">
      <h2>Upload Your Document</h2>
      <p>Upload a PDF to begin a conversation.</p>
      <div className="upload-box">
          <input type="file" id="file-upload" onChange={handleFileChange} accept=".pdf" />
          <label htmlFor="file-upload" className="upload-label">
              {file ? file.name : "Choose a file..."}
          </label>
          <button onClick={handleUpload} disabled={uploadStatus === 'uploading' || !file}>
            {uploadStatus === 'uploading' ? 'Processing...' : 'Upload & Start'}
          </button>
      </div>
      {uploadStatus === 'error' && <p className="error-message">Error uploading file. Please try again.</p>}
    </div>
  );
}

export default FileUpload;