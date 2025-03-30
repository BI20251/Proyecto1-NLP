// src/App.js
import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [predicciones, setPredicciones] = useState([]);
  const [nombreArchivo, setNombreArchivo] = useState('');
  const [cargando, setCargando] = useState(false);
  const [metricas, setMetricas] = useState(null);


  const handleFileChange = (e) => {
    const archivo = e.target.files[0];
    setFile(archivo);
    setNombreArchivo(archivo.name);
  };

  const handleUpload = async () => {
    if (!file) return alert('Por favor selecciona un archivo CSV');
  
    const formData = new FormData();
    formData.append('file', file);
  
    try {
      setCargando(true);
  
      // 📤 Subir archivo
      await axios.post('http://localhost:8000/upload', formData);
  
      // 📥 Obtener predicciones
      const response = await axios.get(`http://localhost:8000/predecir/${nombreArchivo}`);
  
      // 🔍 Verifica si predicciones es un string (como en tu caso actual)
      let data = response.data.predicciones;
  
      if (typeof data === 'string') {
        data = JSON.parse(data);
      }
  
      const resultados = data.map((item) => ({
        prediccion: item.c === "1" ? "FAKE" : "REAL",
        probabilidad: parseFloat(item.p).toFixed(4),
      }));
  
      setPredicciones(resultados);
      setCargando(false);
  
    } catch (error) {
      console.error('Error al procesar:', error);
  
      if (error.response) {
        console.error('Respuesta del backend:', error.response.data);
        alert("Error del servidor: " + JSON.stringify(error.response.data));
      } else {
        alert("Error al conectar con la API. Revisa la consola.");
      }
  
      setCargando(false);
    }
  };
  
  const handleReentrenar = async () => {
    if (!file) return alert('Por favor selecciona un archivo CSV para reentrenar');
  
    const formData = new FormData();
    formData.append('file', file);
  
    try {
      setCargando(true);
  
      // 📤 Subir archivo al backend
      await axios.post('http://localhost:8000/upload', formData);
  
      // 📥 Llamar al endpoint de reentrenamiento
      const response = await axios.get(`http://localhost:8000/reentrenar/${nombreArchivo}`);
      setMetricas(response.data);
      setCargando(false);
  
    } catch (error) {
      console.error('Error al reentrenar:', error);
      alert('Hubo un error al reentrenar el modelo. Revisa la consola.');
      setCargando(false);
    }
  };
  
  return (
    <div style={{ maxWidth: 600, margin: "0 auto", padding: 40, fontFamily: "Arial, sans-serif" }}>
      <h2>📰 Clasificador de Noticias Falsas</h2>

      <input type="file" accept=".csv" onChange={handleFileChange} />
      <br />
      <button
        onClick={handleUpload}
        style={{
          marginTop: 10,
          padding: "8px 16px",
          backgroundColor: "#007bff",
          color: "white",
          border: "none",
          borderRadius: 4,
          cursor: "pointer"
        }}
      >
        Enviar y Predecir
      </button>
      <button
          onClick={handleReentrenar}
          style={{
            marginTop: 10,
            padding: "8px 16px",
            backgroundColor: "#28a745",
            color: "white",
            border: "none",
            borderRadius: 4,
            cursor: "pointer"
          }}
        >
          Reentrenar Modelo
        </button>

      {cargando && <p style={{ marginTop: 20 }}>⏳ Cargando predicciones...</p>}

      {predicciones.length > 0 && (
        <div style={{ marginTop: 30 }}>
          <h3>📊 Resultados:</h3>
          <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 10 }}>
            <thead>
              <tr style={{ backgroundColor: "#f0f0f0" }}>
                <th style={{ padding: 8, border: "1px solid #ccc" }}>#</th>
                <th style={{ padding: 8, border: "1px solid #ccc" }}>Predicción</th>
                <th style={{ padding: 8, border: "1px solid #ccc" }}>Probabilidad</th>
              </tr>
            </thead>
            <tbody>
              {predicciones.map((p, idx) => (
                <tr key={idx}>
                  <td style={{ padding: 8, border: "1px solid #ccc", textAlign: "center" }}>{idx + 1}</td>
                  <td style={{ padding: 8, border: "1px solid #ccc" }}>{p.prediccion}</td>
                  <td style={{ padding: 8, border: "1px solid #ccc" }}>{p.probabilidad}</td>
                </tr>
              ))}
              {metricas && (
                <div style={{ marginTop: 30 }}>
                  <h3>📈 Métricas del Reentrenamiento:</h3>
                  <ul>
                    <li><b>Accuracy:</b> {metricas["Acuracy"].toFixed(4)}</li>
                    <li><b>Precision:</b> {metricas["Precision"].toFixed(4)}</li>
                    <li><b>Recall:</b> {metricas["Recall"].toFixed(4)}</li>
                    <li><b>F1 Score:</b> {metricas["F1 Score"].toFixed(4)}</li>
                  </ul>
                </div>
              )}

            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;
