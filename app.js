// app.js (Node.js Frontend)
const express = require('express');
const path = require('path');
const axios = require('axios');

const app = express();

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Home page
app.get('/', (req, res) => {
  res.render('home'); // views/home.ejs
});

// Form page
app.get('/form', (req, res) => {
  res.render('index', { result: null, error: null, formData: {} });
});

// Handle form submission
app.post('/predict', async (req, res) => {
  try {
    const response = await axios.post("http://127.0.0.1:8000/predict", req.body);
    const predicted = response.data.predicted_anomaly;

    res.render('index', {
      result: predicted,
      error: null,
      formData: req.body
    });
  } catch (err) {
    console.error("Error:", err.message);
    res.render('index', {
      result: null,
      error: "Error connecting to Flask backend or invalid input.",
      formData: req.body
    });
  }
});

app.listen(3000, () => {
  console.log("Frontend running at http://localhost:3000");
});
