const express = require('express');
const path = require('path');

const app = express();

function coi(req, res, next) {
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  next();
}

app.use(coi);

app.use('/', express.static(path.resolve(__dirname)));

module.exports = app.listen(3000, () => {
  console.log('Server is running on port 3000');
});