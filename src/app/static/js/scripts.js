// static/js/script.js
document.addEventListener('submit', async (evt) => {
  evt.preventDefault();
  let url, resultP;

  if (evt.target.id === 'crop-form') {
    url = '/predict/crop';
    resultP = document.getElementById('crop-result');
    const data = Object.fromEntries(new FormData(evt.target).entries());
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify(data)
    });
    const j = await resp.json();
    resultP.textContent = j.recommended_crop;
  }

  document.addEventListener('submit', async evt => {
  evt.preventDefault();
  let url, resultP;

  document.addEventListener('submit', async evt => {
  evt.preventDefault();
  let url, resultP;

  if (evt.target.id === 'fert-form') {
    url = '/predict/fertilizer';
    resultP = document.getElementById('fert-result');
    const data = Object.fromEntries(new FormData(evt.target).entries());
    data.crop = document.getElementById('fert-crop').value;  // if needed
    const resp = await fetch(url, {
      method: 'POST',
      headers:{ 'Content-Type':'application/json' },
      body: JSON.stringify(data)
    });
    const j = await resp.json();
    resultP.textContent = j.recommended_fertilizer || j.error;
  }
  
});

  // … handle crop & disease forms similarly …
  if (evt.target.id === 'disease-form') {
    url = '/predict/disease';
    resultP = document.getElementById('disease-result');
    const fd = new FormData(evt.target);
    const resp = await fetch(url, { method:'POST', body: fd });
    const j = await resp.json();
    resultP.textContent = `${j.disease} (conf ${j.confidence.toFixed(2)})`;
  }
});


})