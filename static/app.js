(function () {
  var input = document.getElementById('photo');
  var preview = document.getElementById('preview');
  if (input && preview) {
    input.addEventListener('change', function () {
      var f = input.files[0];
      if (!f) { preview.hidden = true; return; }
      preview.src = URL.createObjectURL(f);
      preview.hidden = false;
    });
  }
  var btn = document.getElementById('use-location');
  if (btn && navigator.geolocation) {
    btn.hidden = false;
    btn.addEventListener('click', function () {
      btn.textContent = 'Finding location...';
      navigator.geolocation.getCurrentPosition(function (p) {
        document.getElementById('lat').value = p.coords.latitude.toFixed(2);
        document.getElementById('lon').value = p.coords.longitude.toFixed(2);
        btn.textContent = 'Location added';
      }, function () { btn.textContent = 'Could not get location'; });
    });
  }
  var form = document.getElementById('check-form');
  if (form) {
    form.addEventListener('submit', function () {
      var b = form.querySelector('button[type=submit]');
      b.disabled = true; b.textContent = 'Checking...';
    });
  }
})();
