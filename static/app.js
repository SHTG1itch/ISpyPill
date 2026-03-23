/* =========================================================
   ISpyPill — Frontend Application Logic
   Vanilla JS, no frameworks
   ========================================================= */

(function () {
  'use strict';

  // --- State ---
  let refFile = null;
  let groupFile = null;

  // --- DOM refs ---
  const refDropZone    = document.getElementById('ref-drop-zone');
  const groupDropZone  = document.getElementById('group-drop-zone');
  const refDropContent = document.getElementById('ref-drop-content');
  const groupDropContent = document.getElementById('group-drop-content');
  const refPreview     = document.getElementById('ref-preview');
  const groupPreview   = document.getElementById('group-preview');
  const refGallery     = document.getElementById('ref-file-gallery');
  const refCamera      = document.getElementById('ref-file-camera');
  const groupGallery   = document.getElementById('group-file-gallery');
  const groupCamera    = document.getElementById('group-file-camera');
  const countBtn       = document.getElementById('count-btn');
  const btnLabel       = countBtn.querySelector('.btn-label');
  const btnSpinner     = countBtn.querySelector('.btn-spinner');
  const errorBanner    = document.getElementById('error-banner');
  const errorMessage   = document.getElementById('error-message');
  const resultsSection = document.getElementById('results-section');
  const countNumber    = document.getElementById('count-number');
  const resultsMeta    = document.getElementById('results-meta');
  const annotatedImg   = document.getElementById('annotated-img');
  const downloadLink   = document.getElementById('download-link');

  // =========================================================
  // File Handling
  // =========================================================

  function handleFile(file, type) {
    if (!file || !file.type.startsWith('image/')) {
      showError('Please upload a valid image file (PNG, JPG, HEIC, etc.).');
      return;
    }
    if (file.size > 16 * 1024 * 1024) {
      showError('Image is too large. Please use an image under 16 MB.');
      return;
    }

    if (type === 'ref') {
      refFile = file;
      showPreview(file, refPreview, refDropContent, refDropZone);
    } else {
      groupFile = file;
      showPreview(file, groupPreview, groupDropContent, groupDropZone);
    }

    updateCountButton();
    hideError();
  }

  function showPreview(file, imgEl, contentEl, dropZone) {
    const reader = new FileReader();
    reader.onload = function (e) {
      imgEl.src = e.target.result;
      imgEl.classList.remove('hidden');
      contentEl.classList.add('hidden');
      dropZone.classList.add('has-image');
    };
    reader.readAsDataURL(file);
  }

  function updateCountButton() {
    countBtn.disabled = !(refFile && groupFile);
  }

  // =========================================================
  // File Input Events
  // =========================================================

  function bindFileInput(inputEl, type) {
    inputEl.addEventListener('change', function () {
      if (this.files && this.files[0]) {
        handleFile(this.files[0], type);
      }
      // Reset so same file can be re-selected
      this.value = '';
    });
  }

  bindFileInput(refGallery, 'ref');
  bindFileInput(refCamera,  'ref');
  bindFileInput(groupGallery, 'group');
  bindFileInput(groupCamera,  'group');

  // =========================================================
  // Drop Zone Click — delegate to gallery input
  // =========================================================

  function bindDropZoneClick(dropZone, galleryInput) {
    dropZone.addEventListener('click', function (e) {
      // Don't re-trigger if click originated from a button/label inside
      if (e.target.closest('.upload-actions')) return;
      galleryInput.click();
    });
    dropZone.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        galleryInput.click();
      }
    });
  }

  bindDropZoneClick(refDropZone,   refGallery);
  bindDropZoneClick(groupDropZone, groupGallery);

  // =========================================================
  // Drag & Drop
  // =========================================================

  function bindDragDrop(dropZone, type) {
    dropZone.addEventListener('dragover', function (e) {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', function (e) {
      if (!dropZone.contains(e.relatedTarget)) {
        dropZone.classList.remove('drag-over');
      }
    });
    dropZone.addEventListener('drop', function (e) {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) handleFile(file, type);
    });
  }

  bindDragDrop(refDropZone,   'ref');
  bindDragDrop(groupDropZone, 'group');

  // =========================================================
  // Form Submission / Analysis
  // =========================================================

  countBtn.addEventListener('click', async function () {
    if (!refFile || !groupFile) return;

    setLoading(true);
    hideError();
    hideResults();

    const formData = new FormData();
    formData.append('reference_pill', refFile);
    formData.append('group_photo', groupFile);

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        showError(data.error || 'An unknown error occurred. Please try again.');
        return;
      }

      showResults(data);
    } catch (err) {
      if (err.name === 'TypeError') {
        showError('Network error — please check your connection and try again.');
      } else {
        showError('An unexpected error occurred: ' + err.message);
      }
    } finally {
      setLoading(false);
    }
  });

  // =========================================================
  // UI State Helpers
  // =========================================================

  function setLoading(loading) {
    countBtn.disabled = loading;
    if (loading) {
      btnLabel.textContent = 'Analyzing…';
      btnSpinner.classList.remove('hidden');
    } else {
      btnLabel.textContent = 'Count Pills';
      btnSpinner.classList.add('hidden');
      countBtn.disabled = !(refFile && groupFile);
    }
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorBanner.classList.remove('hidden');
    errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function hideError() {
    errorBanner.classList.add('hidden');
  }

  function hideResults() {
    resultsSection.classList.add('hidden');
  }

  function showResults(data) {
    const count = data.count || 0;

    // Animate count from 0 to result
    animateCount(count);

    // Meta info
    const metaParts = [];
    if (typeof data.ref_area_px === 'number') {
      metaParts.push(`Reference pill area: ${data.ref_area_px.toLocaleString()} px²`);
    }
    if (data.is_white_pill !== undefined) {
      metaParts.push(data.is_white_pill ? 'Detection mode: brightness' : 'Detection mode: color');
    }
    resultsMeta.textContent = metaParts.join(' · ');

    // Annotated image
    if (data.annotated_image) {
      const src = 'data:image/jpeg;base64,' + data.annotated_image;
      annotatedImg.src = src;
      downloadLink.href = src;
      downloadLink.classList.remove('hidden');
    } else {
      downloadLink.classList.add('hidden');
    }

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function animateCount(target) {
    const duration = 600; // ms
    const startTime = performance.now();
    const startValue = 0;

    function step(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(startValue + (target - startValue) * eased);
      countNumber.textContent = current.toLocaleString();
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        countNumber.textContent = target.toLocaleString();
      }
    }

    requestAnimationFrame(step);
  }

})();
