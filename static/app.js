/* =========================================================
   ISpyPill — Frontend Logic
   Vanilla JS, no frameworks
   ========================================================= */

(function () {
  'use strict';

  // =========================================================
  // State
  // =========================================================
  let refFile   = null;
  let groupFile = null;

  // =========================================================
  // DOM refs
  // =========================================================
  const refDropZone     = document.getElementById('ref-drop-zone');
  const groupDropZone   = document.getElementById('group-drop-zone');
  const refPlaceholder  = document.getElementById('ref-placeholder');
  const groupPlaceholder= document.getElementById('group-placeholder');
  const refPreviewWrap  = document.getElementById('ref-preview-wrap');
  const groupPreviewWrap= document.getElementById('group-preview-wrap');
  const refPreview      = document.getElementById('ref-preview');
  const groupPreview    = document.getElementById('group-preview');
  const refClear        = document.getElementById('ref-clear');
  const groupClear      = document.getElementById('group-clear');
  const refGallery      = document.getElementById('ref-file-gallery');
  const refCamera       = document.getElementById('ref-file-camera');
  const groupGallery    = document.getElementById('group-file-gallery');
  const groupCamera     = document.getElementById('group-file-camera');
  const countBtn        = document.getElementById('count-btn');
  const btnLabel        = countBtn.querySelector('.btn-label');
  const btnSpinner      = countBtn.querySelector('.btn-spinner');
  const errorBanner     = document.getElementById('error-banner');
  const errorMessage    = document.getElementById('error-message');
  const resultsSection  = document.getElementById('results-section');
  const countNumber     = document.getElementById('count-number');
  const resultsBadges   = document.getElementById('results-badges');
  const annotatedImg    = document.getElementById('annotated-img');
  const downloadLink    = document.getElementById('download-link');
  const resetBtn        = document.getElementById('reset-btn');

  // =========================================================
  // File Handling
  // =========================================================

  function handleFile(file, type) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      showError('Please upload an image file (PNG, JPG, HEIC, etc.).');
      return;
    }
    if (file.size > 16 * 1024 * 1024) {
      showError('Image exceeds 16 MB. Please use a smaller photo.');
      return;
    }
    if (type === 'ref') {
      refFile = file;
      showPreview(file, refPreview, refPlaceholder, refPreviewWrap, refDropZone);
    } else {
      groupFile = file;
      showPreview(file, groupPreview, groupPlaceholder, groupPreviewWrap, groupDropZone);
    }
    hideError();
    updateCountButton();
  }

  function showPreview(file, imgEl, placeholder, previewWrap, dropZone) {
    const reader = new FileReader();
    reader.onload = function (e) {
      imgEl.src = e.target.result;
      placeholder.classList.add('hidden');
      previewWrap.classList.remove('hidden');
      dropZone.classList.add('has-image');
    };
    reader.readAsDataURL(file);
  }

  function clearUpload(type) {
    if (type === 'ref') {
      refFile = null;
      refPreview.src = '';
      refPreviewWrap.classList.add('hidden');
      refPlaceholder.classList.remove('hidden');
      refDropZone.classList.remove('has-image');
      refGallery.value = '';
      refCamera.value  = '';
    } else {
      groupFile = null;
      groupPreview.src = '';
      groupPreviewWrap.classList.add('hidden');
      groupPlaceholder.classList.remove('hidden');
      groupDropZone.classList.remove('has-image');
      groupGallery.value = '';
      groupCamera.value  = '';
    }
    updateCountButton();
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
      this.value = ''; // Allow re-selecting same file
    });
  }

  bindFileInput(refGallery,   'ref');
  bindFileInput(refCamera,    'ref');
  bindFileInput(groupGallery, 'group');
  bindFileInput(groupCamera,  'group');

  // Clear buttons
  refClear.addEventListener('click', function (e) {
    e.stopPropagation();
    clearUpload('ref');
  });
  groupClear.addEventListener('click', function (e) {
    e.stopPropagation();
    clearUpload('group');
  });

  // =========================================================
  // Drop Zone — click delegates to gallery input
  // =========================================================

  function bindDropZoneClick(dropZone, galleryInput) {
    dropZone.addEventListener('click', function (e) {
      // Don't trigger file picker if clicking clear button or when image shown
      if (e.target.closest('.clear-btn')) return;
      if (dropZone.classList.contains('has-image')) return;
      galleryInput.click();
    });
    dropZone.addEventListener('keydown', function (e) {
      if ((e.key === 'Enter' || e.key === ' ') && !dropZone.classList.contains('has-image')) {
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
  // Analysis
  // =========================================================

  countBtn.addEventListener('click', async function () {
    if (!refFile || !groupFile) return;

    setLoading(true);
    hideError();
    hideResults();

    const formData = new FormData();
    formData.append('reference_pill', refFile);
    formData.append('group_photo',    groupFile);

    try {
      const response = await fetch('/analyze', { method: 'POST', body: formData });
      const data     = await response.json();

      if (!response.ok) {
        showError(data.error || 'An unknown error occurred. Please try again.');
        return;
      }
      showResults(data);
    } catch (err) {
      showError(
        err.name === 'TypeError'
          ? 'Network error — please check your connection and try again.'
          : 'An unexpected error occurred: ' + err.message
      );
    } finally {
      setLoading(false);
    }
  });

  // =========================================================
  // Reset / Count Another
  // =========================================================

  resetBtn.addEventListener('click', function () {
    clearUpload('ref');
    clearUpload('group');
    hideResults();
    hideError();
    document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  // =========================================================
  // UI Helpers
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

  function hideError()   { errorBanner.classList.add('hidden'); }
  function hideResults() { resultsSection.classList.add('hidden'); }

  function showResults(data) {
    animateCount(data.count || 0);

    // Info badges
    resultsBadges.innerHTML = '';
    if (typeof data.num_color_clusters === 'number') {
      addBadge(data.num_color_clusters === 1 ? '1 color' : data.num_color_clusters + ' colors');
    }
    if (data.is_white_pill !== undefined) {
      addBadge(data.is_white_pill ? 'Brightness mode' : 'Color mode');
    }
    if (typeof data.ref_area_px === 'number') {
      addBadge(data.ref_area_px.toLocaleString() + ' px² ref');
    }

    // Annotated image
    if (data.annotated_image) {
      const src = 'data:image/jpeg;base64,' + data.annotated_image;
      annotatedImg.src  = src;
      downloadLink.href = src;
    }

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function addBadge(text) {
    const span = document.createElement('span');
    span.className   = 'badge';
    span.textContent = text;
    resultsBadges.appendChild(span);
  }

  function animateCount(target) {
    const duration = 500;
    const start    = performance.now();

    (function step(now) {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      countNumber.textContent = Math.round(eased * target).toLocaleString();
      if (t < 1) requestAnimationFrame(step);
      else        countNumber.textContent = target.toLocaleString();
    })(start);
  }

})();
