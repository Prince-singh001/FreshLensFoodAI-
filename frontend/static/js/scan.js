/**
 * FreshLens AI - Food Scanner Client Module
 * Manages mode switching (Live Scan / Upload), camera streaming with flip controls,
 * drag-and-drop, client-side quality validation, prediction API integration,
 * animated result reveals, and quality review feedback.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Mode Controller Elements
  const modeUploadBtn = document.getElementById("modeUploadBtn");
  const modeLiveBtn = document.getElementById("modeLiveBtn");
  const uploadModePanel = document.getElementById("uploadModePanel");
  const liveModePanel = document.getElementById("liveModePanel");
  const modeIndicatorBadge = document.getElementById("modeIndicatorBadge");

  // Category Filter Tabs
  const tabButtons = document.querySelectorAll(".tab-btn");

  // Upload Area Elements
  const uploadArea = document.getElementById("uploadArea");
  const foodImage = document.getElementById("foodImage");
  const previewBox = document.getElementById("previewBox");
  const previewActionsBar = document.getElementById("previewActionsBar");
  const replaceImageBtn = document.getElementById("replaceImageBtn");
  const removeImageBtn = document.getElementById("removeImageBtn");
  const sampleButtons = document.querySelectorAll(".sample-btn");
  const qualityToast = document.getElementById("qualityToast");
  const qualityToastMessage = document.getElementById("qualityToastMessage");

  // Camera Elements
  const cameraArea = document.getElementById("cameraArea");
  const webcam = document.getElementById("webcam");
  const captureImage = document.getElementById("captureImage");
  const switchCameraBtn = document.getElementById("switchCameraBtn");
  const closeCameraBtn = document.getElementById("closeCameraBtn");
  const cameraErrorState = document.getElementById("cameraErrorState");
  const cameraStatusPill = document.getElementById("cameraStatusPill");
  const cameraErrorIcon = document.getElementById("cameraErrorIcon");
  const cameraErrorTitle = document.getElementById("cameraErrorTitle");
  const cameraErrorMessage = document.getElementById("cameraErrorMessage");
  const retryCameraBtn = document.getElementById("retryCameraBtn");
  const fallbackUploadBtn = document.getElementById("fallbackUploadBtn");

  // Primary Actions
  const scanNow = document.getElementById("scanNow");
  const resetScan = document.getElementById("resetScan");
  const progressOverlay = document.getElementById("progressOverlay");
  const progressStepText = document.getElementById("progressStepText");
  const progressBarInner = document.getElementById("progressBarInner");

  // Result Elements
  const resultCard = document.getElementById("resultCard");
  const scanErrorBox = document.getElementById("scanErrorBox");
  const scanErrorCode = document.getElementById("scanErrorCode");
  const scanErrorMessage = document.getElementById("scanErrorMessage");
  const categoryWarningBox = document.getElementById("categoryWarningBox");
  const categoryWarningMessage = document.getElementById("categoryWarningMessage");
  const annotatedImageBox = document.getElementById("annotatedImageBox");
  const annotatedImage = document.getElementById("annotatedImage");
  const resultSummaryBar = document.getElementById("resultSummaryBar");
  const summaryText = document.getElementById("summaryText");
  const multiObjectsContainer = document.getElementById("multiObjectsContainer");
  const scanInferenceTime = document.getElementById("scanInferenceTime");
  const afterScanActions = document.getElementById("afterScanActions");
  const scanAnotherBtn = document.getElementById("scanAnotherBtn");

  // Feedback Elements
  const feedbackSection = document.getElementById("feedbackSection");
  const feedbackYes = document.getElementById("feedbackYes");
  const feedbackNo = document.getElementById("feedbackNo");
  const correctionForm = document.getElementById("correctionForm");
  const submitFeedbackBtn = document.getElementById("submitFeedbackBtn");
  const correctItemSelect = document.getElementById("correctItemSelect");
  const correctConditionSelect = document.getElementById("correctConditionSelect");
  const feedbackResultMsg = document.getElementById("feedbackResultMsg");

  // State Variables
  let localStream = null;
  let currentFacingMode = "environment"; // "environment" (rear default for food) or "user" (front)
  let activeCategory = "All";
  let lastUploadedImageUrl = "";
  let lastPredictedLabel = "";
  let isAnalyzing = false;
  let isCameraStarting = false;
  let currentActiveMode = "upload";

  // =========================================================================
  // 1. Mode Switching (Upload Image vs Live Camera)
  // =========================================================================
  function switchMode(mode) {
    currentActiveMode = mode;
    if (mode === "upload") {
      stopCamera();
      if (modeUploadBtn) {
        modeUploadBtn.classList.add("active");
        modeUploadBtn.setAttribute("aria-selected", "true");
      }
      if (modeLiveBtn) {
        modeLiveBtn.classList.remove("active");
        modeLiveBtn.setAttribute("aria-selected", "false");
      }
      if (uploadModePanel) uploadModePanel.style.display = "block";
      if (liveModePanel) liveModePanel.style.display = "none";
      if (modeIndicatorBadge) modeIndicatorBadge.textContent = "Upload Mode";
    } else if (mode === "live") {
      if (modeLiveBtn) {
        modeLiveBtn.classList.add("active");
        modeLiveBtn.setAttribute("aria-selected", "true");
      }
      if (modeUploadBtn) {
        modeUploadBtn.classList.remove("active");
        modeUploadBtn.setAttribute("aria-selected", "false");
      }
      if (uploadModePanel) uploadModePanel.style.display = "none";
      if (liveModePanel) liveModePanel.style.display = "block";
      if (modeIndicatorBadge) modeIndicatorBadge.textContent = "Live Camera Mode";
      startCamera();
    }
  }

  if (modeUploadBtn) {
    modeUploadBtn.addEventListener("click", () => switchMode("upload"));
  }
  if (modeLiveBtn) {
    modeLiveBtn.addEventListener("click", () => switchMode("live"));
  }
  if (fallbackUploadBtn) {
    fallbackUploadBtn.addEventListener("click", () => switchMode("upload"));
  }

  // =========================================================================
  // 2. Category Filter Tabs
  // =========================================================================
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const type = btn.getAttribute("data-type");
      activeCategory = type === "all" ? "All" : type.charAt(0).toUpperCase() + type.slice(1);
    });
  });

  // =========================================================================
  // 3. Camera Diagnostics, Streaming & Viewfinder Controls
  // =========================================================================

  function checkEnvironmentSecurity() {
    const hostname = window.location.hostname || "";
    const protocol = window.location.protocol || "";

    const isLocalhost =
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "::1" ||
      hostname.endsWith(".localhost");

    const isLanIp =
      /^192\.168\./.test(hostname) ||
      /^10\./.test(hostname) ||
      /^172\.(1[6-9]|2[0-9]|3[0-1])\./.test(hostname);

    const isSecureContext =
      window.isSecureContext === true ||
      protocol === "https:" ||
      isLocalhost;

    const hasMediaDevices = Boolean(
      navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === "function"
    );

    return {
      hostname,
      protocol,
      isLocalhost,
      isLanIp,
      isSecureContext,
      hasMediaDevices
    };
  }

  function getCameraSupportStatus() {
    const env = checkEnvironmentSecurity();

    // 1. If mediaDevices.getUserMedia is available, attempt real camera access immediately!
    // Do not prematurely block if the browser has already exposed the API.
    if (env.hasMediaDevices) {
      return { supported: true, env };
    }

    // 2. Only show the HTTPS-required message when:
    // - the application is not running on localhost/127.0.0.1
    // - AND the browser does not consider the page a secure context
    // - AND camera access cannot be safely requested
    if (!env.isLocalhost && !env.isSecureContext) {
      if (env.isLanIp) {
        return {
          supported: false,
          title: "Secure Connection (HTTPS) Required",
          message: "Camera access requires HTTPS on this address. Open the secure HTTPS version of FreshLens AI, or open http://localhost:5000 directly on this machine.",
          icon: "🔒"
        };
      }
      return {
        supported: false,
        title: "Secure Connection (HTTPS) Required",
        message: "Camera access requires HTTPS on this address. Open the secure HTTPS version of FreshLens AI.",
        icon: "🔒"
      };
    }

    // 3. If running on localhost/127.0.0.1, or in a secure context, but getUserMedia is missing:
    if (env.isLocalhost) {
      return {
        supported: false,
        title: "Camera Access Notice",
        message: "Camera access is available. Please allow camera permission in your browser.",
        icon: "📷"
      };
    }

    return {
      supported: false,
      title: "Camera Not Supported",
      message: "Your browser does not support media device camera access. Please use an updated modern browser such as Chrome, Edge, Safari, or Firefox.",
      icon: "⚠️"
    };
  }

  function parseCameraError(err) {
    const name = err ? err.name : "";
    const msg = err ? err.message : "";
    const env = checkEnvironmentSecurity();

    if (name === "NotAllowedError" || name === "PermissionDeniedError") {
      return {
        title: "Camera Permission Blocked",
        message: "Camera permission is blocked. Allow camera access in your browser settings and click Try Again.",
        icon: "🚫"
      };
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      return {
        title: "No Camera Detected",
        message: "No camera was detected on this device.",
        icon: "📷"
      };
    }
    if (name === "NotReadableError" || name === "TrackStartError") {
      return {
        title: "Camera In Use",
        message: "Camera is already in use by another app or browser tab. Please close other camera apps and click Try Again.",
        icon: "⚠️"
      };
    }
    if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
      return {
        title: "Camera Settings Not Supported",
        message: "The requested camera settings could not be satisfied by your device. Click Try Again to use default settings.",
        icon: "⚙️"
      };
    }
    if (name === "SecurityError") {
      if (env.isLanIp || !env.isLocalhost) {
        return {
          title: "Secure Connection (HTTPS) Required",
          message: "Camera access requires HTTPS on this address. Open the secure HTTPS version of FreshLens AI.",
          icon: "🔒"
        };
      }
      return {
        title: "Camera Access Notice",
        message: "Camera access is available. Please allow camera permission in your browser.",
        icon: "🔒"
      };
    }
    if (name === "AbortError") {
      return {
        title: "Camera Initialization Interrupted",
        message: "Camera initialization was interrupted. Please click Try Again.",
        icon: "🔄"
      };
    }
    return {
      title: "Camera Access Unavailable",
      message: env.isLocalhost
        ? "Camera access is available. Please allow camera permission in your browser."
        : (msg ? `Unable to access camera (${name}: ${msg}). You can allow camera access in browser settings or upload an image instead.` : "Camera permission was denied or camera is unavailable. You can allow camera access in browser settings or upload an image instead."),
      icon: "📷"
    };
  }

  function showCameraError(title, message, icon = "📷") {
    stopCamera();
    if (cameraArea) cameraArea.style.display = "none";
    if (cameraErrorState) {
      cameraErrorState.style.display = "block";
      if (cameraErrorTitle) cameraErrorTitle.textContent = title;
      if (cameraErrorMessage) cameraErrorMessage.textContent = message;
      if (cameraErrorIcon) cameraErrorIcon.textContent = icon;
    }
    if (cameraStatusPill) cameraStatusPill.textContent = title;
    if (captureImage) captureImage.disabled = true;
    if (switchCameraBtn) switchCameraBtn.disabled = false;
  }

  async function requestCameraStream(facingMode) {
    const attempts = [
      // 1. Ideal facingMode & resolution (1280x720)
      {
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      },
      // 2. Ideal facingMode without resolution constraints
      {
        video: {
          facingMode: { ideal: facingMode }
        },
        audio: false
      },
      // 3. Fallback: simple video constraint (desktop webcams, virtual cameras)
      {
        video: true,
        audio: false
      }
    ];

    let lastErr = null;
    for (const constraints of attempts) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        return stream;
      } catch (err) {
        lastErr = err;
        if (
          err.name === "NotAllowedError" ||
          err.name === "PermissionDeniedError" ||
          err.name === "SecurityError" ||
          err.name === "NotFoundError" ||
          err.name === "DevicesNotFoundError"
        ) {
          throw err;
        }
        console.warn("Retrying camera with relaxed constraints after:", err.name, err.message);
      }
    }
    throw lastErr;
  }

  async function startCamera() {
    if (isCameraStarting) return;
    isCameraStarting = true;

    // Check environment support
    const support = getCameraSupportStatus();
    if (!support.supported) {
      showCameraError(support.title, support.message, support.icon);
      isCameraStarting = false;
      return;
    }

    // Stop existing stream cleanly before re-requesting
    stopCamera();

    // Reset error state and activate viewfinder in initializing state
    if (cameraErrorState) cameraErrorState.style.display = "none";
    if (cameraArea) cameraArea.style.display = "block";
    if (cameraStatusPill) cameraStatusPill.textContent = "Initializing Camera...";
    if (captureImage) captureImage.disabled = true;
    if (switchCameraBtn) switchCameraBtn.disabled = true;

    try {
      const stream = await requestCameraStream(currentFacingMode);
      localStream = stream;

      if (webcam) {
        webcam.srcObject = stream;
        webcam.setAttribute("playsinline", "true");
        webcam.autoplay = true;
        webcam.muted = true;

        try {
          await webcam.play();
        } catch (playErr) {
          await new Promise((resolve) => {
            webcam.onloadedmetadata = () => {
              webcam.play().then(resolve).catch(resolve);
            };
            setTimeout(resolve, 1000);
          });
        }
      }

      const tracks = stream.getVideoTracks();
      if (tracks.length > 0) {
        tracks[0].onended = () => {
          console.warn("Camera track ended unexpectedly.");
          showCameraError(
            "Camera Disconnected",
            "The camera device was disconnected or stopped by the system. Click Try Again to reconnect.",
            "📷"
          );
        };
      }

      if (cameraStatusPill) cameraStatusPill.textContent = "Camera Ready - Place food inside frame";
      if (captureImage) captureImage.disabled = false;
      if (switchCameraBtn) switchCameraBtn.disabled = false;

    } catch (err) {
      console.warn("Camera access failed:", err);
      const errInfo = parseCameraError(err);
      showCameraError(errInfo.title, errInfo.message, errInfo.icon);
    } finally {
      isCameraStarting = false;
    }
  }

  function stopCamera() {
    if (localStream) {
      try {
        localStream.getTracks().forEach((track) => track.stop());
      } catch (e) {
        console.warn("Error stopping camera tracks:", e);
      }
      localStream = null;
    }
    if (webcam) {
      try {
        webcam.pause();
      } catch (e) {}
      webcam.srcObject = null;
    }
    if (captureImage) {
      captureImage.disabled = true;
    }
  }

  // Flip Camera
  if (switchCameraBtn) {
    switchCameraBtn.addEventListener("click", async () => {
      if (isCameraStarting) return;
      currentFacingMode = currentFacingMode === "environment" ? "user" : "environment";
      await startCamera();
    });
  }

  // Close Camera
  if (closeCameraBtn) {
    closeCameraBtn.addEventListener("click", () => {
      switchMode("upload");
    });
  }

  // Retry Camera
  if (retryCameraBtn) {
    retryCameraBtn.addEventListener("click", () => {
      startCamera();
    });
  }

  // Capture Photo
  if (captureImage) {
    captureImage.addEventListener("click", () => {
      if (!localStream || !webcam || isAnalyzing) return;

      const width = webcam.videoWidth || 640;
      const height = webcam.videoHeight || 480;
      if (width === 0 || height === 0) {
        console.warn("Camera video dimensions not ready yet.");
        return;
      }

      if (cameraStatusPill) cameraStatusPill.textContent = "Capturing...";
      captureImage.disabled = true;

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(webcam, 0, 0, width, height);

      canvas.toBlob(async (blob) => {
        if (!blob) {
          console.error("Failed to generate capture image blob.");
          if (cameraStatusPill) cameraStatusPill.textContent = "Camera Ready - Place food inside frame";
          captureImage.disabled = false;
          return;
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        const file = new File([blob], `camera_scan_${timestamp}.jpg`, { type: "image/jpeg" });
        const dt = new DataTransfer();
        dt.items.add(file);
        if (foodImage) foodImage.files = dt.files;

        // Stop camera cleanly to free hardware and conserve battery
        stopCamera();

        // Switch to upload view to display captured photo preview
        switchMode("upload");
        displayImagePreview(canvas.toDataURL("image/jpeg"), file);

        // Immediately execute food freshness analysis through existing AI pipeline
        await executeFoodAnalysis(file);
      }, "image/jpeg", 0.92);
    });
  }

  // =========================================================================
  // 4. File Upload & Drag-and-Drop
  // =========================================================================
  if (uploadArea) {
    ["dragenter", "dragover"].forEach((eventName) => {
      uploadArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add("drag-over");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      uploadArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove("drag-over");
      });
    });

    uploadArea.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        if (foodImage) foodImage.files = dt.files;
        handleFileSelected(dt.files[0]);
      }
    });
  }

  if (foodImage) {
    foodImage.addEventListener("change", () => {
      if (foodImage.files && foodImage.files[0]) {
        handleFileSelected(foodImage.files[0]);
      }
    });
  }

  function handleFileSelected(file) {
    if (!file) return;

    // Basic file validation
    const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type.toLowerCase())) {
      showQualityToast("Unsupported format. Please upload JPG, PNG, or WEBP.");
      return;
    }

    if (file.size > 16 * 1024 * 1024) {
      showQualityToast("Image file exceeds the 16MB limit.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      displayImagePreview(e.target.result, file);
      runPreflightQualityCheck(e.target.result);
    };
    reader.readAsDataURL(file);
    resetResultsView();
  }

  function displayImagePreview(dataUrl, file) {
    if (previewBox) {
      previewBox.innerHTML = `
        <img src="${dataUrl}" alt="Food Preview" style="max-height:280px; border-radius:12px; margin:0 auto; box-shadow:var(--shadow-sm);" />
        <p style="font-size:0.8rem; color:var(--muted); margin-top:8px;">${file.name || 'Selected Food Image'}</p>
      `;
    }
    if (previewActionsBar) previewActionsBar.style.display = "flex";
  }

  // Replace / Remove Image
  if (replaceImageBtn && foodImage) {
    replaceImageBtn.addEventListener("click", () => foodImage.click());
  }

  if (removeImageBtn) {
    removeImageBtn.addEventListener("click", () => {
      if (foodImage) foodImage.value = "";
      if (previewBox) {
        previewBox.innerHTML = `
          <span class="upload-icon">📷</span>
          <h3 style="font-size:1.1rem; margin-bottom:4px;">Drop your food image here</h3>
          <p style="color:var(--muted); font-size:0.85rem;">or choose an image from your device (JPG, PNG, WEBP)</p>
        `;
      }
      if (previewActionsBar) previewActionsBar.style.display = "none";
      if (qualityToast) qualityToast.style.display = "none";
      resetResultsView();
    });
  }

  // Pre-flight Client-Side Quality Check (Darkness / Dimness detection)
  function runPreflightQualityCheck(dataUrl) {
    const img = new Image();
    img.onload = () => {
      if (img.width < 40 || img.height < 40) {
        showQualityToast("The image resolution is very small. For best detection, use a clearer photo.");
        return;
      }
      // Sample brightness on a lightweight canvas
      try {
        const canvas = document.createElement("canvas");
        canvas.width = 40;
        canvas.height = 40;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, 40, 40);
        const data = ctx.getImageData(0, 0, 40, 40).data;
        let total = 0;
        for (let i = 0; i < data.length; i += 4) {
          total += (data[i] + data[i + 1] + data[i + 2]) / 3;
        }
        const avgBrightness = total / (data.length / 4);
        if (avgBrightness < 16) {
          showQualityToast("The photo appears very dark. Consider taking the photo in better lighting.");
        } else {
          if (qualityToast) qualityToast.style.display = "none";
        }
      } catch (e) {
        // Cross-origin canvas limitation fallback
      }
    };
    img.src = dataUrl;
  }

  function showQualityToast(msg) {
    if (qualityToastMessage) qualityToastMessage.textContent = msg;
    if (qualityToast) qualityToast.style.display = "flex";
  }

  // Quick Samples
  sampleButtons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      switchMode("upload");
      resetResultsView();
      const imageUrl = btn.getAttribute("data-image");
      if (previewBox) {
        previewBox.innerHTML = `<img src="${imageUrl}" alt="Sample Food" style="max-height:280px; border-radius:12px; margin:0 auto; box-shadow:var(--shadow-sm);" />`;
      }
      if (previewActionsBar) previewActionsBar.style.display = "flex";

      try {
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        const file = new File([blob], imageUrl.split("/").pop(), { type: blob.type || "image/jpeg" });
        const dt = new DataTransfer();
        dt.items.add(file);
        if (foodImage) foodImage.files = dt.files;
      } catch (err) {
        console.warn("Could not load sample into file input:", err);
      }
    });
  });

  // =========================================================================
  // 5. Reset Results View
  // =========================================================================
  function resetResultsView() {
    if (scanErrorBox) scanErrorBox.style.display = "none";
    if (categoryWarningBox) categoryWarningBox.style.display = "none";
    if (annotatedImageBox) annotatedImageBox.style.display = "none";
    if (resultSummaryBar) resultSummaryBar.style.display = "none";
    if (afterScanActions) afterScanActions.style.display = "none";
    if (feedbackSection) feedbackSection.style.display = "none";
    if (correctionForm) correctionForm.style.display = "none";
    if (feedbackResultMsg) feedbackResultMsg.style.display = "none";
    if (scanInferenceTime) scanInferenceTime.textContent = "";

    if (multiObjectsContainer) {
      multiObjectsContainer.innerHTML = `
        <div id="resultPlaceholder" style="color:var(--muted); text-align:center; padding:40px 16px;">
          <span style="font-size:2.5rem; display:block; margin-bottom:10px;">🥗</span>
          <h3 style="font-size:1.1rem; color:var(--text); margin-bottom:4px;">No Scan Yet</h3>
          <p style="font-size:0.875rem; max-width:320px; margin:0 auto;">
            Capture or upload a food image and click <strong>Analyze Food</strong> to view detected items and freshness assessments.
          </p>
        </div>
      `;
    }
  }

  if (resetScan) {
    resetScan.addEventListener("click", () => {
      stopCamera();
      if (foodImage) foodImage.value = "";
      if (previewBox) {
        previewBox.innerHTML = `
          <span class="upload-icon">📷</span>
          <h3 style="font-size:1.1rem; margin-bottom:4px;">Drop your food image here</h3>
          <p style="color:var(--muted); font-size:0.85rem;">or choose an image from your device (JPG, PNG, WEBP)</p>
        `;
      }
      if (previewActionsBar) previewActionsBar.style.display = "none";
      if (qualityToast) qualityToast.style.display = "none";
      switchMode("upload");
      resetResultsView();
    });
  }

  if (scanAnotherBtn) {
    scanAnotherBtn.addEventListener("click", () => {
      if (foodImage) foodImage.value = "";
      if (previewBox) {
        previewBox.innerHTML = `
          <span class="upload-icon">📷</span>
          <h3 style="font-size:1.1rem; margin-bottom:4px;">Drop your food image here</h3>
          <p style="color:var(--muted); font-size:0.85rem;">or choose an image from your device (JPG, PNG, WEBP)</p>
        `;
      }
      if (previewActionsBar) previewActionsBar.style.display = "none";
      resetResultsView();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  // =========================================================================
  // 6. Prediction Execution & Analysis Workflow
  // =========================================================================
  async function executeFoodAnalysis(file) {
    if (isAnalyzing) return;

    if (!file) {
      alert("Please select or capture a food image first.");
      return;
    }

    resetResultsView();
    isAnalyzing = true;
    if (scanNow) {
      scanNow.disabled = true;
      scanNow.innerHTML = `<span>⏳ Analyzing...</span>`;
    }
    if (captureImage) captureImage.disabled = true;

    // Activate Honest Progress Indicator
    if (progressOverlay) progressOverlay.style.display = "flex";
    if (progressBarInner) progressBarInner.style.width = "25%";
    if (progressStepText) progressStepText.textContent = "Examining image characteristics...";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("selected_category", activeCategory);

    let step1Timer = null;
    let step2Timer = null;

    try {
      step1Timer = setTimeout(() => {
        if (progressBarInner) progressBarInner.style.width = "60%";
        if (progressStepText) progressStepText.textContent = "Detecting food objects & localization...";
      }, 180);

      step2Timer = setTimeout(() => {
        if (progressBarInner) progressBarInner.style.width = "85%";
        if (progressStepText) progressStepText.textContent = "Evaluating freshness & condition...";
      }, 450);

      const response = await fetch("/api/v1/predict", {
        method: "POST",
        body: formData
      });

      if (step1Timer) clearTimeout(step1Timer);
      if (step2Timer) clearTimeout(step2Timer);

      const data = await response.json();
      if (progressOverlay) progressOverlay.style.display = "none";

      if (!response.ok || !data.success) {
        handleScanError(data);
        return;
      }

      renderScanSuccess(data);

      // Success state flash on CTA button
      if (scanNow) {
        scanNow.innerHTML = `<span>✅ Analysis Complete</span>`;
        setTimeout(() => {
          scanNow.innerHTML = `<span>🔍 Analyze Food</span>`;
        }, 2000);
      }

    } catch (err) {
      if (step1Timer) clearTimeout(step1Timer);
      if (step2Timer) clearTimeout(step2Timer);
      if (progressOverlay) progressOverlay.style.display = "none";
      handleScanError({
        error: {
          code: "NETWORK_ERROR",
          message: "Unable to connect to the server. Please check your network connection and try again."
        }
      });
    } finally {
      isAnalyzing = false;
      if (scanNow) scanNow.disabled = false;
      if (captureImage) captureImage.disabled = false;
    }
  }

  if (scanNow) {
    scanNow.addEventListener("click", () => {
      if (foodImage && foodImage.files && foodImage.files[0]) {
        executeFoodAnalysis(foodImage.files[0]);
      } else {
        alert("Please select or capture a food image first.");
      }
    });
  }

  function handleScanError(data) {
    const status = data.status || "";
    const err = data.error || { code: "SCAN_FAILED", message: data.message || "Scan could not be completed." };

    if (annotatedImageBox) annotatedImageBox.style.display = "none";
    if (resultSummaryBar) resultSummaryBar.style.display = "none";
    if (categoryWarningBox) categoryWarningBox.style.display = "none";
    if (afterScanActions) afterScanActions.style.display = "none";
    if (feedbackSection) feedbackSection.style.display = "none";

    if (status === "no_food" || status === "no_supported_food_detected" || data.food_detected === false && status !== "low_confidence") {
      if (scanErrorBox) scanErrorBox.style.display = "none";
      if (multiObjectsContainer) {
        multiObjectsContainer.innerHTML = `
          <div class="rejection-card no-food-card" style="text-align:center; padding:36px 20px; background:var(--bg-alt); border-radius:var(--radius-lg); border:1px solid var(--border);">
            <span style="font-size:3rem; display:block; margin-bottom:12px;">🔍</span>
            <h3 style="font-size:1.35rem; font-weight:800; color:var(--text); margin-bottom:8px;">No Food Detected</h3>
            <p style="font-size:0.95rem; color:var(--text-secondary); max-width:420px; margin:0 auto 12px; line-height:1.5;">
              ${data.message || err.message || "We couldn't confidently identify a supported food item."}
            </p>
            <p style="font-size:0.875rem; color:var(--muted); max-width:400px; margin:0 auto 20px;">
              Please point the camera at a fruit, vegetable, or food item and try again.
            </p>
            <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
              <button type="button" class="btn primary touch-friendly" id="rejectionTryAgainBtn" style="padding:10px 22px; font-size:0.9rem;">
                📷 Try Again
              </button>
              <button type="button" class="btn secondary touch-friendly" id="rejectionUploadBtn" style="padding:10px 22px; font-size:0.9rem;">
                🖼️ Upload Image
              </button>
            </div>
          </div>
        `;

        const tryAgainBtn = document.getElementById("rejectionTryAgainBtn");
        if (tryAgainBtn) {
          tryAgainBtn.addEventListener("click", () => {
            switchMode("live");
            startCamera();
          });
        }
        const uploadBtn = document.getElementById("rejectionUploadBtn");
        if (uploadBtn) {
          uploadBtn.addEventListener("click", () => {
            switchMode("upload");
            if (foodImage) foodImage.click();
          });
        }
      }
    } else if (status === "low_confidence") {
      if (scanErrorBox) scanErrorBox.style.display = "none";
      if (multiObjectsContainer) {
        multiObjectsContainer.innerHTML = `
          <div class="rejection-card low-confidence-card" style="text-align:center; padding:36px 20px; background:var(--bg-alt); border-radius:var(--radius-lg); border:1px solid var(--border);">
            <span style="font-size:3rem; display:block; margin-bottom:12px;">⚠️</span>
            <h3 style="font-size:1.35rem; font-weight:800; color:var(--text); margin-bottom:8px;">Low Confidence</h3>
            <p style="font-size:0.95rem; color:var(--text-secondary); max-width:420px; margin:0 auto 16px; line-height:1.5;">
              ${data.message || err.message || "The food could not be identified confidently."}
            </p>
            <div style="text-align:left; max-width:340px; margin:0 auto 24px; padding:14px 18px; background:var(--card); border-radius:var(--radius-md); border:1px solid var(--border);">
              <strong style="font-size:0.875rem; color:var(--text); display:block; margin-bottom:6px;">Try:</strong>
              <ul style="font-size:0.85rem; color:var(--text-secondary); margin:0; padding-left:18px; line-height:1.6;">
                <li>Better, even lighting</li>
                <li>A closer image of the food item</li>
                <li>A plain, clutter-free background</li>
                <li>Keeping the food centered</li>
              </ul>
            </div>
            <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
              <button type="button" class="btn primary touch-friendly" id="lowConfTryAgainBtn" style="padding:10px 22px; font-size:0.9rem;">
                📷 Try Again
              </button>
              <button type="button" class="btn secondary touch-friendly" id="lowConfUploadBtn" style="padding:10px 22px; font-size:0.9rem;">
                🖼️ Upload Image
              </button>
            </div>
          </div>
        `;

        const tryAgainBtn = document.getElementById("lowConfTryAgainBtn");
        if (tryAgainBtn) {
          tryAgainBtn.addEventListener("click", () => {
            switchMode("live");
            startCamera();
          });
        }
        const uploadBtn = document.getElementById("lowConfUploadBtn");
        if (uploadBtn) {
          uploadBtn.addEventListener("click", () => {
            switchMode("upload");
            if (foodImage) foodImage.click();
          });
        }
      }
    } else {
      if (scanErrorCode) scanErrorCode.textContent = err.code || "Scan Error";
      if (scanErrorMessage) scanErrorMessage.textContent = err.message || "Please upload a clearer food image.";
      if (scanErrorBox) scanErrorBox.style.display = "flex";
    }

    // Scroll smoothly to results card on mobile
    if (window.innerWidth <= 768 && resultCard) {
      resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function renderScanSuccess(data) {
    lastUploadedImageUrl = data.image_url || "";
    lastPredictedLabel = data.label || "";

    if (scanInferenceTime && data.inference_time_ms) {
      scanInferenceTime.textContent = `${data.inference_time_ms} ms`;
    }

    // Annotated bounding box image display
    if (data.annotated_image_url) {
      if (annotatedImage) annotatedImage.src = data.annotated_image_url;
      if (annotatedImageBox) annotatedImageBox.style.display = "block";
    }

    // Category warning
    if (data.warning) {
      if (categoryWarningMessage) categoryWarningMessage.textContent = data.warning;
      if (categoryWarningBox) categoryWarningBox.style.display = "flex";
    }

    // Summary bar
    const sum = data.summary || {};
    const totalObjs = sum.total_objects || (data.objects ? data.objects.length : 1);
    if (summaryText) {
      summaryText.innerHTML = `<strong>${totalObjs} Object${totalObjs > 1 ? 's' : ''} Identified</strong> &nbsp;|&nbsp; 🍎 Fruits: ${sum.fruits || 0} &nbsp;|&nbsp; 🥦 Veg: ${sum.vegetables || 0} &nbsp;|&nbsp; 🍱 Foods: ${sum.food || 0}`;
    }
    if (resultSummaryBar) resultSummaryBar.style.display = "block";

    // Multi-Object or Single-Object Cards
    if (multiObjectsContainer) {
      multiObjectsContainer.innerHTML = "";
      if (data.objects && data.objects.length > 0) {
        data.objects.forEach((obj) => {
          const card = createObjectResultCard(obj);
          multiObjectsContainer.appendChild(card);
        });
      }
    }

    // Show after-scan actions & feedback prompt
    if (afterScanActions) afterScanActions.style.display = "flex";
    if (feedbackSection) feedbackSection.style.display = "block";

    // Smooth scroll to result section on mobile
    if (window.innerWidth <= 768 && resultCard) {
      resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  // =========================================================================
  // 7. Dynamic Object Card Creation (With Meter, Nutrition & AI Handoff)
  // =========================================================================
  function createObjectResultCard(obj) {
    const card = document.createElement("div");
    card.className = "detected-item-card";

    let badgeClass = "neutral";
    let badgeText = obj.freshness_status || "Not Available";

    if (obj.freshness === "Fresh") {
      badgeClass = "fresh";
      badgeText = "Fresh ✅";
    } else if (obj.freshness === "Spoiled") {
      badgeClass = "spoiled";
      badgeText = "Spoiled ⚠️";
    }

    const detConfPct = Math.round(obj.detection_confidence * 100);
    const freshConfPct = obj.freshness_confidence ? Math.round(obj.freshness_confidence * 100) : detConfPct;

    card.innerHTML = `
      <div class="item-card-header">
        <div>
          <h3 style="font-size:1.2rem; font-weight:800; color:var(--text);">${obj.item}</h3>
          <span style="font-size:0.8rem; color:var(--muted);">${obj.category}</span>
        </div>
        <span class="status-badge ${badgeClass}" style="font-size:0.8rem; padding:4px 12px;">${badgeText}</span>
      </div>

      <!-- Animated Confidence Meter -->
      <div class="confidence-meter-container">
        <div style="display:flex; justify-content:space-between; font-size:0.825rem; font-weight:600;">
          <span style="color:var(--text-secondary);">Confidence</span>
          <span style="color:var(--primary); font-weight:700;">${obj.freshness_confidence ? `${freshConfPct}%` : `${detConfPct}%`}</span>
        </div>
        <div class="confidence-bar-track">
          <div class="confidence-bar-fill" style="width: ${obj.freshness_confidence ? freshConfPct : detConfPct}%;"></div>
        </div>
      </div>

      ${obj.nutrition && obj.nutrition.calories_per_100g ? `
        <div class="nutrition-grid">
          <div class="nutrition-item">
            <span class="nut-label">Calories</span>
            <span class="nut-val">${obj.nutrition.calories_per_100g} kcal</span>
          </div>
          <div class="nutrition-item">
            <span class="nut-label">Carbs</span>
            <span class="nut-val">${obj.nutrition.carbs_g}g</span>
          </div>
          <div class="nutrition-item">
            <span class="nut-label">Protein</span>
            <span class="nut-val">${obj.nutrition.protein_g}g</span>
          </div>
        </div>
      ` : ''}

      ${obj.stability_warning ? `
        <div class="warning-box" style="margin-top:12px; padding:8px 12px; font-size:0.825rem;">
          <span>💡</span> ${obj.stability_warning}
        </div>
      ` : ''}

      ${obj.storage_tip || obj.safety_guideline ? `
        <details class="guideline-accordion" style="margin-top:14px;">
          <summary>Storage & Safety Guidance</summary>
          ${obj.storage_tip ? `<p style="margin-top:8px;"><strong>Storage:</strong> ${obj.storage_tip}</p>` : ''}
          ${obj.safety_guideline ? `<p style="margin-top:8px;"><strong>Safety:</strong> ${obj.safety_guideline}</p>` : ''}
        </details>
      ` : ''}

      <!-- Direct Ask AI Action for this detected food -->
      <div style="margin-top:16px;">
        <button class="btn secondary touch-friendly ask-ai-obj-btn" type="button" style="width:100%; border-radius:var(--radius-full); font-size:0.875rem;">
          🤖 Ask AI Assistant About This ${obj.item} ➔
        </button>
      </div>
    `;

    // Contextual handoff to /chatbot
    const askBtn = card.querySelector(".ask-ai-obj-btn");
    if (askBtn) {
      askBtn.addEventListener("click", () => {
        const scanCtx = {
          item: obj.item,
          category: obj.category,
          freshness: obj.freshness,
          confidence: obj.freshness_confidence || obj.detection_confidence,
          storage_tip: obj.storage_tip || "",
          safety_guideline: obj.safety_guideline || ""
        };
        sessionStorage.setItem("freshlens_active_scan", JSON.stringify(scanCtx));
        window.location.href = "/chatbot";
      });
    }

    return card;
  }

  // =========================================================================
  // 8. Supervised Quality Feedback Submission
  // =========================================================================
  if (feedbackYes) {
    feedbackYes.addEventListener("click", async () => {
      feedbackYes.disabled = true;
      feedbackNo.disabled = true;
      try {
        await fetch("/api/v1/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_url: lastUploadedImageUrl,
            predicted_class: lastPredictedLabel,
            correct_class: lastPredictedLabel,
            notes: "Confirmed accurate by user."
          })
        });
        if (feedbackResultMsg) {
          feedbackResultMsg.textContent = "✅ Thank you! Accuracy recorded.";
          feedbackResultMsg.style.color = "var(--primary)";
          feedbackResultMsg.style.display = "block";
        }
      } catch (e) {
        console.error("Feedback submission error:", e);
      }
    });
  }

  if (feedbackNo) {
    feedbackNo.addEventListener("click", () => {
      if (correctionForm) correctionForm.style.display = "block";
    });
  }

  if (submitFeedbackBtn) {
    submitFeedbackBtn.addEventListener("click", async () => {
      const chosenItem = correctItemSelect ? correctItemSelect.value : "";
      const chosenCond = correctConditionSelect ? correctConditionSelect.value : "";

      if (!chosenItem || !chosenCond) {
        alert("Please select both the produce item and freshness condition.");
        return;
      }

      submitFeedbackBtn.disabled = true;
      submitFeedbackBtn.textContent = "Submitting...";

      const formattedLabel = `${chosenCond.toLowerCase()}${chosenItem.toLowerCase().replace('_', '')}`;

      try {
        await fetch("/api/v1/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_url: lastUploadedImageUrl,
            predicted_class: lastPredictedLabel,
            correct_class: formattedLabel,
            notes: `User reported correction to ${chosenItem} (${chosenCond})`
          })
        });
        if (correctionForm) correctionForm.style.display = "none";
        if (feedbackResultMsg) {
          feedbackResultMsg.textContent = "✅ Feedback queued for supervised quality review and batch training!";
          feedbackResultMsg.style.color = "var(--primary)";
          feedbackResultMsg.style.display = "block";
        }
      } catch (err) {
        alert("Feedback submission failed.");
      } finally {
        submitFeedbackBtn.disabled = false;
        submitFeedbackBtn.textContent = "Submit Feedback";
      }
    });
  }

  // =========================================================================
  // 9. Scroll Reveal Animations (IntersectionObserver)
  // =========================================================================
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
        }
      });
    }, { threshold: 0.15 });

    document.querySelectorAll(".reveal-on-scroll").forEach((el) => {
      observer.observe(el);
    });
  } else {
    document.querySelectorAll(".reveal-on-scroll").forEach((el) => {
      el.classList.add("is-visible");
    });
  }

  // =========================================================================
  // 10. Page Lifecycle & Tab Visibility Camera Teardown
  // =========================================================================
  window.addEventListener("beforeunload", stopCamera);
  window.addEventListener("pagehide", stopCamera);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopCamera();
    } else if (currentActiveMode === "live" && liveModePanel && liveModePanel.style.display !== "none") {
      startCamera();
    }
  });
});
