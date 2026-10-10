document.addEventListener("DOMContentLoaded", () => {
  const modeUploadBtn = document.getElementById("modeUploadBtn");
  const modeLiveBtn = document.getElementById("modeLiveBtn");
  const uploadModePanel = document.getElementById("uploadModePanel");
  const liveModePanel = document.getElementById("liveModePanel");
  const modeIndicatorBadge = document.getElementById("modeIndicatorBadge");

  const tabButtons = document.querySelectorAll(".tab-btn");

  const uploadArea = document.getElementById("uploadArea");
  const foodImage = document.getElementById("foodImage");
  const previewBox = document.getElementById("previewBox");
  const previewActionsBar = document.getElementById("previewActionsBar");
  const replaceImageBtn = document.getElementById("replaceImageBtn");
  const removeImageBtn = document.getElementById("removeImageBtn");
  const sampleButtons = document.querySelectorAll(".sample-btn");
  const qualityToast = document.getElementById("qualityToast");
  const qualityToastMessage = document.getElementById("qualityToastMessage");

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

  const scanNow = document.getElementById("scanNow");
  const resetScan = document.getElementById("resetScan");
  const progressOverlay = document.getElementById("progressOverlay");
  const progressStepText = document.getElementById("progressStepText");
  const progressBarInner = document.getElementById("progressBarInner");

  const resultCard = document.getElementById("resultCard");
  const scanErrorBox = document.getElementById("scanErrorBox");
  const scanErrorCode = document.getElementById("scanErrorCode");
  const scanErrorMessage = document.getElementById("scanErrorMessage");
  const categoryWarningBox = document.getElementById("categoryWarningBox");
  const categoryWarningMessage = document.getElementById(
    "categoryWarningMessage",
  );
  const annotatedImageBox = document.getElementById("annotatedImageBox");
  const annotatedImage = document.getElementById("annotatedImage");
  const resultSummaryBar = document.getElementById("resultSummaryBar");
  const viewAnalysisPrompt = document.getElementById("viewAnalysisPrompt");
  const summaryText = document.getElementById("summaryText");
  const multiObjectsContainer = document.getElementById(
    "multiObjectsContainer",
  );
  const scanInferenceTime = document.getElementById("scanInferenceTime");
  const afterScanActions = document.getElementById("afterScanActions");
  const scanAnotherBtn = document.getElementById("scanAnotherBtn");

  const feedbackSection = document.getElementById("feedbackSection");
  const feedbackYes = document.getElementById("feedbackYes");
  const feedbackNo = document.getElementById("feedbackNo");
  const correctionForm = document.getElementById("correctionForm");
  const submitFeedbackBtn = document.getElementById("submitFeedbackBtn");
  const correctItemSelect = document.getElementById("correctItemSelect");
  const correctConditionSelect = document.getElementById(
    "correctConditionSelect",
  );
  const feedbackResultMsg = document.getElementById("feedbackResultMsg");

  // Center Vision & Dashboard Elements
  const visionEmptyState = document.getElementById("visionEmptyState");
  const visionPreviewWrap = document.getElementById("visionPreviewWrap");
  const visionOriginalImg = document.getElementById("visionOriginalImg");
  const visionStatusBadge = document.getElementById("visionStatusBadge");
  const resultStatusBadge = document.getElementById("resultStatusBadge");
  const resultPlaceholder = document.getElementById("resultPlaceholder");
  const activeResultView = document.getElementById("activeResultView");
  const primaryFoodName = document.getElementById("primaryFoodName");
  const primaryCategoryPill = document.getElementById("primaryCategoryPill");
  const primaryFreshnessBadge = document.getElementById("primaryFreshnessBadge");
  const primaryFreshnessLabel = document.getElementById("primaryFreshnessLabel");
  const primaryConfidenceValue = document.getElementById("primaryConfidenceValue");
  const primaryConfBarFill = document.getElementById("primaryConfBarFill");
  const primaryConditionLevel = document.getElementById("primaryConditionLevel");
  const primaryInferenceSpeed = document.getElementById("primaryInferenceSpeed");
  const aiSummaryBox = document.getElementById("aiSummaryBox");
  const aiSummaryText = document.getElementById("aiSummaryText");
  const primaryNutritionStrip = document.getElementById("primaryNutritionStrip");
  const primaryCaloriesVal = document.getElementById("primaryCaloriesVal");
  const primaryCarbsVal = document.getElementById("primaryCarbsVal");
  const primaryProteinVal = document.getElementById("primaryProteinVal");
  const primaryGuidanceAccordion = document.getElementById("primaryGuidanceAccordion");
  const primaryGuidanceContent = document.getElementById("primaryGuidanceContent");

  let localStream = null;
  let currentFacingMode = "environment";
  let activeCategory = "All";
  let lastUploadedImageUrl = "";
  let lastPredictedLabel = "";
  let isAnalyzing = false;
  let isCameraStarting = false;
  let currentActiveMode = "upload";
  let activeRequestController = null;

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

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

      if (uploadModePanel) {
        uploadModePanel.style.display = "block";
      }

      if (liveModePanel) {
        liveModePanel.style.display = "none";
      }

      if (modeIndicatorBadge) {
        modeIndicatorBadge.textContent = "Upload Mode";
      }

      return;
    }

    if (mode === "live") {
      if (modeLiveBtn) {
        modeLiveBtn.classList.add("active");
        modeLiveBtn.setAttribute("aria-selected", "true");
      }

      if (modeUploadBtn) {
        modeUploadBtn.classList.remove("active");
        modeUploadBtn.setAttribute("aria-selected", "false");
      }

      if (uploadModePanel) {
        uploadModePanel.style.display = "none";
      }

      if (liveModePanel) {
        liveModePanel.style.display = "block";
      }

      if (modeIndicatorBadge) {
        modeIndicatorBadge.textContent = "Live Camera Mode";
      }

      startCamera();
    }
  }

  if (modeUploadBtn) {
    modeUploadBtn.addEventListener("click", () => {
      switchMode("upload");
    });
  }

  if (modeLiveBtn) {
    modeLiveBtn.addEventListener("click", () => {
      switchMode("live");
    });
  }

  if (fallbackUploadBtn) {
    fallbackUploadBtn.addEventListener("click", () => {
      switchMode("upload");
    });
  }

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const type = btn.getAttribute("data-type");

      activeCategory =
        type === "all"
          ? "All"
          : type
            ? type.charAt(0).toUpperCase() + type.slice(1)
            : "All";
    });
  });

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
      window.isSecureContext === true || protocol === "https:" || isLocalhost;

    const hasMediaDevices = Boolean(
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === "function",
    );

    return {
      hostname,
      protocol,
      isLocalhost,
      isLanIp,
      isSecureContext,
      hasMediaDevices,
    };
  }

  function getCameraSupportStatus() {
    const env = checkEnvironmentSecurity();

    if (env.hasMediaDevices) {
      return {
        supported: true,
        env,
      };
    }

    if (!env.isLocalhost && !env.isSecureContext) {
      return {
        supported: false,
        title: "Secure Connection (HTTPS) Required",
        message: env.isLanIp
          ? "Camera access requires HTTPS on this address. Open the secure HTTPS version of FoodLens-AI, or use localhost on the same machine."
          : "Camera access requires HTTPS on this address. Open the secure HTTPS version of FoodLens-AI.",
        icon: "🔒",
      };
    }

    if (env.isLocalhost) {
      return {
        supported: false,
        title: "Camera Access Notice",
        message:
          "Camera access is available. Please allow camera permission in your browser.",
        icon: "📷",
      };
    }

    return {
      supported: false,
      title: "Camera Not Supported",
      message:
        "Your browser does not support camera access. Please use a modern browser such as Chrome or Edge.",
      icon: "⚠️",
    };
  }

  function parseCameraError(err) {
    const name = err?.name || "";
    const msg = err?.message || "";
    const env = checkEnvironmentSecurity();

    if (name === "NotAllowedError" || name === "PermissionDeniedError") {
      return {
        title: "Camera Permission Blocked",
        message:
          "Camera permission is blocked. Allow camera access in browser settings and click Try Again.",
        icon: "🚫",
      };
    }

    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      return {
        title: "No Camera Detected",
        message: "No camera was detected on this device.",
        icon: "📷",
      };
    }

    if (name === "NotReadableError" || name === "TrackStartError") {
      return {
        title: "Camera In Use",
        message:
          "The camera is already being used by another application or browser tab.",
        icon: "⚠️",
      };
    }

    if (
      name === "OverconstrainedError" ||
      name === "ConstraintNotSatisfiedError"
    ) {
      return {
        title: "Camera Settings Not Supported",
        message:
          "The requested camera settings are not supported. Try again using default settings.",
        icon: "⚙️",
      };
    }

    if (name === "SecurityError") {
      return {
        title: "Secure Connection (HTTPS) Required",
        message: env.isLocalhost
          ? "Camera access is available. Please allow camera permission."
          : "Camera access requires HTTPS on this address.",
        icon: "🔒",
      };
    }

    return {
      title: "Camera Access Unavailable",
      message:
        msg ||
        "Camera permission was denied or the camera is unavailable. You can upload an image instead.",
      icon: "📷",
    };
  }

  function showCameraError(title, message, icon = "📷") {
    stopCamera();

    if (cameraArea) {
      cameraArea.style.display = "none";
    }

    if (cameraErrorState) {
      cameraErrorState.style.display = "block";

      if (cameraErrorTitle) {
        cameraErrorTitle.textContent = title;
      }

      if (cameraErrorMessage) {
        cameraErrorMessage.textContent = message;
      }

      if (cameraErrorIcon) {
        cameraErrorIcon.textContent = icon;
      }
    }

    if (cameraStatusPill) {
      cameraStatusPill.textContent = title;
    }

    if (captureImage) {
      captureImage.disabled = true;
    }

    if (switchCameraBtn) {
      switchCameraBtn.disabled = false;
    }
  }

  async function requestCameraStream(facingMode) {
    const attempts = [
      {
        video: {
          facingMode: {
            ideal: facingMode,
          },
          width: {
            ideal: 1280,
          },
          height: {
            ideal: 720,
          },
        },
        audio: false,
      },
      {
        video: {
          facingMode: {
            ideal: facingMode,
          },
        },
        audio: false,
      },
      {
        video: true,
        audio: false,
      },
    ];

    let lastError = null;

    for (const constraints of attempts) {
      try {
        return await navigator.mediaDevices.getUserMedia(constraints);
      } catch (err) {
        lastError = err;

        if (
          err.name === "NotAllowedError" ||
          err.name === "PermissionDeniedError" ||
          err.name === "SecurityError" ||
          err.name === "NotFoundError" ||
          err.name === "DevicesNotFoundError"
        ) {
          throw err;
        }
      }
    }

    throw lastError;
  }

  async function startCamera() {
    if (isCameraStarting || localStream) {
      return;
    }

    isCameraStarting = true;

    const support = getCameraSupportStatus();

    if (!support.supported) {
      showCameraError(support.title, support.message, support.icon);

      isCameraStarting = false;
      return;
    }

    stopCamera();

    if (cameraErrorState) {
      cameraErrorState.style.display = "none";
    }

    if (cameraArea) {
      cameraArea.style.display = "block";
    }

    if (cameraStatusPill) {
      cameraStatusPill.textContent = "Initializing Camera...";
    }

    if (captureImage) {
      captureImage.disabled = true;
    }

    if (switchCameraBtn) {
      switchCameraBtn.disabled = true;
    }

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
        } catch (err) {
          await new Promise((resolve) => {
            const timer = setTimeout(resolve, 1000);

            webcam.onloadedmetadata = () => {
              clearTimeout(timer);

              webcam
                .play()
                .catch(() => {})
                .finally(resolve);
            };
          });
        }
      }

      const tracks = stream.getVideoTracks();

      if (tracks.length) {
        tracks[0].onended = () => {
          showCameraError(
            "Camera Disconnected",
            "The camera was disconnected or stopped. Click Try Again to reconnect.",
            "📷",
          );
        };
      }

      if (cameraStatusPill) {
        cameraStatusPill.textContent = "Camera Ready - Place food inside frame";
      }

      if (captureImage) {
        captureImage.disabled = false;
      }

      if (switchCameraBtn) {
        switchCameraBtn.disabled = false;
      }
    } catch (err) {
      const errorInfo = parseCameraError(err);

      showCameraError(errorInfo.title, errorInfo.message, errorInfo.icon);
    } finally {
      isCameraStarting = false;
    }
  }

  function stopCamera() {
    if (localStream) {
      try {
        localStream.getTracks().forEach((track) => {
          track.stop();
        });
      } catch (err) {
        console.warn("Camera stop error:", err);
      }

      localStream = null;
    }

    if (webcam) {
      try {
        webcam.pause();
      } catch (_) {}

      webcam.srcObject = null;
    }

    if (captureImage) {
      captureImage.disabled = true;
    }
  }

  if (switchCameraBtn) {
    switchCameraBtn.addEventListener("click", async () => {
      if (isCameraStarting) {
        return;
      }

      currentFacingMode =
        currentFacingMode === "environment" ? "user" : "environment";

      await startCamera();
    });
  }

  if (closeCameraBtn) {
    closeCameraBtn.addEventListener("click", () => {
      switchMode("upload");
    });
  }

  if (retryCameraBtn) {
    retryCameraBtn.addEventListener("click", () => {
      startCamera();
    });
  }

  if (captureImage) {
    captureImage.addEventListener("click", () => {
      if (!localStream || !webcam || isAnalyzing) {
        return;
      }

      const width = webcam.videoWidth || 640;

      const height = webcam.videoHeight || 480;

      if (!width || !height) {
        if (cameraStatusPill) {
          cameraStatusPill.textContent = "Camera is not ready yet";
        }

        return;
      }

      captureImage.disabled = true;

      const canvas = document.createElement("canvas");

      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");

      if (!ctx) {
        captureImage.disabled = false;
        return;
      }

      ctx.drawImage(webcam, 0, 0, width, height);

      canvas.toBlob(
        async (blob) => {
          if (!blob) {
            captureImage.disabled = false;
            return;
          }

          const timestamp = new Date().toISOString().replace(/[:.]/g, "-");

          const file = new File([blob], `camera_scan_${timestamp}.jpg`, {
            type: "image/jpeg",
          });

          const dt = new DataTransfer();

          dt.items.add(file);

          if (foodImage) {
            foodImage.files = dt.files;
          }

          const previewUrl = canvas.toDataURL("image/jpeg", 0.92);

          stopCamera();

          switchMode("upload");

          displayImagePreview(previewUrl, file);

          await executeFoodAnalysis(file);
        },
        "image/jpeg",
        0.92,
      );
    });
  }

  if (uploadArea) {
    ["dragenter", "dragover"].forEach((eventName) => {
      uploadArea.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();

        uploadArea.classList.add("drag-over");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      uploadArea.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();

        uploadArea.classList.remove("drag-over");
      });
    });

    uploadArea.addEventListener("drop", (event) => {
      const files = event.dataTransfer?.files;

      if (files?.length) {
        if (foodImage) {
          foodImage.files = files;
        }

        handleFileSelected(files[0]);
      }
    });
  }

  if (foodImage) {
    foodImage.addEventListener("change", () => {
      const file = foodImage.files?.[0];

      if (file) {
        handleFileSelected(file);
      }
    });
  }

  function handleFileSelected(file) {
    if (!file) {
      return;
    }

    const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];

    if (!validTypes.includes(file.type.toLowerCase())) {
      showQualityToast("Unsupported format. Please upload JPG, PNG, or WEBP.");

      return;
    }

    if (file.size > 16 * 1024 * 1024) {
      showQualityToast("Image file exceeds the 16MB limit.");

      return;
    }

    resetResultsView();

    const reader = new FileReader();

    reader.onload = (event) => {
      displayImagePreview(event.target.result, file);

      runPreflightQualityCheck(event.target.result);
    };

    reader.onerror = () => {
      showQualityToast("Unable to read the selected image.");
    };

    reader.readAsDataURL(file);
  }

  function displayImagePreview(dataUrl, file) {
    if (previewBox) {
      previewBox.innerHTML = `
        <div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
          <img
            src="${dataUrl}"
            alt="Food Preview"
            style="max-height:100px; max-width:100%; border-radius:8px; object-fit:cover; box-shadow:0 2px 10px rgba(0,0,0,0.35);"
          />
          <span style="font-size:0.75rem; color:var(--text-secondary); max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
            ${escapeHtml(file?.name || "Selected Food Photo")}
          </span>
        </div>
      `;
    }

    if (previewActionsBar) {
      previewActionsBar.style.display = "flex";
    }

    // Synchronize Column 2 (Center AI Vision Viewport)
    if (visionOriginalImg) {
      visionOriginalImg.src = dataUrl;
    }
    if (visionPreviewWrap) {
      visionPreviewWrap.style.display = "flex";
    }
    if (annotatedImageBox) {
      annotatedImageBox.style.display = "none";
    }
    if (visionEmptyState) {
      visionEmptyState.style.display = "none";
    }
    if (visionStatusBadge) {
      visionStatusBadge.className = "vision-status-badge";
      const label = visionStatusBadge.querySelector(".status-indicator-label");
      if (label) label.textContent = "Image Ready";
    }
    if (resultStatusBadge) {
      resultStatusBadge.textContent = "Ready to Analyze";
    }
  }

  if (replaceImageBtn && foodImage) {
    replaceImageBtn.addEventListener("click", () => {
      foodImage.click();
    });
  }

  if (removeImageBtn) {
    removeImageBtn.addEventListener("click", () => {
      resetUploadUI();
      resetResultsView();
    });
  }

  function runPreflightQualityCheck(dataUrl) {
    const img = new Image();

    img.onload = () => {
      if (img.width < 40 || img.height < 40) {
        showQualityToast(
          "The image resolution is very small. For best detection, use a clearer photo.",
        );

        return;
      }

      try {
        const canvas = document.createElement("canvas");

        canvas.width = 40;
        canvas.height = 40;

        const ctx = canvas.getContext("2d");

        if (!ctx) {
          return;
        }

        ctx.drawImage(img, 0, 0, 40, 40);

        const data = ctx.getImageData(0, 0, 40, 40).data;

        let total = 0;

        for (let i = 0; i < data.length; i += 4) {
          total += (data[i] + data[i + 1] + data[i + 2]) / 3;
        }

        const avgBrightness = total / (data.length / 4);

        if (avgBrightness < 16) {
          showQualityToast(
            "The photo appears very dark. Consider better lighting.",
          );
        } else if (qualityToast) {
          qualityToast.style.display = "none";
        }
      } catch (_) {}
    };

    img.src = dataUrl;
  }

  function showQualityToast(message) {
    if (qualityToastMessage) {
      qualityToastMessage.textContent = message;
    }

    if (qualityToast) {
      qualityToast.style.display = "flex";
    }
  }

  sampleButtons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      switchMode("upload");
      resetResultsView();

      const imageUrl = btn.getAttribute("data-image");

      if (!imageUrl) {
        showQualityToast("Sample image is unavailable.");

        return;
      }

      try {
        const response = await fetch(imageUrl);

        if (!response.ok) {
          throw new Error(`Sample image HTTP ${response.status}`);
        }

        const blob = await response.blob();

        const file = new File(
          [blob],
          imageUrl.split("/").pop() || "sample.jpg",
          {
            type: blob.type || "image/jpeg",
          },
        );

        const dt = new DataTransfer();

        dt.items.add(file);

        if (foodImage) {
          foodImage.files = dt.files;
        }

        const previewUrl = URL.createObjectURL(blob);

        displayImagePreview(previewUrl, file);

        await executeFoodAnalysis(file);

        URL.revokeObjectURL(previewUrl);
      } catch (error) {
        console.error("Sample image error:", error);

        showQualityToast(
          "Unable to load the sample image. Please upload an image manually.",
        );
      }
    });
  });

  function resetResultsView() {
    if (scanErrorBox) {
      scanErrorBox.style.display = "none";
    }

    if (categoryWarningBox) {
      categoryWarningBox.style.display = "none";
    }

    if (annotatedImageBox) {
      annotatedImageBox.style.display = "none";
    }

    if (resultSummaryBar) {
      resultSummaryBar.style.display = "none";
    }

    if (viewAnalysisPrompt) {
      viewAnalysisPrompt.style.display = "none";
    }

    if (afterScanActions) {
      afterScanActions.style.display = "none";
    }

    if (feedbackSection) {
      feedbackSection.style.display = "none";
    }

    if (correctionForm) {
      correctionForm.style.display = "none";
    }

    if (feedbackResultMsg) {
      feedbackResultMsg.style.display = "none";
    }

    if (scanInferenceTime) {
      scanInferenceTime.textContent = "--";
    }

    if (activeResultView) {
      activeResultView.style.display = "none";
    }

    if (primaryNutritionStrip) {
      primaryNutritionStrip.style.display = "none";
    }

    if (primaryGuidanceAccordion) {
      primaryGuidanceAccordion.style.display = "none";
    }

    if (resultPlaceholder) {
      resultPlaceholder.style.display = "flex";
    }

    if (resultStatusBadge) {
      resultStatusBadge.textContent = "Awaiting Scan";
    }

    if (multiObjectsContainer) {
      multiObjectsContainer.innerHTML = "";
    }
  }

  function resetUploadUI() {
    if (foodImage) {
      foodImage.value = "";
    }

    if (previewBox) {
      previewBox.innerHTML = `
        <div class="upload-icon-circle">
          <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"></path>
            <path d="M12 12v9"></path>
            <path d="m8 16 4-4 4 4"></path>
          </svg>
        </div>
        <div class="dropzone-text-group">
          <span class="dropzone-main-text">Drag & drop food image here</span>
          <span class="dropzone-sub-text">or <strong class="browse-link">browse files</strong> from device</span>
        </div>
        <div class="format-pill-row">
          <span class="format-pill">JPG</span>
          <span class="format-pill">PNG</span>
          <span class="format-pill">WEBP</span>
          <span class="format-pill">Max 16MB</span>
        </div>
      `;
    }

    if (previewActionsBar) {
      previewActionsBar.style.display = "none";
    }

    if (qualityToast) {
      qualityToast.style.display = "none";
    }

    if (visionPreviewWrap) {
      visionPreviewWrap.style.display = "none";
    }

    if (annotatedImageBox) {
      annotatedImageBox.style.display = "none";
    }

    if (visionEmptyState) {
      visionEmptyState.style.display = "flex";
    }

    if (visionStatusBadge) {
      visionStatusBadge.className = "vision-status-badge";
      const label = visionStatusBadge.querySelector(".status-indicator-label");
      if (label) label.textContent = "Standby";
    }

    if (resultStatusBadge) {
      resultStatusBadge.textContent = "Awaiting Scan";
    }
  }

  if (resetScan) {
    resetScan.addEventListener("click", () => {
      if (activeRequestController) {
        try {
          activeRequestController.abort();
        } catch (_) {}
      }

      stopCamera();
      resetUploadUI();
      switchMode("upload");
      resetResultsView();
    });
  }

  if (scanAnotherBtn) {
    scanAnotherBtn.addEventListener("click", () => {
      resetUploadUI();
      resetResultsView();

      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    });
  }

  async function optimizeImageForUpload(file) {
    if (!file || !file.type.startsWith("image/")) {
      return file;
    }

    const MAX_SIZE = 1600;
    const QUALITY = 0.82;
    const MAX_DIRECT_SIZE = 2 * 1024 * 1024;

    if (file.size <= MAX_DIRECT_SIZE) {
      return file;
    }

    return new Promise((resolve) => {
      const reader = new FileReader();

      reader.onload = (event) => {
        const img = new Image();

        img.onload = () => {
          let width = img.width;

          let height = img.height;

          if (width <= MAX_SIZE && height <= MAX_SIZE) {
            const canvas = document.createElement("canvas");

            canvas.width = width;

            canvas.height = height;

            const ctx = canvas.getContext("2d");

            if (!ctx) {
              resolve(file);
              return;
            }

            ctx.drawImage(img, 0, 0, width, height);

            canvas.toBlob(
              (blob) => {
                if (!blob) {
                  resolve(file);
                  return;
                }

                resolve(
                  new File([blob], "foodlens_scan.jpg", {
                    type: "image/jpeg",
                    lastModified: Date.now(),
                  }),
                );
              },
              "image/jpeg",
              QUALITY,
            );

            return;
          }

          const scale = Math.min(MAX_SIZE / width, MAX_SIZE / height, 1);

          width = Math.round(width * scale);

          height = Math.round(height * scale);

          const canvas = document.createElement("canvas");

          canvas.width = width;

          canvas.height = height;

          const ctx = canvas.getContext("2d");

          if (!ctx) {
            resolve(file);
            return;
          }

          ctx.drawImage(img, 0, 0, width, height);

          canvas.toBlob(
            (blob) => {
              if (!blob) {
                resolve(file);
                return;
              }

              resolve(
                new File([blob], "foodlens_scan.jpg", {
                  type: "image/jpeg",
                  lastModified: Date.now(),
                }),
              );
            },
            "image/jpeg",
            QUALITY,
          );
        };

        img.onerror = () => {
          resolve(file);
        };

        img.src = event.target.result;
      };

      reader.onerror = () => {
        resolve(file);
      };

      reader.readAsDataURL(file);
    });
  }

  async function executeFoodAnalysis(file) {
    if (isAnalyzing) {
      return;
    }

    if (!file) {
      alert("Please select or capture a food image first.");

      return;
    }

    if (!file.type.startsWith("image/")) {
      handleScanError({
        status: "invalid_file",
        error: {
          code: "INVALID_FILE",
          message: "Please select a valid image file.",
        },
      });

      return;
    }

    resetResultsView();

    isAnalyzing = true;

    if (visionStatusBadge) {
      visionStatusBadge.className = "vision-status-badge scanning";
      const label = visionStatusBadge.querySelector(".status-indicator-label");
      if (label) label.textContent = "Analyzing...";
    }

    if (resultStatusBadge) {
      resultStatusBadge.textContent = "Processing...";
    }

    if (scanNow) {
      scanNow.disabled = true;
      scanNow.innerHTML = "<span>⏳ Analyzing...</span>";
    }

    if (captureImage) {
      captureImage.disabled = true;
    }

    if (progressOverlay) {
      progressOverlay.style.display = "flex";
    }

    if (progressBarInner) {
      progressBarInner.style.width = "10%";
    }

    if (progressStepText) {
      progressStepText.textContent = "Preparing image for AI analysis...";
    }

    let step1Timer = null;
    let step2Timer = null;
    let step3Timer = null;
    let timeoutId = null;

    try {
      if (progressBarInner) {
        progressBarInner.style.width = "20%";
      }

      if (progressStepText) {
        progressStepText.textContent = "Optimizing image...";
      }

      const originalSize = file.size;

      const optimizedFile = await optimizeImageForUpload(file);

      console.log("FoodLens image upload:", {
        originalName: file.name,
        originalSizeKB: Math.round(originalSize / 1024),
        optimizedName: optimizedFile.name,
        optimizedSizeKB: Math.round(optimizedFile.size / 1024),
      });

      if (progressBarInner) {
        progressBarInner.style.width = "35%";
      }

      if (progressStepText) {
        progressStepText.textContent = "Sending image to FoodLens-AI...";
      }

      const formData = new FormData();

      formData.append(
        "file",
        optimizedFile,
        optimizedFile.name || "foodlens_scan.jpg",
      );

      formData.append("selected_category", activeCategory || "All");

      step1Timer = setTimeout(() => {
        if (progressBarInner) {
          progressBarInner.style.width = "50%";
        }

        if (progressStepText) {
          progressStepText.textContent = "Detecting food item...";
        }
      }, 1500);

      step2Timer = setTimeout(() => {
        if (progressBarInner) {
          progressBarInner.style.width = "70%";
        }

        if (progressStepText) {
          progressStepText.textContent = "Running AI freshness analysis...";
        }
      }, 5000);

      step3Timer = setTimeout(() => {
        if (progressBarInner) {
          progressBarInner.style.width = "82%";
        }

        if (progressStepText) {
          progressStepText.textContent = "AI model is processing the image...";
        }
      }, 15000);

      activeRequestController = new AbortController();

      const REQUEST_TIMEOUT = 90000;

      timeoutId = setTimeout(() => {
        if (activeRequestController) {
          activeRequestController.abort();
        }
      }, REQUEST_TIMEOUT);

      let response;

      try {
        response = await fetch("/api/v1/predict", {
          method: "POST",
          body: formData,
          headers: {
            Accept: "application/json",
          },
          credentials: "same-origin",
          cache: "no-store",
          signal: activeRequestController.signal,
        });
      } finally {
        if (timeoutId) {
          clearTimeout(timeoutId);

          timeoutId = null;
        }

        activeRequestController = null;
      }

      if (step1Timer) {
        clearTimeout(step1Timer);

        step1Timer = null;
      }

      if (step2Timer) {
        clearTimeout(step2Timer);

        step2Timer = null;
      }

      if (step3Timer) {
        clearTimeout(step3Timer);

        step3Timer = null;
      }

      if (progressBarInner) {
        progressBarInner.style.width = "90%";
      }

      let data = null;

      const contentType = response.headers.get("content-type") || "";

      if (contentType.includes("application/json")) {
        try {
          data = await response.json();
        } catch (jsonError) {
          console.error("FoodLens JSON parse error:", jsonError);

          data = {
            success: false,
            error: {
              code: `HTTP_${response.status}`,
              message: "The server returned an invalid response.",
            },
          };
        }
      } else {
        const text = await response.text();

        data = {
          success: false,
          error: {
            code: `HTTP_${response.status}`,
            message: text || `Server returned HTTP ${response.status}.`,
          },
        };
      }

      console.log("FoodLens API response:", {
        status: response.status,
        ok: response.ok,
        data,
      });

      if (response.status === 504) {
        if (progressStepText) {
          progressStepText.textContent = "Server took too long to respond.";
        }

        if (progressBarInner) {
          progressBarInner.style.width = "100%";
        }

        await new Promise((resolve) => setTimeout(resolve, 300));

        if (progressOverlay) {
          progressOverlay.style.display = "none";
        }

        handleScanError({
          status: "prediction_timeout",
          error: {
            code: "PREDICTION_TIMEOUT",
            message:
              "Food analysis took too long on the server. Please try again with a clear food image.",
          },
        });

        return;
      }

      if (response.status === 502) {
        if (progressOverlay) {
          progressOverlay.style.display = "none";
        }

        handleScanError({
          status: "prediction_unavailable",
          error: {
            code: "PREDICTION_UNAVAILABLE",
            message:
              "The FoodLens-AI server is temporarily unavailable. Please wait a few seconds and try again.",
          },
        });

        return;
      }

      if (response.status === 503) {
        if (progressOverlay) {
          progressOverlay.style.display = "none";
        }

        handleScanError({
          status: "prediction_unavailable",
          error: {
            code: "PREDICTION_UNAVAILABLE",
            message:
              "The AI service is starting up. Please wait a few seconds and try again.",
          },
        });

        return;
      }

      if (!response.ok || data?.success === false) {
        if (progressOverlay) {
          progressOverlay.style.display = "none";
        }

        handleScanError(
          data || {
            error: {
              code: `HTTP_${response.status}`,
              message: `Server returned HTTP ${response.status}.`,
            },
          },
        );

        return;
      }

      if (progressBarInner) {
        progressBarInner.style.width = "100%";
      }

      if (progressStepText) {
        progressStepText.textContent = "Analysis complete.";
      }

      await new Promise((resolve) => setTimeout(resolve, 250));

      if (progressOverlay) {
        progressOverlay.style.display = "none";
      }

      renderScanSuccess(data);

      if (scanNow) {
        scanNow.innerHTML = "<span>✅ Analysis Complete</span>";

        setTimeout(() => {
          if (!isAnalyzing && scanNow) {
            scanNow.innerHTML = "<span>🔍 Analyze Food</span>";
          }
        }, 2000);
      }
    } catch (error) {
      if (step1Timer) {
        clearTimeout(step1Timer);
      }

      if (step2Timer) {
        clearTimeout(step2Timer);
      }

      if (step3Timer) {
        clearTimeout(step3Timer);
      }

      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      activeRequestController = null;

      console.error("FoodLens prediction request failed:", error);

      if (progressOverlay) {
        progressOverlay.style.display = "none";
      }

      if (error?.name === "AbortError") {
        handleScanError({
          status: "prediction_timeout",
          error: {
            code: "PREDICTION_TIMEOUT",
            message:
              "The AI analysis is taking too long. Please try again with a smaller or clearer food image.",
          },
        });
      } else {
        handleScanError({
          status: "network_error",
          error: {
            code: "NETWORK_ERROR",
            message:
              "Unable to connect to the FoodLens-AI prediction service. Please try again.",
          },
        });
      }
    } finally {
      isAnalyzing = false;

      if (scanNow) {
        scanNow.disabled = false;
        scanNow.innerHTML = "<span>🔍 Analyze Food</span>";
      }

      if (captureImage) {
        captureImage.disabled = false;
      }
    }
  }

  if (scanNow) {
    scanNow.addEventListener("click", async (event) => {
      event.preventDefault();

      if (isAnalyzing) {
        return;
      }

      const file = foodImage?.files?.[0];

      if (!file) {
        alert("Please select or capture a food image first.");

        return;
      }

      await executeFoodAnalysis(file);
    });
  }

  function handleScanError(data = {}) {
    const status = data.status || "";

    const err = data.error || {
      code: "SCAN_FAILED",
      message: data.message || "Scan could not be completed.",
    };

    if (annotatedImageBox) {
      annotatedImageBox.style.display = "none";
    }

    if (resultSummaryBar) {
      resultSummaryBar.style.display = "none";
    }

    if (viewAnalysisPrompt) {
      viewAnalysisPrompt.style.display = "none";
    }

    if (categoryWarningBox) {
      categoryWarningBox.style.display = "none";
    }

    if (afterScanActions) {
      afterScanActions.style.display = "none";
    }

    if (feedbackSection) {
      feedbackSection.style.display = "none";
    }

    if (activeResultView) {
      activeResultView.style.display = "none";
    }

    if (resultPlaceholder) {
      resultPlaceholder.style.display = "none";
    }

    if (visionStatusBadge) {
      visionStatusBadge.className = "vision-status-badge";
      const label = visionStatusBadge.querySelector(".status-indicator-label");
      if (label) label.textContent = "Notice";
    }

    if (resultStatusBadge) {
      resultStatusBadge.textContent = "Notice";
    }

    if (status === "poor_image_quality") {
      if (scanErrorCode) {
        scanErrorCode.textContent = "Poor Image Quality";
      }

      if (scanErrorMessage) {
        scanErrorMessage.textContent =
          "Image quality is too low for reliable analysis. Please upload or capture a clearer food photo.";
      }

      if (scanErrorBox) {
        scanErrorBox.style.display = "flex";
      }
    } else if (
      status === "prediction_timeout" ||
      status === "prediction_unavailable"
    ) {
      if (scanErrorCode) {
        scanErrorCode.textContent = err.code || "Service Notice";
      }

      if (scanErrorMessage) {
        scanErrorMessage.textContent =
          err.message ||
          "FoodLens-AI server is taking too long to respond. Please try again.";
      }

      if (scanErrorBox) {
        scanErrorBox.style.display = "flex";
      }
    } else {
      const noFood =
        status === "no_food" ||
        status === "no_supported_food_detected" ||
        (data.food_detected === false && status !== "low_confidence");

      if (noFood) {
        if (scanErrorBox) {
          scanErrorBox.style.display = "none";
        }

        if (multiObjectsContainer) {
          multiObjectsContainer.innerHTML = `
            <div class="rejection-card no-food-card"
              style="text-align:center; padding:28px 16px; background:var(--bg-alt); border-radius:12px; border:1px solid var(--border);">

              <span style="font-size:2.5rem; display:block; margin-bottom:8px;">🔍</span>

              <h3 style="font-size:1.15rem; font-weight:800; color:var(--text); margin-bottom:6px;">
                No Food Detected
              </h3>

              <p style="font-size:0.85rem; color:var(--text-secondary); max-width:380px; margin:0 auto 10px; line-height:1.5;">
                ${escapeHtml(
                  data.message ||
                    err.message ||
                    "No supported food detected in this image.",
                )}
              </p>

              <p style="font-size:0.78rem; color:var(--muted); max-width:340px; margin:0 auto 16px;">
                Please point the camera at a fruit, vegetable, or prepared food item and try again.
              </p>

              <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
                <button type="button" class="btn primary btn-sm" id="rejectionTryAgainBtn">
                  📷 Try Another Image
                </button>

                <button type="button" class="btn secondary btn-sm" id="rejectionUploadBtn">
                  🖼️ Upload Image
                </button>
              </div>
            </div>
          `;

          const tryAgainBtn = document.getElementById("rejectionTryAgainBtn");

          if (tryAgainBtn) {
            tryAgainBtn.addEventListener("click", () => {
              switchMode("live");
            });
          }

          const uploadBtn = document.getElementById("rejectionUploadBtn");

          if (uploadBtn) {
            uploadBtn.addEventListener("click", () => {
              switchMode("upload");

              setTimeout(() => {
                foodImage?.click();
              }, 100);
            });
          }
        }
      } else if (status === "low_confidence") {
        if (scanErrorBox) {
          scanErrorBox.style.display = "none";
        }

        if (multiObjectsContainer) {
          multiObjectsContainer.innerHTML = `
            <div class="rejection-card low-confidence-card"
              style="text-align:center; padding:28px 16px; background:var(--bg-alt); border-radius:12px; border:1px solid var(--border);">

              <span style="font-size:2.5rem; display:block; margin-bottom:8px;">⚠️</span>

              <h3 style="font-size:1.15rem; font-weight:800; color:var(--text); margin-bottom:6px;">
                Low Confidence
              </h3>

              <p style="font-size:0.85rem; color:var(--text-secondary); max-width:380px; margin:0 auto 12px; line-height:1.5;">
                ${escapeHtml(
                  data.message ||
                    err.message ||
                    "The food could not be identified confidently.",
                )}
              </p>

              <div style="text-align:left; max-width:320px; margin:0 auto 18px; padding:10px 14px; background:var(--card); border-radius:8px; border:1px solid var(--border);">
                <strong style="font-size:0.8rem; color:var(--text); display:block; margin-bottom:4px;">
                  Tips for Better Scan:
                </strong>

                <ul style="font-size:0.75rem; color:var(--text-secondary); margin:0; padding-left:16px; line-height:1.5;">
                  <li>Center the produce item</li>
                  <li>Use bright, even lighting</li>
                  <li>Avoid blurry camera angles</li>
                </ul>
              </div>

              <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
                <button type="button" class="btn primary btn-sm" id="lowConfTryAgainBtn">
                  📷 Try Again
                </button>

                <button type="button" class="btn secondary btn-sm" id="lowConfUploadBtn">
                  🖼️ Upload Image
                </button>
              </div>
            </div>
          `;

          const tryAgainBtn = document.getElementById("lowConfTryAgainBtn");

          if (tryAgainBtn) {
            tryAgainBtn.addEventListener("click", () => {
              switchMode("live");
            });
          }

          const uploadBtn = document.getElementById("lowConfUploadBtn");

          if (uploadBtn) {
            uploadBtn.addEventListener("click", () => {
              switchMode("upload");

              setTimeout(() => {
                foodImage?.click();
              }, 100);
            });
          }
        }
      } else {
        if (scanErrorCode) {
          scanErrorCode.textContent = err.code || "Analysis Notice";
        }

        if (scanErrorMessage) {
          scanErrorMessage.textContent =
            err.message || "Unable to complete the analysis. Please upload a clearer food photo.";
        }

        if (scanErrorBox) {
          scanErrorBox.style.display = "flex";
        }
      }
    }

    if (window.innerWidth <= 768 && resultCard) {
      resultCard.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }

  function renderScanSuccess(data) {
    console.log("FoodLens successful analysis:", data);

    try {
      sessionStorage.setItem("foodlens_latest_analysis", JSON.stringify(data));
      sessionStorage.setItem("freshlens_latest_analysis", JSON.stringify(data));
    } catch (err) {
      console.warn("Could not save analysis to sessionStorage:", err);
    }

    lastUploadedImageUrl = data.image_url || "";
    lastPredictedLabel = data.label || "";

    const inferenceMs = Number(data.inference_time_ms) || 0;
    const latencyFormatted = inferenceMs > 0 ? (inferenceMs / 1000).toFixed(2) + "s" : "--";

    if (scanInferenceTime) {
      scanInferenceTime.textContent = latencyFormatted;
    }

    // 1. Column 2 (Center AI Vision Viewport) Update
    if (visionStatusBadge) {
      visionStatusBadge.className = "vision-status-badge completed";
      const label = visionStatusBadge.querySelector(".status-indicator-label");
      if (label) label.textContent = "Food Detected";
    }

    if (data.annotated_image_url && annotatedImage) {
      annotatedImage.src = data.annotated_image_url;
      if (annotatedImageBox) {
        annotatedImageBox.style.display = "flex";
      }
      if (visionPreviewWrap) {
        visionPreviewWrap.style.display = "none";
      }
    } else {
      if (visionPreviewWrap) {
        visionPreviewWrap.style.display = "flex";
      }
      if (annotatedImageBox) {
        annotatedImageBox.style.display = "none";
      }
    }

    if (visionEmptyState) {
      visionEmptyState.style.display = "none";
    }

    // 2. Column 3 (Results Panel) Update
    if (resultStatusBadge) {
      resultStatusBadge.textContent = "Complete";
    }

    if (resultPlaceholder) {
      resultPlaceholder.style.display = "none";
    }

    if (scanErrorBox) {
      scanErrorBox.style.display = "none";
    }

    if (activeResultView) {
      activeResultView.style.display = "block";
    }

    // Extract primary prediction details
    const primaryObj = Array.isArray(data.objects) && data.objects.length > 0 ? data.objects[0] : {};
    const foodName = data.food_name || primaryObj.item || data.label || "Detected Food";
    const category = data.category || primaryObj.category || "Produce";
    const condition = data.condition || primaryObj.freshness_status || primaryObj.freshness || "Fresh";

    // Set Food Title and Category
    if (primaryFoodName) {
      primaryFoodName.textContent = foodName;
    }
    if (primaryCategoryPill) {
      primaryCategoryPill.textContent = category;
    }

    // Set FRESH / SPOILED Hero Badge
    if (primaryFreshnessBadge) {
      primaryFreshnessBadge.className = "freshness-main-badge";
      const isFresh = condition.toLowerCase().includes("fresh");
      const isSpoiled = condition.toLowerCase().includes("spoil");

      if (isFresh) {
        primaryFreshnessBadge.classList.add("fresh");
        if (primaryFreshnessLabel) primaryFreshnessLabel.textContent = "FRESH";
      } else if (isSpoiled) {
        primaryFreshnessBadge.classList.add("spoiled");
        if (primaryFreshnessLabel) primaryFreshnessLabel.textContent = "SPOILED";
      } else {
        primaryFreshnessBadge.classList.add("neutral");
        if (primaryFreshnessLabel) primaryFreshnessLabel.textContent = condition.toUpperCase();
      }
    }

    // Set Confidence Gauge & Score
    let conf = 0;
    if (data.confidence != null && Number.isFinite(Number(data.confidence))) {
      conf = Math.round(Number(data.confidence));
    } else if (primaryObj.freshness_confidence != null) {
      conf = Math.round(Number(primaryObj.freshness_confidence) * 100);
    } else if (primaryObj.detection_confidence != null) {
      conf = Math.round(Number(primaryObj.detection_confidence) * 100);
    }
    conf = Math.max(0, Math.min(100, conf));

    if (primaryConfidenceValue) {
      primaryConfidenceValue.textContent = `${conf}%`;
    }
    if (primaryConfBarFill) {
      primaryConfBarFill.style.width = `${conf}%`;
    }

    // Set Condition Level text
    if (primaryConditionLevel) {
      const isFresh = condition.toLowerCase().includes("fresh");
      const isSpoiled = condition.toLowerCase().includes("spoil");
      primaryConditionLevel.textContent = isFresh ? "High Quality" : (isSpoiled ? "Spoiled / Unsafe" : condition);
      primaryConditionLevel.style.color = isFresh ? "var(--fresh)" : (isSpoiled ? "var(--spoiled)" : "var(--text)");
    }

    // Set Latency text
    if (primaryInferenceSpeed) {
      primaryInferenceSpeed.textContent = latencyFormatted;
    }

    // Category Warning Box
    if (data.warning && categoryWarningMessage && categoryWarningBox) {
      categoryWarningMessage.textContent = data.warning;
      categoryWarningBox.style.display = "flex";
    } else if (categoryWarningBox) {
      categoryWarningBox.style.display = "none";
    }

    // AI Summary Box
    if (aiSummaryText) {
      aiSummaryText.textContent = data.message || "FoodLens-AI identified the uploaded food and analyzed its visible freshness condition.";
    }

    // Summary Statistics Bar
    const summary = data.summary || {};
    const totalObjects = Number(summary.total_objects) || (Array.isArray(data.objects) ? data.objects.length : 1);

    if (summaryText) {
      summaryText.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
          <span><strong>${totalObjects} Item${totalObjects > 1 ? "s" : ""} Identified</strong></span>
          <span style="color:var(--text-secondary); font-size:0.75rem;">
            🍎 Fruits: ${Number(summary.fruits) || 0} &bull; 🥦 Veg: ${Number(summary.vegetables) || 0} &bull; ✅ Fresh: ${Number(summary.fresh) || 0} &bull; ⚠️ Spoiled: ${Number(summary.spoiled) || 0}
          </span>
        </div>
      `;
    }
    if (resultSummaryBar) {
      resultSummaryBar.style.display = totalObjects > 1 ? "block" : "none";
    }

    // Set Primary Nutrition Strip
    const nut = primaryObj.nutrition || {};
    if (primaryNutritionStrip) {
      if (nut.calories_per_100g != null) {
        if (primaryCaloriesVal) primaryCaloriesVal.textContent = `${escapeHtml(nut.calories_per_100g)} kcal`;
        if (primaryCarbsVal) primaryCarbsVal.textContent = `${escapeHtml(nut.carbs_g ?? "-")}g`;
        if (primaryProteinVal) primaryProteinVal.textContent = `${escapeHtml(nut.protein_g ?? "-")}g`;
        primaryNutritionStrip.style.display = "grid";
      } else {
        primaryNutritionStrip.style.display = "none";
      }
    }

    // Set Primary Storage & Safety Guidance Accordion
    if (primaryGuidanceAccordion && primaryGuidanceContent) {
      const storageTip = primaryObj.storage_tip
        ? `<p style="margin-bottom:6px; font-size:0.78rem;"><strong>Storage:</strong> ${escapeHtml(primaryObj.storage_tip)}</p>`
        : "";
      const safetyGuideline = primaryObj.safety_guideline
        ? `<p style="margin:0; font-size:0.78rem;"><strong>Safety:</strong> ${escapeHtml(primaryObj.safety_guideline)}</p>`
        : "";
      if (storageTip || safetyGuideline) {
        primaryGuidanceContent.innerHTML = storageTip + safetyGuideline;
        primaryGuidanceAccordion.style.display = "block";
      } else {
        primaryGuidanceAccordion.style.display = "none";
      }
    }

    // Multi Objects List (Only display additional items when 2 or more objects exist)
    if (multiObjectsContainer) {
      multiObjectsContainer.innerHTML = "";
      if (Array.isArray(data.objects) && data.objects.length > 1) {
        data.objects.slice(1).forEach((obj) => {
          multiObjectsContainer.appendChild(createObjectResultCard(obj));
        });
      }
    }

    if (viewAnalysisPrompt) {
      viewAnalysisPrompt.style.display = "block";
    }

    if (afterScanActions) {
      afterScanActions.style.display = "flex";
    }

    if (feedbackSection) {
      feedbackSection.style.display = "block";
    }

    if (window.innerWidth <= 768 && resultCard) {
      resultCard.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }

  function createObjectResultCard(obj = {}) {
    const row = document.createElement("div");
    row.className = "detected-object-compact-row";

    const item = escapeHtml(obj.item || "Unknown Food");
    const category = escapeHtml(obj.category || "Produce");
    const freshness = obj.freshness || "Fresh";

    let badgeClass = "fresh";
    let badgeText = obj.freshness_status || "Fresh";

    if (freshness.toLowerCase().includes("spoil")) {
      badgeClass = "spoiled";
      badgeText = "Spoiled";
    } else if (!freshness.toLowerCase().includes("fresh")) {
      badgeClass = "neutral";
      badgeText = freshness;
    }

    const detectionConfidence = Number(obj.detection_confidence);
    const freshnessConfidence = Number(obj.freshness_confidence);
    const conf = Number.isFinite(freshnessConfidence)
      ? Math.round(freshnessConfidence * 100)
      : (Number.isFinite(detectionConfidence) ? Math.round(detectionConfidence * 100) : 95);

    row.innerHTML = `
      <div class="compact-obj-left">
        <span class="compact-obj-name">${item}</span>
        <span class="compact-obj-cat">${category}</span>
      </div>
      <div class="compact-obj-right">
        <span class="compact-obj-badge ${badgeClass}">● ${escapeHtml(badgeText)}</span>
        <span class="compact-obj-conf">${conf}%</span>
      </div>
    `;

    return row;
  }

  if (feedbackYes) {
    feedbackYes.addEventListener("click", async () => {
      feedbackYes.disabled = true;

      if (feedbackNo) {
        feedbackNo.disabled = true;
      }

      try {
        const response = await fetch("/api/v1/feedback", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image_url: lastUploadedImageUrl,
            predicted_class: lastPredictedLabel,
            correct_class: lastPredictedLabel,
            notes: "Confirmed accurate by user.",
          }),
        });

        if (!response.ok) {
          throw new Error("Feedback request failed");
        }

        if (feedbackResultMsg) {
          feedbackResultMsg.textContent = "✅ Thank you! Accuracy recorded.";

          feedbackResultMsg.style.color = "var(--primary)";

          feedbackResultMsg.style.display = "block";
        }
      } catch (error) {
        console.error("Feedback submission error:", error);
      }
    });
  }

  if (feedbackNo) {
    feedbackNo.addEventListener("click", () => {
      if (correctionForm) {
        correctionForm.style.display = "block";
      }
    });
  }

  if (submitFeedbackBtn) {
    submitFeedbackBtn.addEventListener("click", async () => {
      const chosenItem = correctItemSelect?.value || "";

      const chosenCondition = correctConditionSelect?.value || "";

      if (!chosenItem || !chosenCondition) {
        alert("Please select both the produce item and freshness condition.");

        return;
      }

      submitFeedbackBtn.disabled = true;

      submitFeedbackBtn.textContent = "Submitting...";

      const formattedLabel = `${chosenCondition.toLowerCase()}${chosenItem
        .toLowerCase()
        .replace(/_/g, "")}`;

      try {
        const response = await fetch("/api/v1/feedback", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image_url: lastUploadedImageUrl,
            predicted_class: lastPredictedLabel,
            correct_class: formattedLabel,
            notes: `User reported correction to ${chosenItem} (${chosenCondition})`,
          }),
        });

        if (!response.ok) {
          throw new Error("Feedback submission failed");
        }

        if (correctionForm) {
          correctionForm.style.display = "none";
        }

        if (feedbackResultMsg) {
          feedbackResultMsg.textContent =
            "✅ Feedback queued for supervised quality review and batch training.";

          feedbackResultMsg.style.color = "var(--primary)";

          feedbackResultMsg.style.display = "block";
        }
      } catch (error) {
        console.error("Feedback error:", error);

        alert("Feedback submission failed.");
      } finally {
        submitFeedbackBtn.disabled = false;

        submitFeedbackBtn.textContent = "Submit Feedback";
      }
    });
  }

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
          }
        });
      },
      {
        threshold: 0.15,
      },
    );

    document.querySelectorAll(".reveal-on-scroll").forEach((element) => {
      observer.observe(element);
    });
  } else {
    document.querySelectorAll(".reveal-on-scroll").forEach((element) => {
      element.classList.add("is-visible");
    });
  }

  window.addEventListener("beforeunload", stopCamera);

  window.addEventListener("pagehide", stopCamera);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopCamera();
    } else if (
      currentActiveMode === "live" &&
      liveModePanel &&
      liveModePanel.style.display !== "none"
    ) {
      startCamera();
    }
  });

  switchMode("upload");
});
