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

  let localStream = null;
  let currentFacingMode = "environment";
  let activeCategory = "All";
  let lastUploadedImageUrl = "";
  let lastPredictedLabel = "";
  let isAnalyzing = false;
  let isCameraStarting = false;
  let currentActiveMode = "upload";

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

      if (uploadModePanel) uploadModePanel.style.display = "block";
      if (liveModePanel) liveModePanel.style.display = "none";
      if (modeIndicatorBadge) modeIndicatorBadge.textContent = "Upload Mode";
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

      if (uploadModePanel) uploadModePanel.style.display = "none";
      if (liveModePanel) liveModePanel.style.display = "block";
      if (modeIndicatorBadge)
        modeIndicatorBadge.textContent = "Live Camera Mode";

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
      return { supported: true, env };
    }

    if (!env.isLocalhost && !env.isSecureContext) {
      return {
        supported: false,
        title: "Secure Connection (HTTPS) Required",
        message: env.isLanIp
          ? "Camera access requires HTTPS on this address. Open the secure HTTPS version of FreshLens AI, or use localhost on the same machine."
          : "Camera access requires HTTPS on this address. Open the secure HTTPS version of FreshLens AI.",
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
      {
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      },
      {
        video: {
          facingMode: { ideal: facingMode },
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
    if (isCameraStarting || localStream) return;

    isCameraStarting = true;

    const support = getCameraSupportStatus();

    if (!support.supported) {
      showCameraError(support.title, support.message, support.icon);
      isCameraStarting = false;
      return;
    }

    stopCamera();

    if (cameraErrorState) cameraErrorState.style.display = "none";
    if (cameraArea) cameraArea.style.display = "block";
    if (cameraStatusPill)
      cameraStatusPill.textContent = "Initializing Camera...";
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

      if (cameraStatusPill)
        cameraStatusPill.textContent = "Camera Ready - Place food inside frame";

      if (captureImage) captureImage.disabled = false;
      if (switchCameraBtn) switchCameraBtn.disabled = false;
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
        localStream.getTracks().forEach((track) => track.stop());
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

    if (captureImage) captureImage.disabled = true;
  }

  if (switchCameraBtn) {
    switchCameraBtn.addEventListener("click", async () => {
      if (isCameraStarting) return;

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
      if (!localStream || !webcam || isAnalyzing) return;

      const width = webcam.videoWidth || 640;
      const height = webcam.videoHeight || 480;

      if (!width || !height) {
        if (cameraStatusPill)
          cameraStatusPill.textContent = "Camera is not ready yet";
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

          if (foodImage) foodImage.files = dt.files;

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
        if (foodImage) foodImage.files = files;
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
    if (!file) return;

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
        <img
          src="${dataUrl}"
          alt="Food Preview"
          style="max-height:280px; max-width:100%; border-radius:12px; margin:0 auto; display:block; box-shadow:var(--shadow-sm);"
        />
        <p style="font-size:0.8rem; color:var(--muted); margin-top:8px; text-align:center;">
          ${escapeHtml(file?.name || "Selected Food Image")}
        </p>
      `;
    }

    if (previewActionsBar) {
      previewActionsBar.style.display = "flex";
    }
  }

  if (replaceImageBtn && foodImage) {
    replaceImageBtn.addEventListener("click", () => {
      foodImage.click();
    });
  }

  if (removeImageBtn) {
    removeImageBtn.addEventListener("click", () => {
      if (foodImage) foodImage.value = "";

      if (previewBox) {
        previewBox.innerHTML = `
          <span class="upload-icon">📷</span>
          <h3 style="font-size:1.1rem; margin-bottom:4px;">Drop your food image here</h3>
          <p style="color:var(--muted); font-size:0.85rem;">
            or choose an image from your device (JPG, PNG, WEBP)
          </p>
        `;
      }

      if (previewActionsBar) previewActionsBar.style.display = "none";

      if (qualityToast) qualityToast.style.display = "none";

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

        if (!ctx) return;

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
    if (qualityToastMessage) qualityToastMessage.textContent = message;

    if (qualityToast) qualityToast.style.display = "flex";
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
          { type: blob.type || "image/jpeg" },
        );

        const dt = new DataTransfer();
        dt.items.add(file);

        if (foodImage) foodImage.files = dt.files;

        const previewUrl = URL.createObjectURL(blob);
        displayImagePreview(previewUrl, file);

        URL.revokeObjectURL(previewUrl);

        await executeFoodAnalysis(file);
      } catch (error) {
        console.error("Sample image error:", error);
        showQualityToast(
          "Unable to load the sample image. Please upload an image manually.",
        );
      }
    });
  });

  function resetResultsView() {
    if (scanErrorBox) scanErrorBox.style.display = "none";
    if (categoryWarningBox) categoryWarningBox.style.display = "none";
    if (annotatedImageBox) annotatedImageBox.style.display = "none";
    if (resultSummaryBar) resultSummaryBar.style.display = "none";
    if (viewAnalysisPrompt) viewAnalysisPrompt.style.display = "none";
    if (afterScanActions) afterScanActions.style.display = "none";
    if (feedbackSection) feedbackSection.style.display = "none";
    if (correctionForm) correctionForm.style.display = "none";
    if (feedbackResultMsg) feedbackResultMsg.style.display = "none";

    if (scanInferenceTime) scanInferenceTime.textContent = "";

    if (multiObjectsContainer) {
      multiObjectsContainer.innerHTML = `
        <div id="resultPlaceholder" style="color:var(--muted); text-align:center; padding:40px 16px;">
          <span style="font-size:2.5rem; display:block; margin-bottom:10px;">🥗</span>
          <h3 style="font-size:1.1rem; color:var(--text); margin-bottom:4px;">
            No Scan Yet
          </h3>
          <p style="font-size:0.875rem; max-width:320px; margin:0 auto;">
            Capture or upload a food image and click
            <strong>Analyze Food</strong>
            to view detected items and freshness assessments.
          </p>
        </div>
      `;
    }
  }

  function resetUploadUI() {
    if (foodImage) foodImage.value = "";

    if (previewBox) {
      previewBox.innerHTML = `
        <span class="upload-icon">📷</span>
        <h3 style="font-size:1.1rem; margin-bottom:4px;">
          Drop your food image here
        </h3>
        <p style="color:var(--muted); font-size:0.85rem;">
          or choose an image from your device (JPG, PNG, WEBP)
        </p>
      `;
    }

    if (previewActionsBar) previewActionsBar.style.display = "none";

    if (qualityToast) qualityToast.style.display = "none";
  }

  if (resetScan) {
    resetScan.addEventListener("click", () => {
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

  async function executeFoodAnalysis(file) {
    if (isAnalyzing) return;

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

    if (scanNow) {
      scanNow.disabled = true;
      scanNow.innerHTML = "<span>⏳ Analyzing...</span>";
    }

    if (captureImage) captureImage.disabled = true;

    if (progressOverlay) progressOverlay.style.display = "flex";

    if (progressBarInner) progressBarInner.style.width = "20%";

    if (progressStepText)
      progressStepText.textContent = "Preparing image for AI analysis...";

    const formData = new FormData();

    formData.append("file", file, file.name || "food.jpg");
    formData.append("selected_category", activeCategory);

    let step1Timer = null;
    let step2Timer = null;

    try {
      step1Timer = setTimeout(() => {
        if (progressBarInner) progressBarInner.style.width = "50%";

        if (progressStepText)
          progressStepText.textContent = "Detecting food item...";
      }, 500);

      step2Timer = setTimeout(() => {
        if (progressBarInner) progressBarInner.style.width = "75%";

        if (progressStepText)
          progressStepText.textContent = "Evaluating freshness & condition...";
      }, 1500);

      const response = await fetch("/api/v1/predict", {
        method: "POST",
        body: formData,
        headers: {
          Accept: "application/json",
        },
        credentials: "same-origin",
      });

      if (step1Timer) clearTimeout(step1Timer);
      if (step2Timer) clearTimeout(step2Timer);

      if (progressBarInner) progressBarInner.style.width = "90%";

      let data = null;

      const contentType = response.headers.get("content-type") || "";

      if (contentType.includes("application/json")) {
        data = await response.json();
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

      if (progressBarInner) progressBarInner.style.width = "100%";

      if (progressStepText) progressStepText.textContent = "Analysis complete.";

      await new Promise((resolve) => setTimeout(resolve, 200));

      if (progressOverlay) progressOverlay.style.display = "none";

      console.log("FreshLens API response:", {
        status: response.status,
        ok: response.ok,
        data,
      });

      if (!response.ok || data?.success === false) {
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
      if (step1Timer) clearTimeout(step1Timer);
      if (step2Timer) clearTimeout(step2Timer);

      console.error("FreshLens prediction request failed:", error);

      if (progressOverlay) progressOverlay.style.display = "none";

      handleScanError({
        status: "network_error",
        error: {
          code: "NETWORK_ERROR",
          message:
            "Unable to connect to the FreshLens AI prediction service. Please try again.",
        },
      });
    } finally {
      isAnalyzing = false;

      if (scanNow) {
        scanNow.disabled = false;

        if (
          scanNow.innerText.includes("Analyzing") ||
          scanNow.innerText.includes("Analysis Complete")
        ) {
          scanNow.innerHTML = "<span>🔍 Analyze Food</span>";
        }
      }

      if (captureImage) captureImage.disabled = false;
    }
  }

  if (scanNow) {
    scanNow.addEventListener("click", async (event) => {
      event.preventDefault();

      if (isAnalyzing) return;

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

    if (annotatedImageBox) annotatedImageBox.style.display = "none";

    if (resultSummaryBar) resultSummaryBar.style.display = "none";

    if (viewAnalysisPrompt) viewAnalysisPrompt.style.display = "none";

    if (categoryWarningBox) categoryWarningBox.style.display = "none";

    if (afterScanActions) afterScanActions.style.display = "none";

    if (feedbackSection) feedbackSection.style.display = "none";

    const noFood =
      status === "no_food" ||
      status === "no_supported_food_detected" ||
      (data.food_detected === false && status !== "low_confidence");

    if (noFood) {
      if (scanErrorBox) scanErrorBox.style.display = "none";

      if (multiObjectsContainer) {
        multiObjectsContainer.innerHTML = `
          <div class="rejection-card no-food-card"
            style="text-align:center; padding:36px 20px; background:var(--bg-alt); border-radius:var(--radius-lg); border:1px solid var(--border);">

            <span style="font-size:3rem; display:block; margin-bottom:12px;">🔍</span>

            <h3 style="font-size:1.35rem; font-weight:800; color:var(--text); margin-bottom:8px;">
              No Food Detected
            </h3>

            <p style="font-size:0.95rem; color:var(--text-secondary); max-width:420px; margin:0 auto 12px; line-height:1.5;">
              ${escapeHtml(
                data.message ||
                  err.message ||
                  "We couldn't confidently identify a supported food item.",
              )}
            </p>

            <p style="font-size:0.875rem; color:var(--muted); max-width:400px; margin:0 auto 20px;">
              Please point the camera at a fruit, vegetable, or supported food item and try again.
            </p>

            <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
              <button type="button" class="btn primary touch-friendly" id="rejectionTryAgainBtn">
                📷 Try Again
              </button>

              <button type="button" class="btn secondary touch-friendly" id="rejectionUploadBtn">
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
      if (scanErrorBox) scanErrorBox.style.display = "none";

      if (multiObjectsContainer) {
        multiObjectsContainer.innerHTML = `
          <div class="rejection-card low-confidence-card"
            style="text-align:center; padding:36px 20px; background:var(--bg-alt); border-radius:var(--radius-lg); border:1px solid var(--border);">

            <span style="font-size:3rem; display:block; margin-bottom:12px;">⚠️</span>

            <h3 style="font-size:1.35rem; font-weight:800; color:var(--text); margin-bottom:8px;">
              Low Confidence
            </h3>

            <p style="font-size:0.95rem; color:var(--text-secondary); max-width:420px; margin:0 auto 16px; line-height:1.5;">
              ${escapeHtml(
                data.message ||
                  err.message ||
                  "The food could not be identified confidently.",
              )}
            </p>

            <div style="text-align:left; max-width:340px; margin:0 auto 24px; padding:14px 18px; background:var(--card); border-radius:var(--radius-md); border:1px solid var(--border);">
              <strong style="font-size:0.875rem; color:var(--text); display:block; margin-bottom:6px;">
                Try:
              </strong>

              <ul style="font-size:0.85rem; color:var(--text-secondary); margin:0; padding-left:18px; line-height:1.6;">
                <li>Better, even lighting</li>
                <li>A closer image of the food item</li>
                <li>A plain background</li>
                <li>Keeping the food centered</li>
              </ul>
            </div>

            <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
              <button type="button" class="btn primary touch-friendly" id="lowConfTryAgainBtn">
                📷 Try Again
              </button>

              <button type="button" class="btn secondary touch-friendly" id="lowConfUploadBtn">
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
      if (scanErrorCode) scanErrorCode.textContent = err.code || "Scan Error";

      if (scanErrorMessage)
        scanErrorMessage.textContent =
          err.message || "Please upload a clearer food image.";

      if (scanErrorBox) scanErrorBox.style.display = "flex";
    }

    if (window.innerWidth <= 768 && resultCard) {
      resultCard.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }

  function renderScanSuccess(data) {
    console.log("FreshLens successful analysis:", data);

    try {
      sessionStorage.setItem("freshlens_latest_analysis", JSON.stringify(data));
    } catch (err) {
      console.warn("Could not save analysis to sessionStorage:", err);
    }

    lastUploadedImageUrl = data.image_url || "";
    lastPredictedLabel = data.label || "";

    if (scanInferenceTime && data.inference_time_ms) {
      scanInferenceTime.textContent = `${data.inference_time_ms} ms`;
    }

    if (data.annotated_image_url && annotatedImage) {
      annotatedImage.src = data.annotated_image_url;

      if (annotatedImageBox) annotatedImageBox.style.display = "block";
    }

    if (data.warning) {
      if (categoryWarningMessage)
        categoryWarningMessage.textContent = data.warning;

      if (categoryWarningBox) categoryWarningBox.style.display = "flex";
    }

    const summary = data.summary || {};

    const totalObjects =
      Number(summary.total_objects) ||
      (Array.isArray(data.objects) ? data.objects.length : 1);

    if (summaryText) {
      summaryText.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:4px;">
          <div>
            <strong>${totalObjects} Object${totalObjects > 1 ? "s" : ""} Identified</strong>
            &nbsp;|&nbsp;
            🍎 Fruits: ${Number(summary.fruits) || 0}
            &nbsp;|&nbsp;
            🥦 Vegetables: ${Number(summary.vegetables) || 0}
            &nbsp;|&nbsp;
            🍱 Foods: ${Number(summary.food) || 0}
          </div>
          <div style="font-size:0.8rem; color:var(--text-secondary); opacity:0.9;">
            ✅ Fresh: ${Number(summary.fresh) || 0} &nbsp;|&nbsp; ⚠️ Spoiled: ${Number(summary.spoiled) || 0}
          </div>
        </div>
      `;
    }

    if (resultSummaryBar) resultSummaryBar.style.display = "block";

    if (viewAnalysisPrompt) {
      viewAnalysisPrompt.style.display = "block";
    }

    if (multiObjectsContainer) {
      multiObjectsContainer.innerHTML = "";

      if (Array.isArray(data.objects) && data.objects.length > 0) {
        data.objects.forEach((obj) => {
          multiObjectsContainer.appendChild(createObjectResultCard(obj));
        });
      } else {
        multiObjectsContainer.innerHTML = `
          <div style="padding:30px;text-align:center;color:var(--muted);">
            Analysis completed, but no result objects were returned.
          </div>
        `;
      }
    }

    if (afterScanActions) afterScanActions.style.display = "flex";

    if (feedbackSection) feedbackSection.style.display = "block";

    if (window.innerWidth <= 768 && resultCard) {
      resultCard.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }

  function createObjectResultCard(obj = {}) {
    const card = document.createElement("div");
    card.className = "detected-item-card";

    const item = escapeHtml(obj.item || "Unknown Food");

    const category = escapeHtml(obj.category || "Food");

    const freshness = obj.freshness || "";

    let badgeClass = "neutral";
    let badgeText = obj.freshness_status || "Not Available";

    if (freshness === "Fresh") {
      badgeClass = "fresh";
      badgeText = "Fresh ✅";
    } else if (freshness === "Spoiled") {
      badgeClass = "spoiled";
      badgeText = "Spoiled ⚠️";
    }

    const detectionConfidence = Number(obj.detection_confidence);

    const freshnessConfidence = Number(obj.freshness_confidence);

    const detConfPct = Number.isFinite(detectionConfidence)
      ? Math.round(detectionConfidence * 100)
      : 0;

    const freshConfPct = Number.isFinite(freshnessConfidence)
      ? Math.round(freshnessConfidence * 100)
      : detConfPct;

    const confidence = Number.isFinite(freshnessConfidence)
      ? freshConfPct
      : detConfPct;

    const nutrition = obj.nutrition || {};

    const nutritionHtml =
      nutrition.calories_per_100g != null
        ? `
          <div class="nutrition-grid">
            <div class="nutrition-item">
              <span class="nut-label">Calories</span>
              <span class="nut-val">${escapeHtml(nutrition.calories_per_100g)} kcal</span>
            </div>

            <div class="nutrition-item">
              <span class="nut-label">Carbs</span>
              <span class="nut-val">${escapeHtml(nutrition.carbs_g ?? "-")}g</span>
            </div>

            <div class="nutrition-item">
              <span class="nut-label">Protein</span>
              <span class="nut-val">${escapeHtml(nutrition.protein_g ?? "-")}g</span>
            </div>
          </div>
        `
        : "";

    const stabilityWarning = obj.stability_warning
      ? `
        <div class="warning-box" style="margin-top:12px; padding:8px 12px; font-size:0.825rem;">
          💡 ${escapeHtml(obj.stability_warning)}
        </div>
      `
      : "";

    const storageTip = obj.storage_tip
      ? `<p style="margin-top:8px;"><strong>Storage:</strong> ${escapeHtml(obj.storage_tip)}</p>`
      : "";

    const safetyGuideline = obj.safety_guideline
      ? `<p style="margin-top:8px;"><strong>Safety:</strong> ${escapeHtml(obj.safety_guideline)}</p>`
      : "";

    const guidanceHtml =
      storageTip || safetyGuideline
        ? `
          <details class="guideline-accordion" style="margin-top:14px;">
            <summary>Storage & Safety Guidance</summary>
            ${storageTip}
            ${safetyGuideline}
          </details>
        `
        : "";

    card.innerHTML = `
      <div class="item-card-header">
        <div>
          <h3 style="font-size:1.2rem; font-weight:800; color:var(--text);">
            ${item}
          </h3>

          <span style="font-size:0.8rem; color:var(--muted);">
            ${category}
          </span>
        </div>

        <span class="status-badge ${badgeClass}" style="font-size:0.8rem; padding:4px 12px;">
          ${escapeHtml(badgeText)}
        </span>
      </div>

      <div class="confidence-meter-container">
        <div style="display:flex; justify-content:space-between; font-size:0.825rem; font-weight:600;">
          <span style="color:var(--text-secondary);">
            Confidence
          </span>

          <span style="color:var(--primary); font-weight:700;">
            ${confidence}%
          </span>
        </div>

        <div class="confidence-bar-track">
          <div
            class="confidence-bar-fill"
            style="width:${Math.max(0, Math.min(100, confidence))}%;">
          </div>
        </div>
      </div>

      ${nutritionHtml}
      ${stabilityWarning}
      ${guidanceHtml}

      <div style="margin-top:16px;">
        <button
          class="btn secondary touch-friendly ask-ai-obj-btn"
          type="button"
          style="width:100%; border-radius:var(--radius-full); font-size:0.875rem;">
          🤖 Ask AI Assistant About This ${item} ➔
        </button>
      </div>
    `;

    const askBtn = card.querySelector(".ask-ai-obj-btn");

    if (askBtn) {
      askBtn.addEventListener("click", () => {
        const scanContext = {
          item: obj.item || "",
          category: obj.category || "",
          freshness: obj.freshness || "",
          confidence: obj.freshness_confidence || obj.detection_confidence || 0,
          storage_tip: obj.storage_tip || "",
          safety_guideline: obj.safety_guideline || "",
        };

        sessionStorage.setItem(
          "freshlens_active_scan",
          JSON.stringify(scanContext),
        );

        window.location.href = "/chatbot";
      });
    }

    return card;
  }

  if (feedbackYes) {
    feedbackYes.addEventListener("click", async () => {
      feedbackYes.disabled = true;

      if (feedbackNo) feedbackNo.disabled = true;

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

        if (!response.ok) throw new Error("Feedback request failed");

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
      if (correctionForm) correctionForm.style.display = "block";
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

        if (!response.ok) throw new Error("Feedback submission failed");

        if (correctionForm) correctionForm.style.display = "none";

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
      { threshold: 0.15 },
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
