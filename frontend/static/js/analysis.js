/**
 * FoodLens-AI - Food Analysis Dashboard Controller
 * Handles multi-object visualization, category grouping, duplicate item breakdowns,
 * nutrition calculation, and safety guidance rendering.
 */
document.addEventListener("DOMContentLoaded", async () => {
  const analysisEmptyState = document.getElementById("analysisEmptyState");
  const analysisContentContainer = document.getElementById("analysisContentContainer");

  const analysisOriginalImage = document.getElementById("analysisOriginalImage");
  const analysisAnnotatedImage = document.getElementById("analysisAnnotatedImage");
  const annotatedDetectionCount = document.getElementById("annotatedDetectionCount");

  const kpiTotalObjects = document.getElementById("kpiTotalObjects");
  const kpiFruits = document.getElementById("kpiFruits");
  const kpiVegetables = document.getElementById("kpiVegetables");
  const kpiOtherFoods = document.getElementById("kpiOtherFoods");
  const kpiFresh = document.getElementById("kpiFresh");
  const kpiSpoiled = document.getElementById("kpiSpoiled");

  const primaryFreshnessBadge = document.getElementById("primaryFreshnessBadge");
  const freshnessStatBars = document.getElementById("freshnessStatBars");
  const nutritionItemCountBadge = document.getElementById("nutritionItemCountBadge");
  const nutritionTotalsGrid = document.getElementById("nutritionTotalsGrid");

  const fruitsFoodGroupsContainer = document.getElementById("fruitsFoodGroupsContainer");
  const vegetablesFoodGroupsContainer = document.getElementById("vegetablesFoodGroupsContainer");
  const otherFoodGroupsContainer = document.getElementById("otherFoodGroupsContainer");
  const fruitsGroupCount = document.getElementById("fruitsGroupCount");
  const vegetablesGroupCount = document.getElementById("vegetablesGroupCount");
  const otherGroupCount = document.getElementById("otherGroupCount");

  const spoiledAlertBanner = document.getElementById("spoiledAlertBanner");
  const spoiledAlertItemsText = document.getElementById("spoiledAlertItemsText");
  const spoiledGuidanceList = document.getElementById("spoiledGuidanceList");
  const freshGuidanceList = document.getElementById("freshGuidanceList");

  const lowConfidenceAdvisory = document.getElementById("lowConfidenceAdvisory");
  const lowConfidenceItemsList = document.getElementById("lowConfidenceItemsList");

  const askAiAssistantBtn = document.getElementById("askAiAssistantBtn");
  const askAiBottomBtn = document.getElementById("askAiBottomBtn");

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Load scan data from URL query id, or sessionStorage, or recent history
  let scanData = null;
  const urlParams = new URLSearchParams(window.location.search);
  const scanId = urlParams.get("id");

  if (scanId) {
    try {
      const resp = await fetch("/api/v1/history");
      if (resp.ok) {
        const histData = await resp.json();
        const records = histData.data || [];
        scanData = records.find(r => String(r.id) === scanId || String(r.timestamp) === scanId);
      }
    } catch (e) {
      console.warn("Could not fetch scan record by ID:", e);
    }
  }

  if (!scanData) {
    const stored = sessionStorage.getItem("foodlens_latest_analysis") || sessionStorage.getItem("freshlens_latest_analysis");
    if (stored) {
      try {
        scanData = JSON.parse(stored);
      } catch (e) {
        console.error("Invalid session storage scan data:", e);
      }
    }
  }

  if (!scanData || (!scanData.objects && !scanData.food_name)) {
    if (analysisEmptyState) analysisEmptyState.style.display = "block";
    if (analysisContentContainer) analysisContentContainer.style.display = "none";
    return;
  }

  renderAnalysisDashboard(scanData);

  function renderAnalysisDashboard(data) {
    if (analysisEmptyState) analysisEmptyState.style.display = "none";
    if (analysisContentContainer) analysisContentContainer.style.display = "block";

    const objects = Array.isArray(data.objects) ? data.objects : [];
    const summary = data.summary || {};
    const totalObjects = objects.length || Number(summary.total_objects) || 1;

    // 1. Render Images
    const originalUrl = data.image_url || "/static/image/apple.jpg";
    const annotatedUrl = data.annotated_image_url || originalUrl;

    if (analysisOriginalImage) {
      analysisOriginalImage.src = originalUrl;
    }
    if (analysisAnnotatedImage) {
      analysisAnnotatedImage.src = annotatedUrl;
    }
    if (annotatedDetectionCount) {
      annotatedDetectionCount.textContent = `${totalObjects} item${totalObjects > 1 ? "s" : ""} boxed`;
    }

    // 2. Calculate summary statistics dynamically
    let countFruits = 0;
    let countVeg = 0;
    let countOther = 0;
    let countFresh = 0;
    let countSpoiled = 0;
    let lowConfItems = [];

    objects.forEach(obj => {
      const cat = String(obj.category || "").toLowerCase();
      if (cat.includes("fruit")) countFruits++;
      else if (cat.includes("veg")) countVeg++;
      else countOther++;

      const cond = String(obj.freshness_status || obj.freshness || "").toLowerCase();
      if (cond.includes("fresh")) countFresh++;
      else if (cond.includes("spoiled") || cond.includes("spoile")) countSpoiled++;

      const conf = Number(obj.confidence ?? obj.freshness_confidence ?? 0);
      if (conf < 0.60 || obj.is_confident === false) {
        lowConfItems.push(obj);
      }
    });

    if (kpiTotalObjects) kpiTotalObjects.textContent = totalObjects;
    if (kpiFruits) kpiFruits.textContent = countFruits;
    if (kpiVegetables) kpiVegetables.textContent = countVeg;
    if (kpiOtherFoods) kpiOtherFoods.textContent = countOther;
    if (kpiFresh) kpiFresh.textContent = countFresh;
    if (kpiSpoiled) kpiSpoiled.textContent = countSpoiled;

    // 3. Freshness Overview Bars
    if (primaryFreshnessBadge) {
      const primaryCond = data.condition || (countSpoiled > countFresh ? "Spoiled" : "Fresh");
      primaryFreshnessBadge.textContent = primaryCond;
      primaryFreshnessBadge.className = `badge ${primaryCond === 'Fresh' ? 'status-badge fresh' : 'status-badge spoiled'}`;
    }

    if (freshnessStatBars) {
      const pctFresh = totalObjects > 0 ? Math.round((countFresh / totalObjects) * 100) : 0;
      const pctSpoiled = totalObjects > 0 ? Math.round((countSpoiled / totalObjects) * 100) : 0;
      const pctLowConf = totalObjects > 0 ? Math.round((lowConfItems.length / totalObjects) * 100) : 0;

      freshnessStatBars.innerHTML = `
        <div class="stat-progress-row">
          <div class="stat-progress-info">
            <span>Fresh Items (${countFresh})</span>
            <strong>${pctFresh}%</strong>
          </div>
          <div class="stat-progress-track">
            <div class="stat-progress-fill fresh-fill" style="width: ${pctFresh}%;"></div>
          </div>
        </div>

        <div class="stat-progress-row" style="margin-top: 12px;">
          <div class="stat-progress-info">
            <span>Spoiled Items (${countSpoiled})</span>
            <strong>${pctSpoiled}%</strong>
          </div>
          <div class="stat-progress-track">
            <div class="stat-progress-fill spoiled-fill" style="width: ${pctSpoiled}%;"></div>
          </div>
        </div>

        ${lowConfItems.length > 0 ? `
          <div class="stat-progress-row" style="margin-top: 12px;">
            <div class="stat-progress-info">
              <span>Low Confidence Items (${lowConfItems.length})</span>
              <strong>${pctLowConf}%</strong>
            </div>
            <div class="stat-progress-track">
              <div class="stat-progress-fill warning-fill" style="width: ${pctLowConf}%;"></div>
            </div>
          </div>
        ` : ''}
      `;
    }

    // 4. Nutrition Overview
    const nutSummary = data.nutrition_summary || {};
    let totalCal = Number(nutSummary.estimated_calories) || 0;
    let totalCarbs = Number(nutSummary.estimated_carbs_g) || 0;
    let totalProt = Number(nutSummary.estimated_protein_g) || 0;
    let totalFiber = Number(nutSummary.estimated_fiber_g) || 0;

    // If summary not in payload, compute directly from objects
    if (!totalCal && objects.length > 0) {
      objects.forEach(obj => {
        const nut = obj.nutrition || {};
        if (nut.calories_per_100g != null) totalCal += Number(nut.calories_per_100g);
        if (nut.carbs_g != null) totalCarbs += Number(nut.carbs_g);
        if (nut.protein_g != null) totalProt += Number(nut.protein_g);
        if (nut.fiber_g != null) totalFiber += Number(nut.fiber_g);
      });
    }

    if (nutritionItemCountBadge) {
      nutritionItemCountBadge.textContent = `${totalObjects} Item${totalObjects > 1 ? "s" : ""}`;
    }

    if (nutritionTotalsGrid) {
      if (totalCal > 0 || totalCarbs > 0 || totalProt > 0) {
        nutritionTotalsGrid.innerHTML = `
          <div class="nut-card">
            <span class="nut-metric-val">${Math.round(totalCal)}</span>
            <span class="nut-metric-unit">kcal</span>
            <span class="nut-metric-label">Estimated Calories</span>
          </div>
          <div class="nut-card">
            <span class="nut-metric-val">${totalCarbs.toFixed(1)}</span>
            <span class="nut-metric-unit">g</span>
            <span class="nut-metric-label">Total Carbs</span>
          </div>
          <div class="nut-card">
            <span class="nut-metric-val">${totalProt.toFixed(1)}</span>
            <span class="nut-metric-unit">g</span>
            <span class="nut-metric-label">Total Protein</span>
          </div>
          <div class="nut-card">
            <span class="nut-metric-val">${totalFiber.toFixed(1)}</span>
            <span class="nut-metric-unit">g</span>
            <span class="nut-metric-label">Total Fiber</span>
          </div>
        `;
      } else {
        nutritionTotalsGrid.innerHTML = `
          <div style="grid-column: 1 / -1; padding: 20px; text-align: center; color: var(--muted); font-size: 0.9rem;">
            Nutritional information unavailable for detected items.
          </div>
        `;
      }
    }

    // 5. Category-Wise Analysis & Duplicate Food Grouping
    renderCategorySection("Fruit", objects.filter(o => String(o.category || "").toLowerCase().includes("fruit")), fruitsFoodGroupsContainer, fruitsGroupCount);
    renderCategorySection("Vegetable", objects.filter(o => String(o.category || "").toLowerCase().includes("veg")), vegetablesFoodGroupsContainer, vegetablesGroupCount);
    renderCategorySection("Other", objects.filter(o => !String(o.category || "").toLowerCase().includes("fruit") && !String(o.category || "").toLowerCase().includes("veg")), otherFoodGroupsContainer, otherGroupCount);

    // 6. Safety & Storage Advisories
    renderSafetyAdvisory(objects);

    // 7. Low Confidence Advisory
    if (lowConfItems.length > 0 && lowConfidenceAdvisory && lowConfidenceItemsList) {
      lowConfidenceAdvisory.style.display = "flex";
      lowConfidenceItemsList.innerHTML = lowConfItems.map((item, i) => {
        const confVal = Math.round((item.confidence || item.detection_confidence || 0) * 100);
        return `<strong>${escapeHtml(item.item || "Food Item")} #${i + 1}</strong> (${confVal}%)`;
      }).join(", ");
    } else if (lowConfidenceAdvisory) {
      lowConfidenceAdvisory.style.display = "none";
    }

    // 8. Ask AI Assistant Setup
    const scanContextText = encodeURIComponent(
      `I scanned an image with FoodLens-AI. Detected items: ${objects.map(o => `${o.item} (${o.freshness_status})`).join(", ")}. Can you provide recipe tips and storage guidance?`
    );
    if (askAiAssistantBtn) {
      askAiAssistantBtn.href = `/chatbot?prompt=${scanContextText}`;
    }
    if (askAiBottomBtn) {
      askAiBottomBtn.href = `/chatbot?prompt=${scanContextText}`;
    }
  }

  function renderCategorySection(categoryName, items, container, countElem) {
    if (!container) return;

    if (countElem) {
      countElem.textContent = `${items.length} detected`;
    }

    if (!items || items.length === 0) {
      container.innerHTML = `
        <div class="empty-category-notice">
          No ${escapeHtml(categoryName.toLowerCase())}s detected in this image.
        </div>
      `;
      return;
    }

    // Group items by food identity (e.g. Apple: 3 detected)
    const grouped = {};
    items.forEach(item => {
      const name = item.item || item.label || "Produce Item";
      if (!grouped[name]) grouped[name] = [];
      grouped[name].push(item);
    });

    container.innerHTML = "";

    Object.keys(grouped).forEach(foodName => {
      const foodItems = grouped[foodName];
      const groupCard = document.createElement("div");
      groupCard.className = "food-group-wrapper";

      const firstItem = foodItems[0] || {};
      const nut = firstItem.nutrition || {};

      groupCard.innerHTML = `
        <div class="food-group-header">
          <div class="food-group-title">
            <h3>${escapeHtml(foodName)}</h3>
            <span class="group-count-badge">${foodItems.length} detected</span>
          </div>
          <span class="group-category-tag">${escapeHtml(firstItem.category || categoryName)}</span>
        </div>

        <div class="food-items-grid">
          ${foodItems.map((obj, idx) => {
        const cond = obj.freshness_status || obj.freshness || "Not Available";
        const isFresh = cond === "Fresh";
        const badgeClass = isFresh ? "fresh" : "spoiled";
        const badgeIcon = isFresh ? "✓ Fresh" : "⚠️ Spoiled";
        const confVal = Math.round((obj.confidence || obj.freshness_confidence || 0) * 100);

        return `
              <div class="individual-food-card">
                <div class="ind-card-top">
                  <div class="ind-card-title">
                    <h4>${escapeHtml(foodName)} #${idx + 1}</h4>
                    <span class="ind-card-sub">${escapeHtml(obj.category || categoryName)}</span>
                  </div>
                  <span class="status-badge ${badgeClass}">${badgeIcon}</span>
                </div>

                <div class="ind-confidence-row">
                  <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                    <span style="color:var(--text-secondary);">Model Confidence</span>
                    <strong style="color:var(--primary);">${confVal}%</strong>
                  </div>
                  <div class="stat-progress-track">
                    <div class="stat-progress-fill ${isFresh ? 'fresh-fill' : 'spoiled-fill'}" style="width:${confVal}%;"></div>
                  </div>
                </div>

                ${nut.calories_per_100g != null ? `
                  <div class="ind-nutrition-snippet">
                    <div class="snippet-item">
                      <span class="snip-label">Calories</span>
                      <span class="snip-val">${escapeHtml(nut.calories_per_100g)} kcal</span>
                    </div>
                    <div class="snippet-item">
                      <span class="snip-label">Carbs</span>
                      <span class="snip-val">${escapeHtml(nut.carbs_g ?? "-")}g</span>
                    </div>
                    <div class="snippet-item">
                      <span class="snip-label">Protein</span>
                      <span class="snip-val">${escapeHtml(nut.protein_g ?? "-")}g</span>
                    </div>
                  </div>
                ` : ''}

                ${(obj.storage_tip || obj.safety_guideline) ? `
                  <div class="ind-guidance-box">
                    ${obj.storage_tip ? `<p><strong>Storage:</strong> ${escapeHtml(obj.storage_tip)}</p>` : ''}
                    ${obj.safety_guideline ? `<p style="margin-top:6px; color: ${isFresh ? 'inherit' : 'var(--spoiled)'};"><strong>Safety:</strong> ${escapeHtml(obj.safety_guideline)}</p>` : ''}
                  </div>
                ` : ''}
              </div>
            `;
      }).join("")}
        </div>
      `;

      container.appendChild(groupCard);
    });
  }

  function renderSafetyAdvisory(objects) {
    const spoiledItems = objects.filter(o => String(o.freshness_status || o.freshness || "").toLowerCase().includes("spoile"));
    const freshItems = objects.filter(o => String(o.freshness_status || o.freshness || "").toLowerCase().includes("fresh"));

    // Spoiled Alert
    if (spoiledItems.length > 0 && spoiledAlertBanner && spoiledGuidanceList) {
      spoiledAlertBanner.style.display = "flex";
      if (spoiledAlertItemsText) {
        spoiledAlertItemsText.textContent = `Spoiled items detected: ${spoiledItems.map(o => o.item).join(", ")}. Do not consume food displaying decay or fungal lesions.`;
      }
      spoiledGuidanceList.innerHTML = spoiledItems.map(o => {
        return `
          <div class="guidance-entry spoiled-entry">
            <strong>⚠️ ${escapeHtml(o.item)} (${escapeHtml(o.freshness_status || "Spoiled")}):</strong>
            <p>${escapeHtml(o.safety_guideline || "Inspect closely. Discard if exhibiting fungal mold, foul odor, or mushy tissue.")}</p>
          </div>
        `;
      }).join("");
    } else if (spoiledAlertBanner) {
      spoiledAlertBanner.style.display = "none";
    }

    // Fresh Guidance
    if (freshItems.length > 0 && freshGuidanceList) {
      // Deduplicate by food item name
      const seen = new Set();
      const uniqueFresh = freshItems.filter(o => {
        if (seen.has(o.item)) return false;
        seen.add(o.item);
        return true;
      });

      freshGuidanceList.innerHTML = uniqueFresh.map(o => {
        return `
          <div class="guidance-entry fresh-entry">
            <strong>🌱 ${escapeHtml(o.item)}:</strong>
            <p>${escapeHtml(o.storage_tip || "Store in cool, dry conditions or refrigeration crisper to maintain freshness.")}</p>
          </div>
        `;
      }).join("");
    }
  }
});
