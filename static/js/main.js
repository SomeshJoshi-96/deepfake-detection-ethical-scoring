document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const themeToggle = document.getElementById('theme-toggle');
    const modeToggles = document.querySelectorAll('.detection-mode-toggle');
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadFormats = document.getElementById('upload-formats');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const videoPreview = document.getElementById('video-preview');
    const removeFileBtn = document.getElementById('remove-file');
    const detectButton = document.getElementById('detect-button');
    const progressContainer = document.getElementById('upload-progress-container');
    const progressBar = document.getElementById('upload-progress-bar');
    const resultsContainer = document.getElementById('results-container');
    const resultBadge = document.getElementById('result-badge');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceValue = document.getElementById('confidence-value');
    const ethicalScoreContainer = document.getElementById('ethical-score-container');
    const ethicalScoreCenter = document.getElementById('ethical-score-center');
    const ethicalText = document.getElementById('ethical-text');
    const errorAlert = document.getElementById('error-alert');
    const recentDetectionsList = document.getElementById('recent-detections-list');
    
    // Feedback elements
    const userFeedbackSection = document.getElementById('user-feedback-section');
    const reasonSelects = document.querySelectorAll('.reason-select');
    const categoryScoreInputs = document.querySelectorAll('.category-score-input');
    const selectedCategoriesSummary = document.getElementById('selected-categories-summary');
    const submitFeedbackBtn = document.getElementById('submit-feedback-btn');
    const feedbackSuccess = document.getElementById('feedback-success');

    // State Variables
    let currentMode = 'image';
    let currentFile = null;
    let ethicalChart = null;
    let recentDetections = [];
    let currentDetectionData = null;

    // Theme Management
    function setTheme(isDark) {
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
        localStorage.setItem('darkTheme', isDark ? 'true' : 'false');
        
        // Update toggle icon
        themeToggle.innerHTML = isDark ? 
            '<i class="fas fa-sun"></i>' : 
            '<i class="fas fa-moon"></i>';
    }

    // Initialize theme from localStorage or default to light theme
    function initializeTheme() {
        const savedTheme = localStorage.getItem('darkTheme');
        if (savedTheme !== null) {
            setTheme(savedTheme === 'true');
        } else {
            // Default to light theme
            setTheme(false);
        }
    }

    // Toggle theme when button is clicked
    themeToggle.addEventListener('click', function() {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        setTheme(currentTheme === 'light');
    });

    // Initialize theme on page load
    initializeTheme();
    
    // Initialize ethical score badges for all categories (removed global ethicalScoreValue)
    // Now using per-category score badges

    // Toggle between image and video modes
    modeToggles.forEach(toggle => {
        toggle.addEventListener('change', function() {
            currentMode = this.value;
            
            // Update accepted file formats
            if (currentMode === 'image') {
                fileInput.setAttribute('accept', '.jpg,.jpeg,.png');
                uploadFormats.textContent = 'Accepted formats: .jpg, .jpeg, .png';
            } else {
                fileInput.setAttribute('accept', '.mp4,.mov,.avi');
                uploadFormats.textContent = 'Accepted formats: .mp4, .mov, .avi';
            }
            
            // Reset file input and previews
            resetFileInput();
        });
    });

    // Drag and Drop Functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function() {
        this.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    // Click to upload
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFileSelection(this.files[0]);
        }
    });

    // File handling
    function handleFileSelection(file) {
        // Validate file type based on current mode
        const validFile = validateFile(file);
        
        if (!validFile) {
            showError(`Invalid file type. Please upload ${currentMode === 'image' ? 'JPG, JPEG, or PNG' : 'MP4, MOV, or AVI'}.`);
            return;
        }
        
        currentFile = file;
        showFilePreview(file);
        detectButton.disabled = false;
    }

    function validateFile(file) {
        const fileName = file.name.toLowerCase();
        
        if (currentMode === 'image') {
            return fileName.endsWith('.jpg') || fileName.endsWith('.jpeg') || fileName.endsWith('.png');
        } else {
            return fileName.endsWith('.mp4') || fileName.endsWith('.mov') || fileName.endsWith('.avi');
        }
    }

    function showFilePreview(file) {
        // Create object URL for preview
        const objectUrl = URL.createObjectURL(file);
        
        // Show preview based on file type
        if (currentMode === 'image') {
            imagePreview.src = objectUrl;
            imagePreview.style.display = 'block';
            videoPreview.style.display = 'none';
            videoPreview.src = '';
        } else {
            videoPreview.src = objectUrl;
            videoPreview.style.display = 'block';
            imagePreview.style.display = 'none';
            imagePreview.src = '';
        }
        
        // Show preview container
        previewContainer.style.display = 'block';
        
        // Hide results if they were shown
        resultsContainer.style.display = 'none';
        
        // Hide feedback success if it was shown
        feedbackSuccess.style.display = 'none';
    }

    // Remove file button
    removeFileBtn.addEventListener('click', function() {
        resetFileInput();
    });

    function resetFileInput() {
        // Clear file input
        fileInput.value = '';
        currentFile = null;
        
        // Hide preview
        previewContainer.style.display = 'none';
        imagePreview.src = '';
        videoPreview.src = '';
        
        // Disable detect button
        detectButton.disabled = true;
        
        // Hide results
        resultsContainer.style.display = 'none';
        
        // Hide error alert
        errorAlert.style.display = 'none';
        
        // Hide feedback form
        userFeedbackSection.style.display = 'none';
        
        // Reset current detection data
        currentDetectionData = null;
    }

    // Load reasons for deepfakes on page load
    fetchDeepfakeReasons();

    function fetchDeepfakeReasons() {
        fetch('/api/deepfake-reasons')
            .then(response => response.json())
            .then(data => {
                // Process each category of reasons
                if (data.general) {
                    populateReasonSelect('general-select', data.general);
                }
                if (data.emotions) {
                    populateReasonSelect('emotions-select', data.emotions);
                }
                if (data.personality) {
                    populateReasonSelect('personality-select', data.personality);
                }
                if (data.broad) {
                    populateReasonSelect('broad-select', data.broad);
                }
            })
            .catch(error => {
                console.error('Error fetching deepfake reasons:', error);
                showError('Failed to load deepfake reasons. Please refresh the page.');
            });
    }
    
    function populateReasonSelect(selectId, reasons) {
        const select = document.getElementById(selectId);
        if (!select) return;
        
        // Clear existing options
        select.innerHTML = '<option value="" selected disabled>Select a reason...</option>';
        
        // Add each reason as an option
        reasons.forEach(reason => {
            const option = document.createElement('option');
            option.value = reason.id;
            option.textContent = reason.text;
            select.appendChild(option);
        });
        
        // Add change event listener to each select
        select.addEventListener('change', function() {
            const categoryId = this.getAttribute('data-category');
            const tabButton = document.getElementById(`${categoryId}-tab`);
            
            // Add visual indicator to tab if a reason is selected
            if (this.value) {
                if (tabButton) {
                    tabButton.classList.add('text-success');
                    // Only add icon if it doesn't already have one
                    if (!tabButton.innerHTML.includes('fa-check-circle')) {
                        tabButton.innerHTML = `<i class="fas fa-check-circle me-1"></i>${tabButton.textContent}`;
                    }
                }
            } else {
                // Remove visual indicator if nothing is selected
                if (tabButton) {
                    tabButton.classList.remove('text-success');
                    tabButton.innerHTML = tabButton.textContent.replace('<i class="fas fa-check-circle me-1"></i>', '');
                }
            }
            
            // Update the categories selected counter
            updateSelectedCategoriesCounter();
            
            // Update the categories summary display
            updateCategoriesSummary();
        });
    }

    // Detect button
    detectButton.addEventListener('click', function() {
        if (!currentFile) {
            showError('Please select a file first.');
            return;
        }
        // Show progress bar
        progressContainer.style.display = "block";
        progressBar.style.width = "0%";
        progressBar.style.backgroundSize = "200% 100%";
        progressBar.classList.add("animated-progress");
        
        // Create form data
        const formData = new FormData();
        formData.append("file", currentFile);
        
        // Determine endpoint based on current mode
        const endpoint = currentMode === "image" ? "/detect/image" : "/detect/video";
        
        // Simulate progress more smoothly
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += (currentMode === "image" ? 3 : 2);
            if (progress <= 90) {
                progressBar.style.width = `${progress}%`;
            }
        }, 100);
        
        // Send request to backend
        fetch(endpoint, {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'Detection failed. Please try again.');
                });
            }
            
            return response.json();
        })
        .then(data => {
            // Store the detection data
            currentDetectionData = data;
            
            // Hide progress after a small delay
            setTimeout(() => {
                progressContainer.style.display = 'none';
                
                // Display results
                displayResults(data);
                
                // Add to recent detections
                addToRecentDetections(data);
                
                // Show feedback form if it's a fake
                if (data.result.toLowerCase() === 'fake') {
                    userFeedbackSection.style.display = 'block';
                    feedbackSuccess.style.display = 'none';
                    
                    // Reset tab indicators and counter when showing feedback form
                    const categoryTabs = ['general', 'emotions', 'personality', 'broad'];
                    categoryTabs.forEach(category => {
                        const tabButton = document.getElementById(`${category}-tab`);
                        if (tabButton) {
                            tabButton.classList.remove('text-success');
                            tabButton.innerHTML = tabButton.textContent.replace('<i class="fas fa-check-circle me-1"></i>', '');
                        }
                    });
                    
                    // Reset the counter
                    updateSelectedCategoriesCounter();
                } else {
                    userFeedbackSection.style.display = 'none';
                }
            }, 500);
        })
        .catch(error => {
            clearInterval(progressInterval);
            progressContainer.style.display = 'none';
            showError(error.message);
        });
    });

    // Display results
    function displayResults(data) {
        // Set result badge with animation
        resultBadge.textContent = data.result.toUpperCase();
        resultBadge.className = `badge ${data.result.toLowerCase()} animated-fade-in`;
        
        // Set confidence score with animation
        const confidencePercent = data.confidence.toFixed(1);
        confidenceBar.style.width = `${confidencePercent}%`;
        confidenceValue.textContent = `${confidencePercent}%`;
        
        // Set color based on result with animation
        confidenceBar.className = `progress-bar animated-progress bg-${data.result === 'real' ? 'success' : 'danger'}`;
        
        // Handle ethical score (only show feedback for fake media)
        if (data.result === 'fake' && data.ethical_score !== undefined) {
            // Display the ethical score component
            ethicalScoreContainer.style.display = 'block';
            ethicalScoreCenter.textContent = Math.round(data.ethical_score);
            
            // Update ethical chart
            updateEthicalChart(data.ethical_score);
            
            // Set ethical impact text with more detailed analysis
            if (data.ethical_score > 7) {
                ethicalText.textContent = 'Low concern - Minor manipulation with limited potential harm. This type of deepfake is less likely to cause significant issues.';
            } else if (data.ethical_score > 5) {
                ethicalText.textContent = 'Moderate concern - Significant manipulation with potential for ethical impact. This type of deepfake could be misleading in certain contexts.';
            } else {
                ethicalText.textContent = 'High concern - Severe manipulation with significant potential for harm. This type of deepfake could be used maliciously and requires careful consideration.';
            }
            
            // Fetch reasons for deepfake classifications
            fetchDeepfakeReasons();
            
            // Show the user feedback section after a short delay
            setTimeout(() => {
                userFeedbackSection.style.display = 'block';
                
                // Reset any previous feedback
                resetFeedbackForm();
                
                // Smooth scroll to the feedback section
                userFeedbackSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 1000);
        } else {
            // Show ethical assessment for real media too, but without feedback form
            ethicalScoreContainer.style.display = 'block';
            ethicalScoreCenter.textContent = '0';
            updateEthicalChart(0);
            ethicalText.textContent = 'No ethical concerns - This media appears to be authentic with no signs of manipulation.';
            
            // Hide the feedback section for real media
            userFeedbackSection.style.display = 'none';
        }
        
        // Show results container
        resultsContainer.style.display = 'block';
        
        // Add this detection to recent detections history
        addToRecentDetections(data);
    }

    // Update ethical chart
    function updateEthicalChart(score) {
        if (ethicalChart) {
            ethicalChart.destroy();
        }
        
        const ctx = document.getElementById('ethical-chart').getContext('2d');
        
        // Determine color based on ethical score
        let color;
        if (score < 30) {
            color = '#34a853';  // Green for low concern
        } else if (score < 70) {
            color = '#fbbc05';  // Yellow for moderate concern
        } else {
            color = '#ea4335';  // Red for high concern
        }
        
        ethicalChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [score, 100 - score],
                    backgroundColor: [color, '#e0e0e0'],
                    borderWidth: 0
                }]
            },
            options: {
                cutout: '75%',
                responsive: true,
                maintainAspectRatio: true,
                animation: {
                    animateRotate: true,
                    animateScale: true
                },
                plugins: {
                    tooltip: {
                        enabled: false
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    // Add to recent detections
    function addToRecentDetections(data) {
        // Create a new detection record
        const detection = {
            id: data.detection_id || Date.now(), // Use provided ID or fallback to timestamp
            type: currentMode,
            result: data.result,
            confidence: data.confidence,
            file_type: data.file_type || currentMode,
            timestamp: new Date().toLocaleTimeString()
        };
        
        // Add to the beginning of the array
        recentDetections.unshift(detection);
        
        // Limit to 5 items
        if (recentDetections.length > 5) {
            recentDetections.pop();
        }
        
        // Update UI
        updateRecentDetectionsList();
    }

    function updateRecentDetectionsList() {
        // Clear the list
        recentDetectionsList.innerHTML = '';
        
        // If no detections yet
        if (recentDetections.length === 0) {
            recentDetectionsList.innerHTML = `
                <div class="text-center text-muted">
                    No detections yet. Your detection history will appear here.
                </div>
            `;
            return;
        }
        
        // Add each detection
        recentDetections.forEach((detection, index) => {
            const detectionItem = document.createElement('div');
            const resultClass = detection.result.toLowerCase();
            
            // Create the item with the result class for styling
            detectionItem.className = `recent-item ${resultClass}`;
            // Add animation delay based on item index
            detectionItem.style.animationDelay = `${index * 0.1}s`;
            detectionItem.classList.add('animated-fade-in');
            
            detectionItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div class="recent-item-title">
                        <i class="fas fa-${detection.type === 'image' ? 'image' : 'video'}"></i>
                        ${detection.type.charAt(0).toUpperCase() + detection.type.slice(1)} Analysis
                    </div>
                    <span class="recent-item-result ${resultClass}">
                        ${detection.result.toUpperCase()}
                    </span>
                </div>
                <div class="recent-item-confidence">
                    Confidence: <strong>${detection.confidence.toFixed(1)}%</strong>
                </div>
                <div class="recent-item-time">
                    <i class="far fa-clock"></i> ${detection.timestamp}
                </div>
            `;
            recentDetectionsList.appendChild(detectionItem);
        });
    }

    // Error handling
    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.style.display = 'block';
        
        // Hide error after 5 seconds
        setTimeout(() => {
            errorAlert.style.display = 'none';
        }, 5000);
    }
    
    // Per-category ethical score input handling
    categoryScoreInputs.forEach(input => {
        input.addEventListener('input', function() {
            const score = parseInt(this.value);
            const category = this.getAttribute('data-category');
            const scoreValueElement = document.getElementById(`${category}-score-value`);
            
            if (scoreValueElement) {
                scoreValueElement.textContent = score;
                
                // Update badge color based on score
                scoreValueElement.className = "badge ms-2";
                if (score <= 3) {
                    scoreValueElement.classList.add("bg-success");
                    scoreValueElement.style.color = "#fff";
                } else if (score <= 7) {
                    scoreValueElement.classList.add("bg-warning");
                    scoreValueElement.style.color = "#000";
                } else {
                    scoreValueElement.classList.add("bg-danger");
                    scoreValueElement.style.color = "#fff";
                }
            }
            
            // Update the summary of selected categories with scores
            updateCategoriesSummary();
        });
    });
    
    // Function to update the summary of selected categories with their scores
    function updateCategoriesSummary() {
        const categoryTabs = ['general', 'emotions', 'personality', 'broad'];
        let selectedCategories = [];
        
        // Check which categories have been selected
        for (const category of categoryTabs) {
            const select = document.getElementById(`${category}-select`);
            if (select && select.value) {
                const selectedOption = select.options[select.selectedIndex];
                const reasonText = selectedOption ? selectedOption.textContent : "";
                const scoreInput = document.getElementById(`${category}-score-input`);
                const score = scoreInput ? parseInt(scoreInput.value) : 0;
                
                // Add to selected categories
                selectedCategories.push({
                    category: category,
                    reason: reasonText,
                    score: score
                });
            }
        }
        
        // Update the summary UI
        if (selectedCategories.length > 0) {
            let summaryHtml = '<ul class="list-group">';
            
            selectedCategories.forEach(item => {
                // Determine badge color based on score
                let badgeClass = "bg-warning";
                if (item.score <= 3) {
                    badgeClass = "bg-success";
                } else if (item.score >= 8) {
                    badgeClass = "bg-danger";
                }
                
                // Create list item with category, reason and score
                summaryHtml += `
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        <div>
                            <span class="fw-bold text-capitalize">${item.category}:</span> 
                            <span class="ms-2">${item.reason}</span>
                        </div>
                        <span class="badge ${badgeClass} rounded-pill">${item.score}</span>
                    </li>
                `;
            });
            
            summaryHtml += '</ul>';
            selectedCategoriesSummary.innerHTML = summaryHtml;
        } else {
            selectedCategoriesSummary.innerHTML = `
                <p class="text-center text-muted">
                    Select categories and provide scores to see a summary here.
                </p>
            `;
        }
    }
    
    // Function to update the counter that shows how many categories have been selected
    function updateSelectedCategoriesCounter() {
        const categoryTabs = ['general', 'emotions', 'personality', 'broad'];
        let selectedCount = 0;
        
        for (const category of categoryTabs) {
            const select = document.getElementById(`${category}-select`);
            if (select && select.value) {
                selectedCount++;
            }
        }
        
        // Update counter in the feedback header
        const counterElem = document.getElementById('categories-selected-counter');
        if (counterElem) {
            counterElem.textContent = selectedCount;
        }
    }
    
    // Function to reset the feedback form to its initial state
    function resetFeedbackForm() {
        // Reset dropdown selections
        reasonSelects.forEach(select => {
            select.selectedIndex = 0;
        });
        
        // Reset individual category sliders
        const categoryTabs = ['general', 'emotions', 'personality', 'broad'];
        categoryTabs.forEach(category => {
            // Reset sliders for each category to default value (5)
            const slider = document.getElementById(`${category}-score-input`);
            const valueDisplay = document.getElementById(`${category}-score-value`);
            
            if (slider) {
                slider.value = 0;
            }
            
            if (valueDisplay) {
                valueDisplay.textContent = '0';
                valueDisplay.className = "badge bg-warning ms-2";
                valueDisplay.style.color = "#000";
            }
            
            // Reset tab indicators
            const tabButton = document.getElementById(`${category}-tab`);
            if (tabButton) {
                tabButton.classList.remove('text-success');
                tabButton.innerHTML = tabButton.textContent.replace('<i class="fas fa-check-circle me-1"></i>', '');
            }
        });
        
        // Reset the selected categories counter
        updateSelectedCategoriesCounter();
        
        // Reset the categories summary section
        selectedCategoriesSummary.innerHTML = `
            <p class="text-center text-muted">
                Select categories and provide scores to see a summary here.
            </p>
        `;
        
        // Hide success message if visible
        if (feedbackSuccess) {
            feedbackSuccess.style.display = 'none';
        }
    }
    
    // Submit feedback
    submitFeedbackBtn.addEventListener('click', function() {
        // Collect feedback from all categories
        const categories = {};
        let hasSelection = false;
        
        // Check each category for selections
        const categoryTabs = ['general', 'emotions', 'personality', 'broad'];
        
        for (const category of categoryTabs) {
            const select = document.getElementById(`${category}-select`);
            if (select && select.value) {
                hasSelection = true;
                const selectedOption = select.options[select.selectedIndex];
                const reasonText = selectedOption ? selectedOption.textContent : "";
                
                // Get category-specific score from the slider
                const scoreInput = document.getElementById(`${category}-score-input`);
                const categoryScore = scoreInput ? parseInt(scoreInput.value) : 0;
                
                // Add to categories data with category-specific score
                if(parseInt(select.value) != 0){
                    categories[category] = {
                        reason_id: parseInt(select.value),
                        reason_text: reasonText,
                        ethical_score: categoryScore
                    };
                }
            }
        }
        
        // Validate form
        if (!hasSelection) {
            showError("Please select at least one reason in any category.");
            return;
        }
        
        if (!currentDetectionData) {
            showError("Detection data not found. Please try again.");
            return;
        }
        
        // Get active tab for backward compatibility
        const activeTab = document.querySelector('.nav-link.active');
        const activeCategory = activeTab ? activeTab.id.replace('-tab', '') : 'general';
        const activeSelect = document.getElementById(`${activeCategory}-select`);
        
        // Get reason data from active category for backward compatibility
        let reasonId = 0;
        let reasonText = "";
        
        if (activeSelect && activeSelect.value) {
            reasonId = parseInt(activeSelect.value);
            const selectedOption = activeSelect.options[activeSelect.selectedIndex];
            reasonText = selectedOption ? selectedOption.textContent : "";
        }
        
        // Prepare feedback data with multi-category support
        const feedbackData = {
            detection_id: currentDetectionData.detection_id,
            file_name: currentFile ? currentFile.name : 'unknown',
            file_type: currentMode,
            is_fake: true,
            confidence_score: currentDetectionData.confidence,
            categories: categories,
        };
        
        // Send feedback to server
        fetch("/api/submit-feedback", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(feedbackData)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'Failed to submit feedback. Please try again.');
                });
            }
            return response.json();
        })
        .then(data => {
            // Show success message
            feedbackSuccess.style.display = 'block';
            
            // Reset all form elements
            resetFeedbackForm();
            
            // Update the categories summary
            updateCategoriesSummary();
            
            // Hide form after 3 seconds
            setTimeout(() => {
                userFeedbackSection.style.display = 'none';
            }, 3000);
        })
        .catch(error => {
            showError(error.message);
        });
    });
});
