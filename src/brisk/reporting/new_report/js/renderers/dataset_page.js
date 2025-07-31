class DatasetPageRenderer {
    constructor(datasetData) {
        this.datasetData = datasetData;
        this.selectedFeatureIndex = 0;
    }

    render() {
        const template = document.getElementById('dataset-template').content.cloneNode(true);
        this.renderDatasetSummary(template);
        this.renderDatasetFeatures(template);
        this.renderDatasetDistribution(template);
        return template;
    }

    renderDatasetSummary(template) {
        // Set the dataset title
        const titleElement = template.querySelector('.dataset-title');
        titleElement.textContent = `Dataset: ${this.datasetData.ID}`;

        // Create summary content sections in new order
        this.renderDatasetSplits(template); // Move to top
        this.renderCombinedInfoTables(template); // Combine dataset info and target feature
        this.renderDataManagerSettings(template); // Make collapsible
        this.renderCorrelationMatrix(template);
    }

    renderCombinedInfoTables(template) {
        const summaryDiv = template.querySelector('.dataset-summary');
        
        // Insert after title and splits, before second hr
        const secondHr = summaryDiv.querySelectorAll('hr')[1];
        
        const combinedSection = document.createElement('div');
        combinedSection.className = 'combined-info-section';
        
        // Create side-by-side layout
        const infoContainer = document.createElement('div');
        infoContainer.className = 'info-container';
        
        // Left side - Dataset Information
        const datasetInfoCard = document.createElement('div');
        datasetInfoCard.className = 'info-card';
        
        const datasetTitle = document.createElement('h3');
        datasetTitle.textContent = 'Dataset Information';
        datasetTitle.className = 'info-card-title';
        
        const datasetContent = document.createElement('div');
        datasetContent.className = 'info-card-content';
        datasetContent.innerHTML = `
            <div class="info-item-compact">
                <span class="info-label-compact">Features:</span>
                <span class="info-value-compact">${this.datasetData.size[1]}</span>
            </div>
            <div class="info-item-compact">
                <span class="info-label-compact">Observations:</span>
                <span class="info-value-compact">${this.datasetData.size[0]}</span>
            </div>
        `;
        
        datasetInfoCard.appendChild(datasetTitle);
        datasetInfoCard.appendChild(datasetContent);
        
        // Right side - Target Feature
        const targetInfoCard = document.createElement('div');
        targetInfoCard.className = 'info-card';
        
        const targetTitle = document.createElement('h3');
        targetTitle.textContent = 'Target Feature';
        targetTitle.className = 'info-card-title';
        
        const targetContent = document.createElement('div');
        targetContent.className = 'info-card-content';
        
        // Create target stats from the dataset
        Object.entries(this.datasetData.target_stats).forEach(([stat, value]) => {
            const statItem = document.createElement('div');
            statItem.className = 'info-item-compact';
            statItem.innerHTML = `
                <span class="info-label-compact">${stat.charAt(0).toUpperCase() + stat.slice(1)}:</span>
                <span class="info-value-compact">${value}</span>
            `;
            targetContent.appendChild(statItem);
        });
        
        targetInfoCard.appendChild(targetTitle);
        targetInfoCard.appendChild(targetContent);
        
        infoContainer.appendChild(datasetInfoCard);
        infoContainer.appendChild(targetInfoCard);
        combinedSection.appendChild(infoContainer);
        
        secondHr.parentNode.insertBefore(combinedSection, secondHr);
    }

    renderDataManagerSettings(template) {
        const summaryDiv = template.querySelector('.dataset-summary');
        const secondHr = summaryDiv.querySelectorAll('hr')[1];
        
        // Get the actual DataManager instance
        const dataManager = window.app.reportData.data_managers[this.datasetData.data_manager_id];
        
        const dmSection = document.createElement('div');
        dmSection.className = 'collapsible-section';
        
        // Create collapsible header
        const dmHeader = document.createElement('div');
        dmHeader.className = 'collapsible-header';
        
        const headerText = document.createElement('span');
        headerText.textContent = 'DataManager Settings';
        
        const collapseIcon = document.createElement('span');
        collapseIcon.className = 'collapse-icon';
        collapseIcon.textContent = '▲'; // Start collapsed
        
        dmHeader.appendChild(headerText);
        dmHeader.appendChild(collapseIcon);
        
        // Create collapsible content with actual DataManager data
        const dmContent = document.createElement('div');
        dmContent.className = 'collapsible-content';
        
        if (dataManager) {
            dmContent.innerHTML = `
                <div class="info-item-compact">
                    <span class="info-label-compact">n_splits:</span>
                    <span class="info-value-compact">${dataManager.n_splits}</span>
                </div>
                <div class="info-item-compact">
                    <span class="info-label-compact">split_method:</span>
                    <span class="info-value-compact">${dataManager.split_method}</span>
                </div>
            `;
            
            // Add any additional DataManager properties
            Object.entries(dataManager).forEach(([key, value]) => {
                if (!['ID', 'n_splits', 'split_method'].includes(key)) {
                    const item = document.createElement('div');
                    item.className = 'info-item-compact';
                    item.innerHTML = `
                        <span class="info-label-compact">${key}:</span>
                        <span class="info-value-compact">${value}</span>
                    `;
                    dmContent.appendChild(item);
                }
            });
        } else {
            dmContent.innerHTML = '<div class="info-item-compact">DataManager not found</div>';
        }
        
        // Start collapsed
        dmContent.style.display = 'none';
        
        // Add click handler for collapse/expand
        dmHeader.addEventListener('click', () => {
            const isCollapsed = dmContent.style.display === 'none';
            dmContent.style.display = isCollapsed ? 'block' : 'none';
            collapseIcon.textContent = isCollapsed ? '▼' : '▲';
        });
        
        dmSection.appendChild(dmHeader);
        dmSection.appendChild(dmContent);
        
        secondHr.parentNode.insertBefore(dmSection, secondHr);
    }

    renderCorrelationMatrix(template) {
        const summaryDiv = template.querySelector('.dataset-summary');
        const secondHr = summaryDiv.querySelectorAll('hr')[1];
        
        const corrSection = document.createElement('div');
        corrSection.className = 'correlation-section';
        
        const corrContent = document.createElement('div');
        corrContent.className = 'correlation-content';
        
        // Render the correlation matrix plot without description
        if (this.datasetData.corr_matrix) {
            const plotDiv = document.createElement('div');
            plotDiv.className = 'correlation-plot';
            
            // Handle SVG content directly
            if (this.datasetData.corr_matrix.image.startsWith('<svg') || this.datasetData.corr_matrix.image.startsWith('<?xml')) {
                plotDiv.innerHTML = this.datasetData.corr_matrix.image;
            } else {
                plotDiv.innerHTML = '<div class="correlation-placeholder">Correlation Matrix</div>';
            }
            
            corrContent.appendChild(plotDiv);
        } else {
            corrContent.innerHTML = '<div class="correlation-placeholder">Correlation Matrix</div>';
        }
        
        corrSection.appendChild(corrContent);
        
        secondHr.parentNode.insertBefore(corrSection, secondHr);
    }

    renderDatasetSplits(template) {
        const splitsContainer = template.querySelector('.dataset-splits');
        
        // Create splits navigation
        const splitsNav = document.createElement('div');
        splitsNav.className = 'splits-nav';
        
        // Create split text elements
        this.datasetData.splits.forEach((split, index) => {
            const splitText = document.createElement('span');
            splitText.textContent = `Split ${index}`;
            splitText.className = `split-text ${index === 0 ? 'selected' : ''}`;
            splitText.dataset.split = index;
            
            splitText.addEventListener('click', () => {
                // Update active state
                splitsNav.querySelectorAll('.split-text').forEach(text => {
                    text.classList.remove('selected');
                });
                splitText.classList.add('selected');                
            });
            
            splitsNav.appendChild(splitText);
        });
        
        splitsContainer.appendChild(splitsNav);
    }

    renderDatasetFeatures(template) {
        const container = template.querySelector('.dataset-features');
        
        // Create features navigation
        const featuresNav = document.createElement('div');
        featuresNav.className = 'features-nav';
        
        const displayFeatures = this.datasetData.features;
        
        displayFeatures.forEach((feature, index) => {
            const featureText = document.createElement('span');
            featureText.textContent = feature; // Use actual feature name
            featureText.className = `feature-text ${index === this.selectedFeatureIndex ? 'selected' : ''}`;
            featureText.dataset.featureIndex = index;
            
            featureText.addEventListener('click', () => {
                this.selectedFeatureIndex = index;
                
                // Update active state
                featuresNav.querySelectorAll('.feature-text').forEach(text => {
                    text.classList.remove('selected');
                });
                featureText.classList.add('selected');
                
                // Re-render distribution section with new feature
                this.renderDatasetDistribution(template);                
            });
            
            featuresNav.appendChild(featureText);
        });
        
        container.appendChild(featuresNav);
    }

    renderDatasetDistribution(template) {
        const container = template.querySelector('.dataset-distribution');
        
        // Create the layout structure
        const distributionLayout = document.createElement('div');
        distributionLayout.className = 'distribution-layout';
        
        // Left side - feature distribution table
        const distributionTable = document.createElement('div');
        distributionTable.className = 'distribution-table-container';
        
        // Right side - feature distribution plot
        const plotContainer = document.createElement('div');
        plotContainer.className = 'distribution-plot-container';
        
        // Render feature distribution if available
        if (this.datasetData.feature_distributions && this.datasetData.feature_distributions.length > 0) {
            const selectedDistribution = this.datasetData.feature_distributions[this.selectedFeatureIndex] || this.datasetData.feature_distributions[0];
            
            // Render table with description
            const tableTitle = document.createElement('h3');
            tableTitle.textContent = selectedDistribution.table.name;
            tableTitle.className = 'section-title';
            
            // Create table using TableRenderer
            const tableRenderer = new TableRenderer(selectedDistribution.table);
            const tableElement = tableRenderer.render();
            
            distributionTable.appendChild(tableTitle);
            distributionTable.appendChild(tableElement);
            
            // Render plot with description
            const plotRenderer = new PlotRenderer(selectedDistribution.plot);
            const plotElement = plotRenderer.render();
            
            plotContainer.appendChild(plotElement);
        } else {
            // Fallback content
            const tableTitle = document.createElement('h3');
            tableTitle.textContent = 'Feature Distribution';
            tableTitle.className = 'section-title';
            
            const noDataMessage = document.createElement('p');
            noDataMessage.textContent = 'No feature distribution data available';
            noDataMessage.className = 'no-data-message';
            
            distributionTable.appendChild(tableTitle);
            distributionTable.appendChild(noDataMessage);
            
            const plotPlaceholder = document.createElement('div');
            plotPlaceholder.className = 'plot-placeholder';
            plotPlaceholder.textContent = 'Distribution Plot';
            
            plotContainer.appendChild(plotPlaceholder);
        }
        
        distributionLayout.appendChild(distributionTable);
        distributionLayout.appendChild(plotContainer);
        container.appendChild(distributionLayout);
    }
}