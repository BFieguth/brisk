class DatasetPageRenderer {
    constructor(datasetData) {
        this.datasetData = datasetData;
        this.selectedFeatureIndex = 0;
        this.selectedSplitIndex = 0;
        this.selectedSplitId = this.datasetData.splits[this.selectedSplitIndex];
    }

    render() {
        const template = document.getElementById('dataset-template').content.cloneNode(true);
        this.renderDatasetSummary(template);
        this.renderDatasetFeatures(template);
        this.renderDatasetDistribution(template);
        this.renderDataManagerSettings(template);
        return template;
    }

    renderDatasetSummary(template) {
        const titleElement = template.querySelector('.dataset-title');
        titleElement.textContent = `Dataset: ${this.datasetData.ID}`;
        this.renderDatasetSplits(template);
        this.renderCombinedInfoTables(template);
        this.renderCorrelationMatrix(template);
    }

    renderCombinedInfoTables(template) {
        const datasetInfoContent = template.querySelector('.dataset-info-content');
        datasetInfoContent.innerHTML = '';

        const targetInfoContent = template.querySelector('.target-info-content');        
        targetInfoContent.innerHTML = '';

        const currentSplitSizes = this.datasetData.split_sizes[this.selectedSplitId] || {};
        const currentTargetStats = this.datasetData.split_target_stats[this.selectedSplitId] || {};
        
        
        // Populate dataset information
        if (currentSplitSizes.features && currentSplitSizes.total_obs) {
            datasetInfoContent.innerHTML = `
                <div class="info-item-compact">
                    <span class="info-label-compact">Features:</span>
                    <span class="info-value-compact">${currentSplitSizes.features}</span>
                </div>
                <div class="info-item-compact">
                    <span class="info-label-compact">Total Observations:</span>
                    <span class="info-value-compact">${currentSplitSizes.total_obs}</span>
                </div>
                <div class="info-item-compact">
                    <span class="info-label-compact">Train Observations:</span>
                    <span class="info-value-compact">${currentSplitSizes.train_obs || 'N/A'}</span>
                </div>
                <div class="info-item-compact">
                    <span class="info-label-compact">Test Observations:</span>
                    <span class="info-value-compact">${currentSplitSizes.test_obs || 'N/A'}</span>
                </div>
            `;
        } else {
            datasetInfoContent.innerHTML = '<div class="info-item-compact">No dataset information available</div>';
        }
        
        // Populate target feature stats
        if (Object.keys(currentTargetStats).length > 0) {
            Object.entries(currentTargetStats).forEach(([stat, value]) => {
                const statItem = document.createElement('div');
                statItem.className = 'info-item-compact';
                statItem.innerHTML = `
                    <span class="info-label-compact">${stat.charAt(0).toUpperCase() + stat.slice(1)}:</span>
                    <span class="info-value-compact">${value}</span>
                `;
                targetInfoContent.appendChild(statItem);
            });
        } else {
            targetInfoContent.innerHTML = '<div class="info-item-compact">No target stats available</div>';
        }
    }

    renderDataManagerSettings(template) {
        const dmContent = template.querySelector('.collapsible-content');
        dmContent.innerHTML = '';
        if (!dmContent) return;
        
        const dataManager = window.app.reportData.data_managers[this.datasetData.data_manager_id];
                
        if (dataManager) {
            Object.entries(dataManager).forEach(([key, value]) => {
                if (!['ID'].includes(key)) {
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
    }

    renderCorrelationMatrix(template) {
        const corrPlot = template.querySelector('.correlation-plot');        
        corrPlot.innerHTML = '';
        
        const currentCorrMatrix = this.datasetData.split_corr_matrices[this.selectedSplitId];
        
        if (currentCorrMatrix && currentCorrMatrix.image) {
            corrPlot.innerHTML = currentCorrMatrix.image;
        } else {
            corrPlot.innerHTML = '<div class="correlation-placeholder">Correlation Matrix</div>';
        }
    }

    renderDatasetSplits(template) {
        const splitsContainer = template.querySelector('.dataset-splits');
        const splitsNav = splitsContainer.querySelector('.splits-nav');
        splitsNav.innerHTML = '';
        
        this.datasetData.splits.forEach((splitId, index) => {
            const splitText = document.createElement('span');
            splitText.textContent = `Split ${index}`;
            splitText.className = `split-text ${index === this.selectedSplitIndex ? 'selected' : ''}`;
            splitText.dataset.split = index;
            splitText.dataset.splitId = splitId;
            
            splitsNav.appendChild(splitText);
        });
    }

    renderDatasetFeatures(template) {
        const container = template.querySelector('.dataset-features');
        const featuresNav = container.querySelector('.features-nav');
        featuresNav.innerHTML = '';
        
        const displayFeatures = this.datasetData.features;
        
        displayFeatures.forEach((feature, index) => {
            const featureText = document.createElement('span');
            featureText.textContent = feature;
            featureText.className = `feature-text ${index === this.selectedFeatureIndex ? 'selected' : ''}`;
            featureText.dataset.featureIndex = index;
            
            featuresNav.appendChild(featureText);
        });
    }

    renderDatasetDistribution(template) {
        const tableContainer = template.querySelector('.distribution-table-content');
        tableContainer.innerHTML = '';

        const plotContainer = template.querySelector('.distribution-plot-content');
        plotContainer.innerHTML = '';

        const tableTitle = template.querySelector('.distribution-table-title');
        
        
        const currentFeatureDistributions = this.datasetData.split_feature_distributions[this.selectedSplitId];
        
        if (currentFeatureDistributions && currentFeatureDistributions.length > 0) {
            const selectedDistribution = currentFeatureDistributions[this.selectedFeatureIndex] || currentFeatureDistributions[0];
            
            tableTitle.textContent = selectedDistribution.tables[0].name;
            
            for (const table of selectedDistribution.tables) {
                const tableRenderer = new TableRenderer(table);
                const tableElement = tableRenderer.render();
                tableContainer.appendChild(tableElement);
            }

            const plotRenderer = new PlotRenderer(selectedDistribution.plot);
            const plotElement = plotRenderer.render();
            plotContainer.appendChild(plotElement);
        } else {
            tableTitle.textContent = 'Feature Distribution';
            
            const noDataMessage = document.createElement('p');
            noDataMessage.textContent = 'No feature distribution data available';
            noDataMessage.className = 'no-data-message';
            tableContainer.appendChild(noDataMessage);
            
            const plotPlaceholder = document.createElement('div');
            plotPlaceholder.className = 'plot-placeholder';
            plotPlaceholder.textContent = 'Distribution Plot';
            plotContainer.appendChild(plotPlaceholder);
        }
    }
}